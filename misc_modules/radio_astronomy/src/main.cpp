#include <algorithm>
#include <atomic>
#include <cmath>
#include <mutex>
#include <string>
#include <vector>

#include <fftw3.h>
#include <imgui.h>
#include <module.h>
#include <core.h>
#include <dsp/buffer/reshaper.h>
#include <dsp/sink/handler_sink.h>
#include <gui/gui.h>
#include <gui/style.h>
#include <signal_path/signal_path.h>

SDRPP_MOD_INFO{
    /* Name:            */ "radio_astronomy",
    /* Description:     */ "Background-corrected spectrum integration for radio astronomy",
    /* Author:          */ "SDR++ contributors",
    /* Version:         */ 0, 1, 0,
    /* Max instances    */ 1
};

ConfigManager config;

class RadioAstronomyModule : public ModuleManager::Instance {
public:
    RadioAstronomyModule(std::string instanceName) : name(std::move(instanceName)) {
        loadConfig();

        fftIn = (fftwf_complex*)fftwf_malloc(FFT_SIZE * sizeof(fftwf_complex));
        fftOut = (fftwf_complex*)fftwf_malloc(FFT_SIZE * sizeof(fftwf_complex));
        fftPlan = fftwf_plan_dft_1d(FFT_SIZE, fftIn, fftOut, FFTW_FORWARD, FFTW_ESTIMATE);
        powerFrame.resize(FFT_SIZE);

        reshape.init(&iqStream, FFT_SIZE, 0);
        handler.init(&reshape.out, dataHandler, this);

        redrawHandler.ctx = this;
        redrawHandler.handler = fftRedraw;
        gui::menu.registerEntry(name, menuHandler, this, NULL);
        gui::waterfall.onFFTRedraw.bindHandler(&redrawHandler);
        startDSP();
    }

    ~RadioAstronomyModule() {
        gui::menu.removeEntry(name);
        gui::waterfall.onFFTRedraw.unbindHandler(&redrawHandler);
        stopDSP();
        fftwf_destroy_plan(fftPlan);
        fftwf_free(fftIn);
        fftwf_free(fftOut);
    }

    void postInit() {}

    void enable() {
        enabled = true;
        startDSP();
    }

    void disable() {
        stopDSP();
        enabled = false;
    }

    bool isEnabled() { return enabled; }

private:
    static constexpr int FFT_SIZE = 1024;
    static constexpr double HYDROGEN_LINE_HZ = 1420405751.0;

    void loadConfig() {
        config.acquire();
        if (config.conf[name].contains("intermediateAverage")) {
            intermediateAverage = std::clamp((int)config.conf[name]["intermediateAverage"], 1, 10000);
        }
        if (config.conf[name].contains("dynamicAverage")) {
            dynamicAverage = std::clamp((int)config.conf[name]["dynamicAverage"], 1, 10000);
        }
        if (config.conf[name].contains("correctionRange")) {
            correctionRange = std::max((float)config.conf[name]["correctionRange"], 0.1f);
        }
        if (config.conf[name].contains("showHydrogenLine")) {
            showHydrogenLine = config.conf[name]["showHydrogenLine"];
        }
        config.release();
    }

    void saveConfig() {
        config.acquire();
        config.conf[name]["intermediateAverage"] = intermediateAverage;
        config.conf[name]["dynamicAverage"] = dynamicAverage;
        config.conf[name]["correctionRange"] = correctionRange;
        config.conf[name]["showHydrogenLine"] = showHydrogenLine;
        config.release(true);
    }

    void startDSP() {
        if (streamBound) { return; }
        sigpath::iqFrontEnd.bindIQStream(&iqStream);
        streamBound = true;
        reshape.start();
        handler.start();
    }

    void stopDSP() {
        if (!streamBound) { return; }
        handler.stop();
        reshape.stop();
        sigpath::iqFrontEnd.unbindIQStream(&iqStream);
        streamBound = false;
    }

    void resetAccumulatorLocked() {
        intermediateSum.assign(FFT_SIZE, 0.0f);
        movingSum.assign(FFT_SIZE, 0.0f);
        history.clear();
        values.clear();
        intermediateCount = 0;
        historyIndex = 0;
        historyCount = 0;
        observationCount = 0;
    }

    void acquireBackground() {
        std::lock_guard<std::mutex> lock(stateMutex);
        baseline.clear();
        resetAccumulatorLocked();
        backgroundCount = 0;
        backgroundSampling = true;
        observing = false;
        processing.store(true);
    }

    void startIntegration() {
        std::lock_guard<std::mutex> lock(stateMutex);
        if (backgroundSampling || baseline.empty()) { return; }
        observing = true;
        processing.store(true);
    }

    void stopIntegration() {
        std::lock_guard<std::mutex> lock(stateMutex);
        observing = false;
        processing.store(backgroundSampling);
    }

    void resetIntegration() {
        std::lock_guard<std::mutex> lock(stateMutex);
        const bool resume = observing && !baseline.empty();
        resetAccumulatorLocked();
        observing = resume;
        processing.store(backgroundSampling || observing);
    }

    void setParameters(int newIntermediateAverage, int newDynamicAverage, float newCorrectionRange) {
        std::lock_guard<std::mutex> lock(stateMutex);
        intermediateAverage = std::clamp(newIntermediateAverage, 1, 10000);
        dynamicAverage = std::clamp(newDynamicAverage, 1, 10000);
        correctionRange = std::max(newCorrectionRange, 0.1f);
        baseline.clear();
        backgroundSampling = false;
        observing = false;
        backgroundCount = 0;
        processing.store(false);
        resetAccumulatorLocked();
        saveConfig();
    }

    void processPowerFrame(const float* power) {
        std::lock_guard<std::mutex> lock(stateMutex);
        if (!backgroundSampling && !observing) { return; }

        if (intermediateSum.size() != FFT_SIZE) {
            intermediateSum.assign(FFT_SIZE, 0.0f);
        }
        for (int i = 0; i < FFT_SIZE; i++) {
            intermediateSum[i] += power[i];
        }
        intermediateCount++;
        if (intermediateCount < intermediateAverage) { return; }

        std::vector<float> amplitudeFrame(FFT_SIZE);
        for (int i = 0; i < FFT_SIZE; i++) {
            amplitudeFrame[i] = std::sqrt((intermediateSum[i] / intermediateCount) / FFT_SIZE);
        }
        std::fill(intermediateSum.begin(), intermediateSum.end(), 0.0f);
        intermediateCount = 0;

        if (movingSum.size() != FFT_SIZE) {
            movingSum.assign(FFT_SIZE, 0.0f);
        }
        if (history.size() < (size_t)dynamicAverage) {
            for (int i = 0; i < FFT_SIZE; i++) {
                movingSum[i] += amplitudeFrame[i];
            }
            history.push_back(std::move(amplitudeFrame));
            historyCount++;
        }
        else {
            std::vector<float>& oldest = history[historyIndex];
            for (int i = 0; i < FFT_SIZE; i++) {
                movingSum[i] += amplitudeFrame[i] - oldest[i];
            }
            oldest = std::move(amplitudeFrame);
            historyIndex = (historyIndex + 1) % dynamicAverage;
        }

        if (backgroundSampling) {
            backgroundCount = historyCount;
            if (historyCount >= dynamicAverage) {
                baseline.resize(FFT_SIZE);
                for (int i = 0; i < FFT_SIZE; i++) {
                    baseline[i] = movingSum[i] / historyCount;
                }
                backgroundSampling = false;
                processing.store(false);
                resetAccumulatorLocked();
            }
            return;
        }

        observationCount++;
        values.resize(FFT_SIZE);
        for (int i = 0; i < FFT_SIZE; i++) {
            const float observation = movingSum[i] / historyCount;
            values[i] = 20.0f * std::log10(std::max(observation, 1e-20f) / std::max(baseline[i], 1e-20f));
        }
    }

    static void dataHandler(dsp::complex_t* data, int count, void* ctx) {
        RadioAstronomyModule* self = (RadioAstronomyModule*)ctx;
        if (!self->processing.load() || count != FFT_SIZE) { return; }

        for (int i = 0; i < FFT_SIZE; i++) {
            self->fftIn[i][0] = data[i].re;
            self->fftIn[i][1] = data[i].im;
        }
        fftwf_execute(self->fftPlan);

        const int half = FFT_SIZE / 2;
        for (int i = 0; i < FFT_SIZE; i++) {
            const int source = (i + half) % FFT_SIZE;
            const float real = self->fftOut[source][0];
            const float imag = self->fftOut[source][1];
            self->powerFrame[i] = (real * real) + (imag * imag);
        }
        self->processPowerFrame(self->powerFrame.data());
    }

    static void menuHandler(void* ctx) {
        RadioAstronomyModule* self = (RadioAstronomyModule*)ctx;
        bool hasBackground;
        bool acquiring;
        bool running;
        int backgroundProgress;
        int integrationProgress;
        {
            std::lock_guard<std::mutex> lock(self->stateMutex);
            hasBackground = !self->baseline.empty();
            acquiring = self->backgroundSampling;
            running = self->observing;
            backgroundProgress = self->backgroundCount;
            integrationProgress = self->observationCount;
        }

        if (ImGui::Button(("Acquire Background##" + self->name).c_str())) {
            self->acquireBackground();
        }
        ImGui::SameLine();
        if (!hasBackground || running) { style::beginDisabled(); }
        if (ImGui::Button(("Start##" + self->name).c_str())) {
            self->startIntegration();
        }
        if (!hasBackground || running) { style::endDisabled(); }
        ImGui::SameLine();
        if (!running) { style::beginDisabled(); }
        if (ImGui::Button(("Stop##" + self->name).c_str())) {
            self->stopIntegration();
        }
        if (!running) { style::endDisabled(); }
        ImGui::SameLine();
        if (ImGui::Button(("Reset##" + self->name).c_str())) {
            self->resetIntegration();
        }

        if (acquiring) {
            ImGui::Text("Acquiring background: %d/%d", backgroundProgress, self->dynamicAverage);
        }
        else if (hasBackground) {
            ImGui::Text("Integration: %d%s", integrationProgress, running ? " (running)" : " (stopped)");
        }
        else {
            ImGui::TextUnformatted("Background required");
        }

        int newIntermediateAverage = self->intermediateAverage;
        int newDynamicAverage = self->dynamicAverage;
        float newCorrectionRange = self->correctionRange;
        bool changed = false;
        ImGui::LeftLabel("Intermediate Average");
        ImGui::FillWidth();
        changed |= ImGui::InputInt(("##radio_astronomy_intermediate_" + self->name).c_str(), &newIntermediateAverage, 1, 10);

        ImGui::LeftLabel("Dynamic Average");
        ImGui::FillWidth();
        changed |= ImGui::InputInt(("##radio_astronomy_dynamic_" + self->name).c_str(), &newDynamicAverage, 1, 10);

        ImGui::LeftLabel("Correction Range (dB)");
        ImGui::FillWidth();
        changed |= ImGui::InputFloat(("##radio_astronomy_range_" + self->name).c_str(), &newCorrectionRange, 0.5f, 1.0f, "%.1f");

        bool newShowHydrogenLine = self->showHydrogenLine;
        if (ImGui::Checkbox(("Hydrogen Line##" + self->name).c_str(), &newShowHydrogenLine)) {
            {
                std::lock_guard<std::mutex> lock(self->stateMutex);
                self->showHydrogenLine = newShowHydrogenLine;
            }
            self->saveConfig();
        }
        if (changed) {
            self->setParameters(newIntermediateAverage, newDynamicAverage, newCorrectionRange);
        }
    }

    static void fftRedraw(ImGui::WaterFall::FFTRedrawArgs args, void* ctx) {
        RadioAstronomyModule* self = (RadioAstronomyModule*)ctx;
        std::vector<float> spectrum;
        float range;
        bool drawHydrogen;
        {
            std::lock_guard<std::mutex> lock(self->stateMutex);
            spectrum = self->values;
            range = self->correctionRange;
            drawHydrogen = self->showHydrogenLine;
        }

        if (drawHydrogen && HYDROGEN_LINE_HZ >= args.lowFreq && HYDROGEN_LINE_HZ <= args.highFreq) {
            const float x = args.min.x + (HYDROGEN_LINE_HZ - args.lowFreq) * args.freqToPixelRatio;
            args.window->DrawList->AddLine(ImVec2(x, args.min.y), ImVec2(x, args.max.y), IM_COL32(255, 235, 0, 255), 1.5f);
        }
        if (spectrum.size() != FFT_SIZE) { return; }

        const double sampleRate = sigpath::iqFrontEnd.getEffectiveSamplerate();
        const double center = gui::waterfall.getCenterFrequency();
        const double fftLow = center - (sampleRate / 2.0);
        const float zeroY = (args.min.y + args.max.y) / 2.0f;
        const float scale = ((args.max.y - args.min.y) / 2.0f) / range;
        const ImU32 color = IM_COL32(255, 166, 66, 255);
        args.window->DrawList->AddLine(ImVec2(args.min.x, zeroY), ImVec2(args.max.x, zeroY), color, 1.0f);

        ImVec2 previous;
        bool havePrevious = false;
        for (int i = 0; i < FFT_SIZE; i++) {
            const double frequency = fftLow + ((double)i / FFT_SIZE) * sampleRate;
            if (frequency < args.lowFreq || frequency > args.highFreq) { continue; }
            const float x = args.min.x + (frequency - args.lowFreq) * args.freqToPixelRatio;
            const float value = std::clamp(spectrum[i], -range, range);
            const ImVec2 point(x, zeroY - (value * scale));
            if (havePrevious) {
                args.window->DrawList->AddLine(previous, point, color, 1.5f);
            }
            previous = point;
            havePrevious = true;
        }

        const float textOffset = 8.0f * style::uiScale;
        for (float line = range; line >= -range; line -= 1.0f) {
            char label[16];
            snprintf(label, sizeof(label), "%+.0f", line);
            const float y = zeroY - (line * scale);
            args.window->DrawList->AddText(ImVec2(args.max.x + textOffset, y - ImGui::CalcTextSize(label).y / 2.0f), color, label);
        }
    }

    std::string name;
    bool enabled = true;
    bool streamBound = false;
    std::atomic<bool> processing = false;

    dsp::stream<dsp::complex_t> iqStream;
    dsp::buffer::Reshaper<dsp::complex_t> reshape;
    dsp::sink::Handler<dsp::complex_t> handler;
    fftwf_complex* fftIn = nullptr;
    fftwf_complex* fftOut = nullptr;
    fftwf_plan fftPlan = nullptr;
    std::vector<float> powerFrame;

    std::mutex stateMutex;
    std::vector<float> baseline;
    std::vector<float> intermediateSum;
    std::vector<float> movingSum;
    std::vector<std::vector<float>> history;
    std::vector<float> values;
    int intermediateAverage = 10;
    int dynamicAverage = 10;
    float correctionRange = 3.0f;
    int intermediateCount = 0;
    int historyIndex = 0;
    int historyCount = 0;
    int backgroundCount = 0;
    int observationCount = 0;
    bool backgroundSampling = false;
    bool observing = false;
    bool showHydrogenLine = true;

    EventHandler<ImGui::WaterFall::FFTRedrawArgs> redrawHandler;
};

MOD_EXPORT void _INIT_() {
    json defaults = json({});
    std::string root = (std::string)core::args["root"];
    config.setPath(root + "/radio_astronomy_config.json");
    config.load(defaults);
    config.enableAutoSave();
}

MOD_EXPORT ModuleManager::Instance* _CREATE_INSTANCE_(std::string name) {
    return new RadioAstronomyModule(name);
}

MOD_EXPORT void _DELETE_INSTANCE_(void* instance) {
    delete (RadioAstronomyModule*)instance;
}

MOD_EXPORT void _END_() {
    config.disableAutoSave();
    config.save();
}