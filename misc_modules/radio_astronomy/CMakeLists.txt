cmake_minimum_required(VERSION 3.13)
project(radio_astronomy)

file(GLOB SRC "src/*.cpp")

include(${SDRPP_MODULE_CMAKE})