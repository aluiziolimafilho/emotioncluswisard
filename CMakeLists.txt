cmake_minimum_required(VERSION 3.10)
project(emotioncluswisard)

set(CMAKE_CXX_STANDARD 11)

add_executable(emotioncluswisard
        cluswisard.cc
        cluswisard_bind.cc
        main.cc
        Makefile
        mnist.cc
        wisard.cc)
