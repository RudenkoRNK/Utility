cmake_minimum_required(VERSION 3.10)

project(Utility)

add_subdirectory(Utility)

option(UTILITY_TEST "Build Evolution tests" OFF)
if (UTILITY_TEST)
  add_subdirectory(UtilityTest)
endif ()
