cmake_minimum_required(VERSION 3.10)

project(Utility)

add_library(Utility INTERFACE)

find_package(Boost REQUIRED)

target_link_libraries(Utility INTERFACE ${Boost_LIBRARIES})

target_include_directories(Utility INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(Utility INTERFACE ${Boost_INCLUDE_DIRS})

file(GLOB UtilitySources
  CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/include/Utility/*.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/Utility/*.cpp
)

option(UTILITY_TEST "Build Evolution tests" OFF)
if (UTILITY_TEST)
  add_subdirectory(UtilityTest)
endif ()
