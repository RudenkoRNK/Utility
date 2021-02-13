add_library(Utility INTERFACE)

target_include_directories(Utility INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(Utility INTERFACE cxx_std_20)
