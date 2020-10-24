add_library(Utility INTERFACE)

find_package(Boost REQUIRED)

target_link_libraries(Utility INTERFACE ${Boost_LIBRARIES})

target_include_directories(Utility INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(Utility INTERFACE ${Boost_INCLUDE_DIRS})

file(GLOB UtilitySourcesLocal
  CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/include/Utility/*.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/Utility/*.cpp
)
set(UtilitySources ${UtilitySourcesLocal} CACHE INTERNAL "")
