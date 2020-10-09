add_executable(UtilityTest)

find_package(Boost REQUIRED)

target_link_libraries(UtilityTest PRIVATE ${Boost_LIBRARIES})
target_link_libraries(UtilityTest PRIVATE Utility)

target_include_directories(UtilityTest PRIVATE ${Boost_INCLUDE_DIRS})

set_property(TARGET UtilityTest PROPERTY CXX_STANDARD 20)

file(GLOB UtilityTestSources
  CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

target_sources(UtilityTest PRIVATE ${UtilitySources})
target_sources(UtilityTest PRIVATE ${UtilityTestSources})
