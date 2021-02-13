add_executable(UtilityTest)

find_package(Boost REQUIRED COMPONENTS unit_test_framework)

target_link_libraries(UtilityTest PRIVATE Boost::unit_test_framework)
target_link_libraries(UtilityTest PRIVATE Utility)

target_include_directories(UtilityTest PRIVATE ${Boost_INCLUDE_DIRS})

file(GLOB UtilityTestSources
  CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/*.[hcm]pp
  ${CMAKE_CURRENT_SOURCE_DIR}/../Utility/include/Utility/*.[hcm]pp
)
target_sources(UtilityTest PRIVATE ${UtilityTestSources})
