# Unit tests for color correction module

if(NOT TESTING_ENABLED)
    return()
endif()

# Test sources
file(GLOB_RECURSE TEST_SOURCES "*_test.cpp")
file(GLOB_RECURSE INTEGRATION_TEST_SOURCES "Test_Integration_*.cpp")
file(GLOB_RECURSE INTERNAL_TEST_SOURCES "*_internal_test.cpp")

# Unit tests
if(TEST_SOURCES)
    add_executable(color_correction_unit_tests ${TEST_SOURCES})
    target_link_libraries(color_correction_unit_tests 
        PRIVATE 
            color_correction
            GTest::gtest_main
            ${OpenCV_LIBS}
    )
    add_test(NAME ColorCorrectionUnitTests COMMAND color_correction_unit_tests)
endif()

# Integration tests
if(INTEGRATION_TEST_SOURCES)
    add_executable(color_correction_integration_tests ${INTEGRATION_TEST_SOURCES})
    target_link_libraries(color_correction_integration_tests 
        PRIVATE 
            color_correction
            GTest::gtest_main
            ${OpenCV_LIBS}
    )
    add_test(NAME ColorCorrectionIntegrationTests COMMAND color_correction_integration_tests)
endif()

# Internal (white-box) tests
if(INTERNAL_TEST_SOURCES)
    add_executable(color_correction_internal_tests ${INTERNAL_TEST_SOURCES})
    target_link_libraries(color_correction_internal_tests 
        PRIVATE 
            color_correction_internal
            GTest::gtest_main
            ${OpenCV_LIBS}
    )
    add_test(NAME ColorCorrectionInternalTests COMMAND color_correction_internal_tests)
endif()