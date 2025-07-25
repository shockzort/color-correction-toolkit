# Applications and CLI tools

# Calibration application
add_executable(color_calibrator calibration_app.cpp)
target_link_libraries(color_calibrator 
    PRIVATE 
        color_correction
        ${OpenCV_LIBS}
)

# Correction application  
add_executable(color_corrector correction_app.cpp)
target_link_libraries(color_corrector 
    PRIVATE 
        color_correction
        ${OpenCV_LIBS}
)