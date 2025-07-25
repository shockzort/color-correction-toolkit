#pragma once

// Main public API header - includes all interfaces
#include "ICalibrationManager.hpp"
#include "IColorCorrector.hpp"
#include "IConfiguration.hpp"

// Common types for public API
#include <shared/types/Common.hpp>

// Version information
namespace ColorCorrection {

constexpr int VERSION_MAJOR = 1;
constexpr int VERSION_MINOR = 0;
constexpr int VERSION_PATCH = 0;
constexpr const char* VERSION_STRING = "1.0.0";

// Simplified API for common use cases
namespace SimpleAPI {

// One-function calibration: image path -> calibration file
bool calibrateFromImage(const std::string& imagePath, 
                       const std::string& calibrationOutputPath);

// One-function correction: input path + calibration -> output path
bool correctImage(const std::string& inputPath,
                 const std::string& calibrationPath,
                 const std::string& outputPath);

// Batch correction with progress callback
bool correctImageBatch(const std::vector<std::string>& inputPaths,
                      const std::string& outputDirectory,
                      const std::string& calibrationPath,
                      Interface::BatchProgressCallback progressCallback = nullptr);

// Video correction
bool correctVideo(const std::string& inputVideoPath,
                 const std::string& outputVideoPath,
                 const std::string& calibrationPath);

// Quality assessment
float assessCalibrationQuality(const std::string& calibrationPath);

// Check if calibration file is valid
bool isCalibrationValid(const std::string& calibrationPath);

}  // namespace SimpleAPI

}  // namespace ColorCorrection