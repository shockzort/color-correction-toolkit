#pragma once

#include "IConfiguration.hpp"
#include <shared/types/Common.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <functional>
#include <memory>

namespace ColorCorrection::Interface {

// Forward declarations for domain objects
namespace Domain {
    class DetectionResult;
    class CalibrationData;
    class ColorPatch;
}

// Progress callback: (current_step, total_steps, description)
using ProgressCallback = std::function<void(int, int, const std::string&)>;

// Detection callback: (detection_result)
using DetectionCallback = std::function<void(const Domain::DetectionResult&)>;

enum class CalibrationStage {
    PREPROCESSING,
    DETECTION,
    COLOR_EXTRACTION,
    MATRIX_CALCULATION,
    VALIDATION,
    SERIALIZATION,
    COMPLETED,
    FAILED
};

struct CalibrationProgress {
    CalibrationStage stage = CalibrationStage::PREPROCESSING;
    int currentStep = 0;
    int totalSteps = 6;
    float percentage = 0.0f;
    std::string currentOperation;
    std::string lastError;
    
    bool isCompleted() const { return stage == CalibrationStage::COMPLETED; }
    bool hasFailed() const { return stage == CalibrationStage::FAILED; }
};

class ICalibrationManager {
  public:
    virtual ~ICalibrationManager() = default;

    // Main calibration workflow
    virtual bool calibrateFromImage(const Types::Image& image, 
                                   const std::string& outputPath = "") = 0;
    
    virtual bool calibrateFromVideo(const std::string& videoPath,
                                   const std::string& outputPath = "",
                                   int maxFrames = 10) = 0;
    
    virtual bool calibrateFromImageSequence(const std::vector<Types::Image>& images,
                                           const std::string& outputPath = "") = 0;

    // Step-by-step calibration (for advanced users)
    virtual Domain::DetectionResult detectColorChecker(const Types::Image& image) = 0;
    
    virtual std::vector<Domain::ColorPatch> extractColors(const Types::Image& image,
                                                         const Domain::DetectionResult& detection) = 0;
    
    virtual bool calculateCorrectionMatrix(const std::vector<Domain::ColorPatch>& patches,
                                         const std::string& outputPath = "") = 0;

    // Validation and quality assessment
    virtual bool validateCalibration(const std::string& calibrationPath,
                                   const Types::Image& testImage = Types::Image()) const = 0;
    
    virtual float assessCalibrationQuality(const std::string& calibrationPath) const = 0;

    // Calibration data management
    virtual bool saveCalibrationData(const Domain::CalibrationData& data,
                                   const std::string& filename) const = 0;
    
    virtual std::unique_ptr<Domain::CalibrationData> loadCalibrationData(
        const std::string& filename) const = 0;

    // Configuration and settings
    virtual void setConfiguration(std::shared_ptr<IConfiguration> config) = 0;
    virtual std::shared_ptr<IConfiguration> getConfiguration() const = 0;

    // Progress monitoring
    virtual void setProgressCallback(ProgressCallback callback) = 0;
    virtual void setDetectionCallback(DetectionCallback callback) = 0;
    virtual CalibrationProgress getProgress() const = 0;

    // Batch operations
    virtual bool calibrateMultipleImages(const std::vector<std::string>& imagePaths,
                                       const std::string& outputDirectory) = 0;

    // Debugging and visualization
    virtual Types::Image visualizeDetection(const Types::Image& image,
                                           const Domain::DetectionResult& detection) const = 0;
    
    virtual Types::Image createQualityReport(const Domain::CalibrationData& data) const = 0;

    // Factory methods for different calibration strategies
    enum class CalibrationStrategy {
        AUTOMATIC,      // Fully automatic with fallbacks
        CONSERVATIVE,   // High quality thresholds, may fail more often
        AGGRESSIVE,     // Lower thresholds, accepts lower quality
        MANUAL          // User-guided calibration
    };
    
    virtual void setCalibrationStrategy(CalibrationStrategy strategy) = 0;
    virtual CalibrationStrategy getCalibrationStrategy() const = 0;

    // Error handling
    virtual std::string getLastError() const = 0;
    virtual void clearErrors() = 0;
    virtual bool hasErrors() const = 0;
};

// Factory functions
std::unique_ptr<ICalibrationManager> createCalibrationManager(
    std::shared_ptr<IConfiguration> config = nullptr);

std::unique_ptr<ICalibrationManager> createCalibrationManager(
    ICalibrationManager::CalibrationStrategy strategy);

}  // namespace ColorCorrection::Interface