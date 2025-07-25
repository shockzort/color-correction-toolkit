#pragma once

#include "IConfiguration.hpp"
#include <shared/types/Common.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <functional>
#include <memory>

namespace ColorCorrection::Interface {

// Forward declarations
namespace Domain {
    class CalibrationData;
    class CorrectionMatrix;
}

// Progress callback for batch operations: (current_item, total_items, current_file)
using BatchProgressCallback = std::function<void(int, int, const std::string&)>;

enum class CorrectionMethod {
    MATRIX_TRANSFORM,  // Basic 3x3 matrix transformation
    LUT_BASED,         // Lookup table optimization
    GPU_ACCELERATED,   // GPU-based processing
    HYBRID            // Automatic selection based on image size
};

enum class ColorSpace {
    SRGB,
    LINEAR_RGB,
    LAB,
    XYZ
};

struct CorrectionOptions {
    CorrectionMethod method = CorrectionMethod::HYBRID;
    ColorSpace inputColorSpace = ColorSpace::SRGB;
    ColorSpace outputColorSpace = ColorSpace::SRGB;
    bool preserveAlpha = true;
    bool clampValues = true;
    float qualityFactor = 1.0f;  // 0.1 = fast, 1.0 = full quality
    
    // GPU-specific options
    bool useGPU = true;
    int gpuDeviceId = -1;  // -1 = auto-select
    
    // LUT-specific options
    int lutResolution = 256;
    bool cacheLUT = true;
};

struct CorrectionResult {
    bool success = false;
    std::string errorMessage;
    Types::Image correctedImage;
    CorrectionMethod usedMethod = CorrectionMethod::MATRIX_TRANSFORM;
    float processingTimeMs = 0.0f;
    float memoryUsageMB = 0.0f;
    
    // Quality metrics (if available)
    float estimatedDeltaE = -1.0f;
    Types::ConfidenceScore qualityScore = Types::ConfidenceScore::fromValue(0.0f);
};

class IColorCorrector {
  public:
    virtual ~IColorCorrector() = default;

    // Core correction functionality
    virtual CorrectionResult correctImage(const Types::Image& input,
                                        const std::string& calibrationPath,
                                        const CorrectionOptions& options = {}) = 0;
    
    virtual CorrectionResult correctImage(const Types::Image& input,
                                        const Domain::CalibrationData& calibrationData,
                                        const CorrectionOptions& options = {}) = 0;

    // Batch processing
    virtual bool correctImageBatch(const std::vector<std::string>& inputPaths,
                                 const std::string& outputDirectory,
                                 const std::string& calibrationPath,
                                 const CorrectionOptions& options = {}) = 0;
    
    virtual bool correctImageBatch(const std::vector<Types::Image>& images,
                                 std::vector<Types::Image>& correctedImages,
                                 const Domain::CalibrationData& calibrationData,
                                 const CorrectionOptions& options = {}) = 0;

    // Video processing
    virtual bool correctVideo(const std::string& inputVideoPath,
                            const std::string& outputVideoPath,
                            const std::string& calibrationPath,
                            const CorrectionOptions& options = {}) = 0;

    // Real-time processing (for streaming/preview)
    virtual bool initializeRealTimeCorrection(const std::string& calibrationPath,
                                            const CorrectionOptions& options = {}) = 0;
    
    virtual CorrectionResult correctFrameRealTime(const Types::Image& frame) = 0;
    
    virtual void finalizeRealTimeCorrection() = 0;

    // Calibration management
    virtual bool loadCalibration(const std::string& calibrationPath) = 0;
    virtual bool setCalibrationData(const Domain::CalibrationData& data) = 0;
    virtual std::unique_ptr<Domain::CalibrationData> getCurrentCalibration() const = 0;
    virtual bool hasValidCalibration() const = 0;

    // Configuration
    virtual void setConfiguration(std::shared_ptr<IConfiguration> config) = 0;
    virtual std::shared_ptr<IConfiguration> getConfiguration() const = 0;

    // Performance optimization
    virtual void precomputeLUT(int resolution = 256) = 0;
    virtual void clearLUTCache() = 0;
    virtual bool isLUTCached() const = 0;
    
    virtual bool initializeGPU(int deviceId = -1) = 0;
    virtual void releaseGPU() = 0;
    virtual bool isGPUInitialized() const = 0;

    // Quality assessment
    virtual float estimateOutputQuality(const Types::Image& input,
                                      const CorrectionOptions& options = {}) const = 0;
    
    virtual CorrectionResult testCorrection(const Types::Image& testImage,
                                          bool generateReport = false) const = 0;

    // Progress monitoring for batch operations
    virtual void setBatchProgressCallback(BatchProgressCallback callback) = 0;

    // Debugging and visualization
    virtual Types::Image visualizeCorrection(const Types::Image& input,
                                           const Types::Image& corrected) const = 0;
    
    virtual std::string generateCorrectionReport(const CorrectionResult& result) const = 0;

    // Color space utilities
    virtual Types::Image convertColorSpace(const Types::Image& input,
                                         ColorSpace from,
                                         ColorSpace to) const = 0;

    // Direct matrix operations (for advanced users)
    virtual CorrectionResult applyMatrix(const Types::Image& input,
                                       const Domain::CorrectionMatrix& matrix,
                                       const CorrectionOptions& options = {}) = 0;

    // Performance profiling
    struct PerformanceMetrics {
        float averageProcessingTimeMs = 0.0f;
        float averageMemoryUsageMB = 0.0f;
        int imagesProcessed = 0;
        CorrectionMethod preferredMethod = CorrectionMethod::MATRIX_TRANSFORM;
        std::string deviceInfo;
    };
    
    virtual PerformanceMetrics getPerformanceMetrics() const = 0;
    virtual void resetPerformanceMetrics() = 0;

    // Error handling
    virtual std::string getLastError() const = 0;
    virtual void clearErrors() = 0;
    virtual bool hasErrors() const = 0;
};

// Factory functions
std::unique_ptr<IColorCorrector> createColorCorrector(
    std::shared_ptr<IConfiguration> config = nullptr);

std::unique_ptr<IColorCorrector> createColorCorrector(
    const std::string& calibrationPath);

// Utility functions
CorrectionOptions createFastCorrectionOptions();
CorrectionOptions createHighQualityCorrectionOptions();
CorrectionOptions createGPUCorrectionOptions();

}  // namespace ColorCorrection::Interface