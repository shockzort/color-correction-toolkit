#pragma once

#include "../domain/CorrectionMatrix.hpp"
#include "../domain/CalibrationData.hpp"
#include "../processing/ColorSpaceConverter.hpp"
#include <shared/types/Common.hpp>
#include <shared/utils/Logger.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>

namespace ColorCorrection::Internal::Correction {

class LinearCorrector {
  public:
    struct CorrectionSettings {
        bool useLinearRGB;           // Work in linear RGB space
        bool clampValues;            // Clamp output values to [0,1]
        bool preserveAlpha;          // Preserve alpha channel if present
        float qualityFactor;         // Quality vs speed trade-off (0.1-1.0)
        
        // Error handling
        bool enableErrorRecovery;    // Fallback to identity on errors
        bool validateInputs;         // Validate input images/matrices
        
        // Performance settings
        bool enableParallelProcessing;
        int numThreads;                 // 0 = auto-detect
        
        // Output format options
        bool maintainInputType;      // Keep same bit depth as input
        bool enableDithering;       // Dithering for 8-bit output
        
        // Quality validation
        bool performQualityCheck;   // Check output quality
        float minQualityThreshold;   // Minimum acceptable quality
        
        CorrectionSettings()
            : useLinearRGB(true)
            , clampValues(true)
            , preserveAlpha(true)
            , qualityFactor(1.0f)
            , enableErrorRecovery(true)
            , validateInputs(true)
            , enableParallelProcessing(true)
            , numThreads(0)
            , maintainInputType(true)
            , enableDithering(false)
            , performQualityCheck(false)
            , minQualityThreshold(0.8f) {}
    };

    struct CorrectionResult {
        Types::Image correctedImage;
        bool success = false;
        
        // Performance metrics
        float processingTimeMs = 0.0f;
        float memoryUsageMB = 0.0f;
        
        // Quality metrics (if enabled)
        float estimatedAccuracy = 0.0f;
        float dynamicRange = 0.0f;
        
        // Processing details
        Types::ColorSpace inputColorSpace = Types::ColorSpace::SRGB;
        Types::ColorSpace workingColorSpace = Types::ColorSpace::LINEAR_RGB;
        Types::ColorSpace outputColorSpace = Types::ColorSpace::SRGB;
        
        bool wasClampingRequired = false;
        int clampedPixels = 0;
        
        std::string errorMessage;
        
        bool isValid() const;
    };

    LinearCorrector(const CorrectionSettings& settings = CorrectionSettings{});

    CorrectionResult correctImage(const Types::Image& input, 
                                const Domain::CorrectionMatrix& correctionMatrix);

    // Convenience method using calibration data
    CorrectionResult correctImage(const Types::Image& input, 
                                const Domain::CalibrationData& calibrationData);

    // Process single pixel (for testing/validation)
    Types::ColorValue correctPixel(const Types::ColorValue& inputPixel,
                                 const Domain::CorrectionMatrix& correctionMatrix) const;

    // Estimate processing time for given image
    float estimateProcessingTime(const Types::Image& input) const;

    void setSettings(const CorrectionSettings& settings);
    CorrectionSettings getSettings() const;

  private:
    CorrectionSettings settings_;
    Processing::ColorSpaceConverter colorConverter_;

    bool validateInput(const Types::Image& input) const;
    Types::Image correctInLinearSpace(const Types::Image& input, 
                                    const Domain::CorrectionMatrix& correctionMatrix,
                                    CorrectionResult& result);
    Types::Image correctInSRGBSpace(const Types::Image& input, 
                                  const Domain::CorrectionMatrix& correctionMatrix,
                                  CorrectionResult& result);
    Types::Image postProcessImage(const Types::Image& processed, 
                                const Types::Image& original,
                                CorrectionResult& result);
    Types::ColorValue clampColorValues(const Types::ColorValue& color) const;
    int countOutOfRangePixels(const Types::Image& image) const;
    void clampFloatImage(Types::Image& image) const;
    void applyDithering(Types::Image& image) const;
    float assessOutputQuality(const Types::Image& output, const Types::Image& input) const;
    CorrectionResult applyIdentityCorrection(const Types::Image& input);
};

}  // namespace ColorCorrection::Internal::Correction