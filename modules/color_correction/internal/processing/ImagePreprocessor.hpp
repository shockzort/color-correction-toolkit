#pragma once

#include <shared/types/Common.hpp>
#include <shared/utils/Logger.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

namespace ColorCorrection::Internal::Processing {

class ImagePreprocessor {
  public:
    struct PreprocessingSettings {
        // CLAHE (Contrast Limited Adaptive Histogram Equalization)
        bool enableCLAHE;
        double claheClipLimit;
        cv::Size claheTileGridSize;
        
        // Noise reduction
        bool enableDenoising;
        int denoisingStrength;
        int denoisingTemplateWindowSize;
        int denoisingSearchWindowSize;
        
        // Gaussian blur for smoothing
        bool enableGaussianBlur;
        cv::Size gaussianKernelSize;
        double gaussianSigmaX;
        double gaussianSigmaY;
        
        // Sharpening
        bool enableSharpening;
        float sharpeningStrength;
        
        // Color space preprocessing
        bool normalizeChannels;
        bool balanceWhite;
        
        // Geometric preprocessing
        bool enablePerspectiveCorrection;
        bool enableRotationCorrection;
        float maxRotationDegrees;
        
        // Quality enhancement
        bool enhanceContrast;
        float contrastAlpha;
        int contrastBeta;
        
        // Image size constraints
        int maxImageWidth;
        int maxImageHeight;
        bool maintainAspectRatio;
        
        PreprocessingSettings();
    };

    ImagePreprocessor(const PreprocessingSettings& settings = PreprocessingSettings{});

    Types::Image preprocess(const Types::Image& input);

    Types::Image preprocessForDetection(const Types::Image& input, 
                                      Types::DetectionMethod targetMethod);

    Types::Image correctPerspective(const Types::Image& input, 
                                  const std::vector<Types::Point2D>& corners);

    Types::Image correctRotation(const Types::Image& input);

    void setSettings(const PreprocessingSettings& settings);

    PreprocessingSettings getSettings() const;

    // Utility function to assess image quality
    float assessImageQuality(const Types::Image& image);

  private:
    PreprocessingSettings settings_;

    Types::Image resizeIfNeeded(const Types::Image& input);
    Types::Image normalizeChannels(const Types::Image& input);
    Types::Image balanceWhite(const Types::Image& input);
    Types::Image reduceNoise(const Types::Image& input);
    Types::Image enhanceContrast(const Types::Image& input);
    Types::Image applyCLAHE(const Types::Image& input);
    Types::Image applyGaussianBlur(const Types::Image& input);
    Types::Image applySharpen(const Types::Image& input);
};

}  // namespace ColorCorrection::Internal::Processing