#include "ImagePreprocessor.hpp"

namespace ColorCorrection::Internal::Processing {

// PreprocessingSettings constructor
ImagePreprocessor::PreprocessingSettings::PreprocessingSettings()
    : enableCLAHE(true),
      claheClipLimit(2.0),
      claheTileGridSize(8, 8),
      enableDenoising(true),
      denoisingStrength(5),
      denoisingTemplateWindowSize(7),
      denoisingSearchWindowSize(21),
      enableGaussianBlur(false),
      gaussianKernelSize(3, 3),
      gaussianSigmaX(0.0),
      gaussianSigmaY(0.0),
      enableSharpening(false),
      sharpeningStrength(0.5f),
      normalizeChannels(false),
      balanceWhite(false),
      enablePerspectiveCorrection(true),
      enableRotationCorrection(true),
      maxRotationDegrees(10.0f),
      enhanceContrast(true),
      contrastAlpha(1.2f),
      contrastBeta(10),
      maxImageWidth(2048),
      maxImageHeight(1536),
      maintainAspectRatio(true) {}

// ImagePreprocessor constructor
ImagePreprocessor::ImagePreprocessor(const PreprocessingSettings& settings)
    : settings_(settings) {
    LOG_INFO("Image preprocessor initialized");
}

// Main preprocessing method
Types::Image ImagePreprocessor::preprocess(const Types::Image& input) {
    if (input.empty()) {
        LOG_ERROR("Input image is empty");
        return Types::Image();
    }

    LOG_DEBUG("Starting image preprocessing on ", input.cols, "x", input.rows, " image");

    Types::Image processed = input.clone();

    try {
        // Step 1: Resize if needed
        processed = resizeIfNeeded(processed);
        
        // Step 2: Color space normalization
        if (settings_.normalizeChannels) {
            processed = normalizeChannels(processed);
        }
        
        // Step 3: White balance correction
        if (settings_.balanceWhite) {
            processed = balanceWhite(processed);
        }
        
        // Step 4: Noise reduction
        if (settings_.enableDenoising) {
            processed = reduceNoise(processed);
        }
        
        // Step 5: Contrast enhancement
        if (settings_.enhanceContrast) {
            processed = enhanceContrast(processed);
        }
        
        // Step 6: CLAHE for adaptive histogram equalization
        if (settings_.enableCLAHE) {
            processed = applyCLAHE(processed);
        }
        
        // Step 7: Gaussian blur (if enabled)
        if (settings_.enableGaussianBlur) {
            processed = applyGaussianBlur(processed);
        }
        
        // Step 8: Sharpening (if enabled)
        if (settings_.enableSharpening) {
            processed = applySharpen(processed);
        }
        
        LOG_DEBUG("Image preprocessing completed successfully");
        return processed;
        
    } catch (const cv::Exception& e) {
        LOG_ERROR("OpenCV exception during preprocessing: ", e.what());
        return input; // Return original on error
    } catch (const std::exception& e) {
        LOG_ERROR("Exception during preprocessing: ", e.what());
        return input;
    }
}

// Detection-specific preprocessing
Types::Image ImagePreprocessor::preprocessForDetection(const Types::Image& input, 
                                  Types::DetectionMethod targetMethod) {
    // Customize preprocessing based on detection method
    PreprocessingSettings methodSettings = settings_;
    
    switch (targetMethod) {
        case Types::DetectionMethod::MCC_DETECTOR:
            // MCC detector works best with minimal preprocessing
            methodSettings.enableCLAHE = true;
            methodSettings.claheClipLimit = 1.5;
            methodSettings.enableDenoising = true;
            methodSettings.denoisingStrength = 3;
            methodSettings.enableSharpening = false;
            break;
            
        case Types::DetectionMethod::CONTOUR_BASED:
            // Contour detection needs high contrast and edge enhancement
            methodSettings.enableCLAHE = true;
            methodSettings.claheClipLimit = 3.0;
            methodSettings.enhanceContrast = true;
            methodSettings.contrastAlpha = 1.3f;
            methodSettings.enableSharpening = true;
            methodSettings.sharpeningStrength = 0.7f;
            break;
            
        case Types::DetectionMethod::TEMPLATE_MATCHING:
            // Template matching benefits from noise reduction and normalization
            methodSettings.enableDenoising = true;
            methodSettings.denoisingStrength = 7;
            methodSettings.normalizeChannels = true;
            methodSettings.enableCLAHE = true;
            methodSettings.claheClipLimit = 2.0;
            break;
    }

    // Temporarily use method-specific settings
    PreprocessingSettings originalSettings = settings_;
    settings_ = methodSettings;
    
    Types::Image result = preprocess(input);
    
    // Restore original settings
    settings_ = originalSettings;
    
    return result;
}

// Perspective correction
Types::Image ImagePreprocessor::correctPerspective(const Types::Image& input, 
                              const std::vector<Types::Point2D>& corners) {
    if (input.empty() || corners.size() != 4) {
        LOG_ERROR("Invalid input for perspective correction");
        return input;
    }

    try {
        // Define target rectangle (normalized ColorChecker aspect ratio ~1.5:1)
        float targetWidth = 600.0f;
        float targetHeight = 400.0f;
        
        std::vector<cv::Point2f> targetCorners = {
            cv::Point2f(0, 0),
            cv::Point2f(targetWidth, 0),
            cv::Point2f(targetWidth, targetHeight),
            cv::Point2f(0, targetHeight)
        };

        // Calculate perspective transform
        cv::Mat transformMatrix = cv::getPerspectiveTransform(corners, targetCorners);
        
        // Apply transformation
        Types::Image corrected;
        cv::warpPerspective(input, corrected, transformMatrix, 
                          cv::Size(static_cast<int>(targetWidth), static_cast<int>(targetHeight)));
        
        LOG_DEBUG("Perspective correction applied successfully");
        return corrected;
        
    } catch (const cv::Exception& e) {
        LOG_ERROR("Error in perspective correction: ", e.what());
        return input;
    }
}

// Rotation correction
Types::Image ImagePreprocessor::correctRotation(const Types::Image& input) {
    if (input.empty()) {
        return input;
    }

    try {
        // Detect rotation using line detection
        Types::Image gray;
        if (input.channels() > 1) {
            cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = input.clone();
        }

        // Edge detection
        Types::Image edges;
        cv::Canny(gray, edges, 50, 150, 3);
        
        // Hough line detection
        std::vector<cv::Vec2f> lines;
        cv::HoughLines(edges, lines, 1, CV_PI/180, 100);
        
        if (lines.empty()) {
            return input; // No lines detected
        }

        // Calculate dominant angle
        std::vector<float> angles;
        for (const auto& line : lines) {
            float rho = line[0];
            float theta = line[1];
            float angle = (theta - CV_PI/2) * 180.0f / CV_PI;
            
            // Normalize angle to [-45, 45] range
            while (angle > 45) angle -= 90;
            while (angle < -45) angle += 90;
            
            if (std::abs(angle) <= settings_.maxRotationDegrees) {
                angles.push_back(angle);
            }
        }

        if (angles.empty()) {
            return input;
        }

        // Calculate median angle
        std::sort(angles.begin(), angles.end());
        float medianAngle = angles[angles.size() / 2];
        
        if (std::abs(medianAngle) < 1.0f) {
            return input; // Rotation too small to correct
        }

        // Apply rotation correction
        cv::Point2f center(input.cols / 2.0f, input.rows / 2.0f);
        cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, medianAngle, 1.0);
        
        Types::Image rotated;
        cv::warpAffine(input, rotated, rotationMatrix, input.size());
        
        LOG_DEBUG("Rotation correction applied: ", medianAngle, " degrees");
        return rotated;
        
    } catch (const cv::Exception& e) {
        LOG_ERROR("Error in rotation correction: ", e.what());
        return input;
    }
}

// Settings setter
void ImagePreprocessor::setSettings(const PreprocessingSettings& settings) {
    settings_ = settings;
}

// Settings getter
ImagePreprocessor::PreprocessingSettings ImagePreprocessor::getSettings() const {
    return settings_;
}

// Image quality assessment
float ImagePreprocessor::assessImageQuality(const Types::Image& image) {
    if (image.empty()) {
        return 0.0f;
    }

    float quality = 0.0f;
    int factors = 0;

    try {
        // Factor 1: Contrast (standard deviation)
        cv::Scalar mean, stddev;
        cv::meanStdDev(image, mean, stddev);
        
        double avgStddev = stddev[0];
        if (image.channels() >= 3) {
            avgStddev = (stddev[0] + stddev[1] + stddev[2]) / 3.0;
        }
        
        quality += std::min(1.0f, static_cast<float>(avgStddev / 50.0)); // Good contrast > 50
        factors++;

        // Factor 2: Sharpness (Laplacian variance)
        Types::Image gray;
        if (image.channels() > 1) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image.clone();
        }
        
        Types::Image laplacian;
        cv::Laplacian(gray, laplacian, CV_64F);
        
        cv::Scalar laplacianMean, laplacianStddev;
        cv::meanStdDev(laplacian, laplacianMean, laplacianStddev);
        
        double variance = laplacianStddev[0] * laplacianStddev[0];
        quality += std::min(1.0f, static_cast<float>(variance / 1000.0)); // Good sharpness > 1000
        factors++;

        // Factor 3: Brightness distribution (avoid over/under exposure)
        cv::Scalar meanBrightness = cv::mean(gray);
        double brightness = meanBrightness[0];
        
        if (brightness > 50 && brightness < 200) {
            quality += 1.0f; // Good brightness range
        } else if (brightness > 30 && brightness < 220) {
            quality += 0.7f; // Acceptable brightness
        } else {
            quality += 0.3f; // Poor brightness
        }
        factors++;

        // Factor 4: Noise level (estimate)
        Types::Image blurred;
        cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
        
        Types::Image noise;
        cv::absdiff(gray, blurred, noise);
        
        cv::Scalar noiseMean = cv::mean(noise);
        double noiseLevel = noiseMean[0];
        
        quality += std::max(0.0f, 1.0f - static_cast<float>(noiseLevel / 30.0)); // Low noise < 30
        factors++;

    } catch (const cv::Exception& e) {
        LOG_WARN("Error assessing image quality: ", e.what());
        return 0.5f; // Default medium quality
    }

    return factors > 0 ? quality / factors : 0.0f;
}

// Private helper methods

Types::Image ImagePreprocessor::resizeIfNeeded(const Types::Image& input) {
    if (input.cols <= settings_.maxImageWidth && input.rows <= settings_.maxImageHeight) {
        return input;
    }

    double scaleX = static_cast<double>(settings_.maxImageWidth) / input.cols;
    double scaleY = static_cast<double>(settings_.maxImageHeight) / input.rows;
    
    double scale = settings_.maintainAspectRatio ? std::min(scaleX, scaleY) : std::min(scaleX, scaleY);
    
    cv::Size newSize(static_cast<int>(input.cols * scale), 
                    static_cast<int>(input.rows * scale));
    
    Types::Image resized;
    cv::resize(input, resized, newSize, 0, 0, cv::INTER_AREA);
    
    LOG_DEBUG("Image resized from ", input.cols, "x", input.rows, 
             " to ", newSize.width, "x", newSize.height);
    
    return resized;
}

Types::Image ImagePreprocessor::normalizeChannels(const Types::Image& input) {
    if (input.channels() < 3) {
        return input;
    }

    Types::Image normalized;
    input.convertTo(normalized, CV_32F);
    
    std::vector<Types::Image> channels;
    cv::split(normalized, channels);
    
    for (auto& channel : channels) {
        cv::Scalar mean, stddev;
        cv::meanStdDev(channel, mean, stddev);
        
        if (stddev[0] > 1e-6) { // Avoid division by zero
            channel = (channel - mean[0]) / stddev[0];
            channel = channel * 50.0 + 128.0; // Rescale to reasonable range
        }
    }
    
    cv::merge(channels, normalized);
    normalized.convertTo(normalized, input.type());
    
    return normalized;
}

Types::Image ImagePreprocessor::balanceWhite(const Types::Image& input) {
    if (input.channels() < 3) {
        return input;
    }

    Types::Image balanced;
    input.convertTo(balanced, CV_32F);
    
    std::vector<Types::Image> channels;
    cv::split(balanced, channels);
    
    // Calculate channel means
    cv::Scalar meanB = cv::mean(channels[0]);
    cv::Scalar meanG = cv::mean(channels[1]);
    cv::Scalar meanR = cv::mean(channels[2]);
    
    // Calculate correction factors (normalize to green channel)
    double factorB = meanG[0] / meanB[0];
    double factorR = meanG[0] / meanR[0];
    
    // Apply corrections (with limits to avoid overcorrection)
    factorB = std::clamp(factorB, 0.5, 2.0);
    factorR = std::clamp(factorR, 0.5, 2.0);
    
    channels[0] *= factorB;
    channels[2] *= factorR;
    
    cv::merge(channels, balanced);
    balanced.convertTo(balanced, input.type());
    
    return balanced;
}

Types::Image ImagePreprocessor::reduceNoise(const Types::Image& input) {
    Types::Image denoised;
    
    if (input.channels() >= 3) {
        cv::fastNlMeansDenoisingColored(input, denoised, 
            settings_.denoisingStrength, settings_.denoisingStrength,
            settings_.denoisingTemplateWindowSize, settings_.denoisingSearchWindowSize);
    } else {
        cv::fastNlMeansDenoising(input, denoised, 
            settings_.denoisingStrength,
            settings_.denoisingTemplateWindowSize, settings_.denoisingSearchWindowSize);
    }
    
    return denoised;
}

Types::Image ImagePreprocessor::enhanceContrast(const Types::Image& input) {
    Types::Image enhanced;
    input.convertTo(enhanced, -1, settings_.contrastAlpha, settings_.contrastBeta);
    return enhanced;
}

Types::Image ImagePreprocessor::applyCLAHE(const Types::Image& input) {
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(settings_.claheClipLimit, settings_.claheTileGridSize);
    
    if (input.channels() == 1) {
        Types::Image result;
        clahe->apply(input, result);
        return result;
    } else if (input.channels() >= 3) {
        Types::Image lab;
        cv::cvtColor(input, lab, cv::COLOR_BGR2Lab);
        
        std::vector<Types::Image> channels;
        cv::split(lab, channels);
        
        clahe->apply(channels[0], channels[0]); // Apply to L channel only
        
        cv::merge(channels, lab);
        
        Types::Image result;
        cv::cvtColor(lab, result, cv::COLOR_Lab2BGR);
        return result;
    }
    
    return input;
}

Types::Image ImagePreprocessor::applyGaussianBlur(const Types::Image& input) {
    Types::Image blurred;
    cv::GaussianBlur(input, blurred, settings_.gaussianKernelSize,
                    settings_.gaussianSigmaX, settings_.gaussianSigmaY);
    return blurred;
}

Types::Image ImagePreprocessor::applySharpen(const Types::Image& input) {
    // Unsharp mask technique
    Types::Image blurred;
    cv::GaussianBlur(input, blurred, cv::Size(0, 0), 2.0);
    
    Types::Image sharpened;
    cv::addWeighted(input, 1.0 + settings_.sharpeningStrength, blurred, 
                   -settings_.sharpeningStrength, 0, sharpened);
    
    return sharpened;
}

}  // namespace ColorCorrection::Internal::Processing