#include "LinearCorrector.hpp"
#include <algorithm>

namespace ColorCorrection::Internal::Correction {

bool LinearCorrector::CorrectionResult::isValid() const {
    return success && !correctedImage.empty();
}

LinearCorrector::LinearCorrector(const CorrectionSettings& settings)
    : settings_(settings), colorConverter_() {
    LOG_INFO("Linear corrector initialized");
}

LinearCorrector::CorrectionResult LinearCorrector::correctImage(const Types::Image& input, 
                            const Domain::CorrectionMatrix& correctionMatrix) {
    CorrectionResult result;
    
    if (input.empty()) {
        result.errorMessage = "Input image is empty";
        LOG_ERROR(result.errorMessage);
        return result;
    }

    if (!correctionMatrix.isValid()) {
        result.errorMessage = "Correction matrix is invalid";
        LOG_ERROR(result.errorMessage);
        return settings_.enableErrorRecovery ? applyIdentityCorrection(input) : result;
    }

    auto startTime = std::chrono::high_resolution_clock::now();

    try {
        LOG_DEBUG("Correcting image ", input.cols, "x", input.rows, " with ", 
                 input.channels(), " channels");

        // Validate inputs if enabled
        if (settings_.validateInputs && !validateInput(input)) {
            result.errorMessage = "Input validation failed";
            return result;
        }

        // Store processing details
        result.inputColorSpace = Types::ColorSpace::SRGB;
        result.outputColorSpace = Types::ColorSpace::SRGB;
        
        if (settings_.useLinearRGB) {
            result.workingColorSpace = Types::ColorSpace::LINEAR_RGB;
        } else {
            result.workingColorSpace = Types::ColorSpace::SRGB;
        }

        // Apply correction
        Types::Image processedImage;
        if (settings_.useLinearRGB) {
            processedImage = correctInLinearSpace(input, correctionMatrix, result);
        } else {
            processedImage = correctInSRGBSpace(input, correctionMatrix, result);
        }

        if (processedImage.empty()) {
            result.errorMessage = "Correction processing failed";
            return result;
        }

        // Post-processing
        processedImage = postProcessImage(processedImage, input, result);
        
        // Quality check if enabled
        if (settings_.performQualityCheck) {
            float quality = assessOutputQuality(processedImage, input);
            result.estimatedAccuracy = quality;
            
            if (quality < settings_.minQualityThreshold) {
                LOG_WARN("Output quality below threshold: ", quality, " < ", 
                        settings_.minQualityThreshold);
                if (!settings_.enableErrorRecovery) {
                    result.errorMessage = "Output quality below acceptable threshold";
                    return result;
                }
            }
        }

        result.correctedImage = processedImage;
        result.success = true;

        // Calculate performance metrics
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        result.processingTimeMs = duration.count() / 1000.0f;
        
        // Estimate memory usage (rough approximation)
        result.memoryUsageMB = (input.total() * input.elemSize() * 2.0f) / (1024.0f * 1024.0f);
        
        LOG_INFO("Image correction completed successfully in ", result.processingTimeMs, "ms");

    } catch (const cv::Exception& e) {
        result.errorMessage = "OpenCV exception: " + std::string(e.what());
        LOG_ERROR(result.errorMessage);
        
        if (settings_.enableErrorRecovery) {
            LOG_WARN("Applying identity correction as fallback");
            return applyIdentityCorrection(input);
        }
    } catch (const std::exception& e) {
        result.errorMessage = "Exception: " + std::string(e.what());
        LOG_ERROR(result.errorMessage);
        
        if (settings_.enableErrorRecovery) {
            return applyIdentityCorrection(input);
        }
    }

    return result;
}

LinearCorrector::CorrectionResult LinearCorrector::correctImage(const Types::Image& input, 
                            const Domain::CalibrationData& calibrationData) {
    return correctImage(input, calibrationData.getCorrectionMatrix());
}

Types::ColorValue LinearCorrector::correctPixel(const Types::ColorValue& inputPixel,
                             const Domain::CorrectionMatrix& correctionMatrix) const {
    if (!correctionMatrix.isValid()) {
        return inputPixel;
    }

    Types::ColorValue processedPixel = inputPixel;
    
    // Convert to linear RGB if needed
    if (settings_.useLinearRGB) {
        processedPixel = colorConverter_.sRGBToLinear(processedPixel);
    }
    
    // Apply correction matrix
    processedPixel = correctionMatrix.applyCorrection(processedPixel);
    
    // Convert back to sRGB if needed
    if (settings_.useLinearRGB) {
        processedPixel = colorConverter_.linearToSRGB(processedPixel);
    }
    
    // Clamp values if enabled
    if (settings_.clampValues) {
        processedPixel = clampColorValues(processedPixel);
    }
    
    return processedPixel;
}

float LinearCorrector::estimateProcessingTime(const Types::Image& input) const {
    if (input.empty()) return 0.0f;
    
    // Base time per megapixel (empirically determined)
    float baseMsPerMegapixel = settings_.useLinearRGB ? 15.0f : 8.0f;
    
    float megapixels = (input.cols * input.rows) / 1000000.0f;
    float estimatedTime = baseMsPerMegapixel * megapixels;
    
    // Adjust for quality factor
    estimatedTime *= (2.0f - settings_.qualityFactor); // Higher quality = slower
    
    // Adjust for parallel processing
    if (settings_.enableParallelProcessing) {
        int numThreads = settings_.numThreads > 0 ? settings_.numThreads : 
                       std::thread::hardware_concurrency();
        estimatedTime /= std::max(1, numThreads / 2); // Conservative estimate
    }
    
    return estimatedTime;
}

void LinearCorrector::setSettings(const CorrectionSettings& settings) {
    settings_ = settings;
}

LinearCorrector::CorrectionSettings LinearCorrector::getSettings() const {
    return settings_;
}

bool LinearCorrector::validateInput(const Types::Image& input) const {
    // Check basic properties
    if (input.empty()) {
        LOG_ERROR("Input image is empty");
        return false;
    }

    if (input.channels() < 3) {
        LOG_WARN("Input image has fewer than 3 channels");
    }

    // Check for reasonable size
    if (input.cols < 1 || input.rows < 1 || input.cols > 16384 || input.rows > 16384) {
        LOG_ERROR("Input image has unreasonable dimensions: ", input.cols, "x", input.rows);
        return false;
    }

    // Check data type
    int depth = input.depth();
    if (depth != CV_8U && depth != CV_16U && depth != CV_32F) {
        LOG_WARN("Unusual input image depth: ", depth);
    }

    return true;
}

Types::Image LinearCorrector::correctInLinearSpace(const Types::Image& input, 
                                const Domain::CorrectionMatrix& correctionMatrix,
                                CorrectionResult& result) {
    // Convert to linear RGB
    Types::Image linearImage = colorConverter_.convertImageSRGBToLinear(input);
    
    if (linearImage.empty()) {
        LOG_ERROR("Failed to convert input to linear RGB");
        return Types::Image();
    }

    // Apply correction matrix
    Types::Image correctedLinear;
    correctionMatrix.applyToImage(linearImage, correctedLinear);
    
    if (correctedLinear.empty()) {
        LOG_ERROR("Matrix application failed");
        return Types::Image();
    }

    // Convert back to sRGB
    Types::Image srgbImage = colorConverter_.convertImageLinearToSRGB(correctedLinear);
    
    return srgbImage;
}

Types::Image LinearCorrector::correctInSRGBSpace(const Types::Image& input, 
                              const Domain::CorrectionMatrix& correctionMatrix,
                              CorrectionResult& result) {
    // Apply correction directly in sRGB space
    Types::Image correctedImage;
    correctionMatrix.applyToImage(input, correctedImage);
    
    return correctedImage;
}

Types::Image LinearCorrector::postProcessImage(const Types::Image& processed, 
                            const Types::Image& original,
                            CorrectionResult& result) {
    Types::Image output = processed.clone();
    
    // Clamping if enabled
    if (settings_.clampValues) {
        int clampedCount = 0;
        
        if (output.type() == CV_8UC3 || output.type() == CV_8UC4) {
            // 8-bit images - check for overflow
            cv::Mat overflowMask;
            cv::threshold(output, overflowMask, 255, 255, cv::THRESH_BINARY);
            
            // Convert to single channel for countNonZero
            if (overflowMask.channels() > 1) {
                cv::Mat grayMask;
                cv::cvtColor(overflowMask, grayMask, cv::COLOR_BGR2GRAY);
                clampedCount = cv::countNonZero(grayMask);
            } else {
                clampedCount = cv::countNonZero(overflowMask);
            }
            
            // Clamp values
            cv::threshold(output, output, 255, 255, cv::THRESH_TRUNC);
            cv::threshold(output, output, 0, 0, cv::THRESH_TOZERO);
            
        } else if (output.type() == CV_32FC3 || output.type() == CV_32FC4) {
            // Float images - clamp to [0,1]
            clampedCount = countOutOfRangePixels(output);
            clampFloatImage(output);
        }
        
        result.clampedPixels = clampedCount;
        result.wasClampingRequired = (clampedCount > 0);
        
        if (clampedCount > 0) {
            LOG_DEBUG("Clamped ", clampedCount, " out-of-range pixels");
        }
    }

    // Preserve alpha channel if present and requested
    if (settings_.preserveAlpha && original.channels() == 4 && output.channels() >= 3) {
        if (output.channels() == 3) {
            // Add alpha channel from original
            std::vector<Types::Image> channels;
            cv::split(output, channels);
            
            std::vector<Types::Image> originalChannels;
            cv::split(original, originalChannels);
            
            if (originalChannels.size() >= 4) {
                channels.push_back(originalChannels[3]); // Copy alpha
                cv::merge(channels, output);
            }
        } else if (output.channels() == 4) {
            // Replace alpha channel
            std::vector<Types::Image> outputChannels, originalChannels;
            cv::split(output, outputChannels);
            cv::split(original, originalChannels);
            
            if (originalChannels.size() >= 4 && outputChannels.size() >= 4) {
                outputChannels[3] = originalChannels[3].clone();
                cv::merge(outputChannels, output);
            }
        }
    }

    // Maintain input type if requested
    if (settings_.maintainInputType && output.type() != original.type()) {
        Types::Image convertedOutput;
        
        if (original.type() == CV_8UC3 || original.type() == CV_8UC4) {
            // Convert to 8-bit
            if (output.type() == CV_32FC3 || output.type() == CV_32FC4) {
                output.convertTo(convertedOutput, original.type(), 255.0);
            } else {
                output.convertTo(convertedOutput, original.type());
            }
            
            // Apply dithering if enabled
            if (settings_.enableDithering) {
                applyDithering(convertedOutput);
            }
            
        } else {
            output.convertTo(convertedOutput, original.type());
        }
        
        output = convertedOutput;
    }

    return output;
}

Types::ColorValue LinearCorrector::clampColorValues(const Types::ColorValue& color) const {
    return Types::ColorValue(
        std::clamp(color[0], 0.0f, 1.0f),
        std::clamp(color[1], 0.0f, 1.0f),
        std::clamp(color[2], 0.0f, 1.0f)
    );
}

int LinearCorrector::countOutOfRangePixels(const Types::Image& image) const {
    int count = 0;
    
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            if (image.channels() >= 3) {
                cv::Vec3f pixel = image.at<cv::Vec3f>(y, x);
                for (int c = 0; c < 3; ++c) {
                    if (pixel[c] < 0.0f || pixel[c] > 1.0f) {
                        count++;
                        break;
                    }
                }
            }
        }
    }
    
    return count;
}

void LinearCorrector::clampFloatImage(Types::Image& image) const {
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            if (image.channels() >= 3) {
                cv::Vec3f& pixel = image.at<cv::Vec3f>(y, x);
                for (int c = 0; c < 3; ++c) {
                    pixel[c] = std::clamp(pixel[c], 0.0f, 1.0f);
                }
            }
        }
    }
}

void LinearCorrector::applyDithering(Types::Image& image) const {
    // Simple Floyd-Steinberg dithering
    if (image.type() != CV_8UC3 && image.type() != CV_8UC4) {
        return;
    }

    Types::Image floatImage;
    image.convertTo(floatImage, CV_32F, 1.0/255.0);

    for (int y = 0; y < floatImage.rows - 1; ++y) {
        for (int x = 1; x < floatImage.cols - 1; ++x) {
            for (int c = 0; c < std::min(3, floatImage.channels()); ++c) {
                float oldPixel = floatImage.at<cv::Vec3f>(y, x)[c];
                float newPixel = std::round(oldPixel * 255.0f) / 255.0f;
                float error = oldPixel - newPixel;
                
                floatImage.at<cv::Vec3f>(y, x)[c] = newPixel;
                
                // Distribute error
                floatImage.at<cv::Vec3f>(y, x + 1)[c] += error * 7.0f / 16.0f;
                floatImage.at<cv::Vec3f>(y + 1, x - 1)[c] += error * 3.0f / 16.0f;
                floatImage.at<cv::Vec3f>(y + 1, x)[c] += error * 5.0f / 16.0f;
                floatImage.at<cv::Vec3f>(y + 1, x + 1)[c] += error * 1.0f / 16.0f;
            }
        }
    }

    floatImage.convertTo(image, image.type(), 255.0);
}

float LinearCorrector::assessOutputQuality(const Types::Image& output, const Types::Image& input) const {
    try {
        // Basic quality metrics
        cv::Scalar meanOutput, stddevOutput;
        cv::meanStdDev(output, meanOutput, stddevOutput);
        
        // Calculate contrast
        double contrast = (stddevOutput[0] + stddevOutput[1] + stddevOutput[2]) / 3.0;
        float contrastScore = std::min(1.0f, static_cast<float>(contrast / 50.0)); // Good contrast > 50
        
        // Calculate dynamic range
        double minVal, maxVal;
        cv::minMaxLoc(output, &minVal, &maxVal);
        float dynamicRange = static_cast<float>(maxVal - minVal);
        float rangeScore = std::min(1.0f, dynamicRange / 255.0f);
        
        // Check for artifacts (simplified)
        Types::Image gray;
        if (output.channels() >= 3) {
            cv::cvtColor(output, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = output.clone();
        }
        
        Types::Image laplacian;
        cv::Laplacian(gray, laplacian, CV_64F);
        cv::Scalar meanLap, stddevLap;
        cv::meanStdDev(laplacian, meanLap, stddevLap);
        
        float sharpness = static_cast<float>(stddevLap[0]);
        float sharpnessScore = std::min(1.0f, sharpness / 1000.0f); // Good sharpness > 1000
        
        // Combine scores
        float overallQuality = (contrastScore + rangeScore + sharpnessScore) / 3.0f;
        
        return std::clamp(overallQuality, 0.0f, 1.0f);
        
    } catch (const cv::Exception& e) {
        LOG_WARN("Error assessing output quality: ", e.what());
        return 0.5f; // Default moderate quality
    }
}

LinearCorrector::CorrectionResult LinearCorrector::applyIdentityCorrection(const Types::Image& input) {
    CorrectionResult result;
    result.correctedImage = input.clone();
    result.success = true;
    result.processingTimeMs = 0.1f; // Minimal processing time
    result.estimatedAccuracy = 0.0f; // No correction applied
    result.errorMessage = "Applied identity correction (no change)";
    
    LOG_INFO("Applied identity correction as fallback");
    return result;
}

}  // namespace ColorCorrection::Internal::Correction