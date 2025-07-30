#include "MCCDetector.hpp"
#include <shared/utils/Logger.hpp>

namespace ColorCorrection::Internal::Detection {

MCCDetector::MCCDetector() {
    try {
        detector_ = cv::mcc::CCheckerDetector::create();
        if (!detector_) {
            LOG_ERROR("Failed to create OpenCV MCC detector - feature not available");
            isAvailable_ = false;
        } else {
            isAvailable_ = true;
            LOG_INFO("OpenCV MCC detector initialized successfully");
        }
    } catch (const cv::Exception& e) {
        LOG_ERROR("OpenCV MCC detector initialization failed: ", e.what());
        isAvailable_ = false;
    }
}

Domain::DetectionResult MCCDetector::detect(const Types::Image& image) {
    if (!isAvailable_ || !detector_) {
        return Domain::DetectionResult::createFailure(
            Types::DetectionMethod::MCC_DETECTOR, 
            "MCC detector not available");
    }

    if (image.empty()) {
        return Domain::DetectionResult::createFailure(
            Types::DetectionMethod::MCC_DETECTOR, 
            "Input image is empty");
    }

    try {
        LOG_DEBUG("Starting MCC detection on image ", image.cols, "x", image.rows);

        // Convert to appropriate format if needed
        Types::Image processedImage;
        if (image.channels() == 4) {
            cv::cvtColor(image, processedImage, cv::COLOR_BGRA2BGR);
        } else if (image.channels() == 1) {
            cv::cvtColor(image, processedImage, cv::COLOR_GRAY2BGR);
        } else {
            processedImage = image.clone();
        }

        // Detect ColorChecker
        bool detected = detector_->process(processedImage, cv::mcc::MCC24);
        
        if (!detected) {
            LOG_DEBUG("MCC detector found no ColorChecker patterns");
            return Domain::DetectionResult::createFailure(
                Types::DetectionMethod::MCC_DETECTOR, 
                "No ColorChecker pattern detected");
        }

        // Get detection results
        std::vector<cv::Ptr<cv::mcc::CChecker>> checkers = detector_->getListColorChecker();
        
        if (checkers.empty()) {
            return Domain::DetectionResult::createFailure(
                Types::DetectionMethod::MCC_DETECTOR, 
                "No valid ColorChecker found");
        }

        // Use the first (best) detection
        cv::Ptr<cv::mcc::CChecker> checker = checkers[0];
        
        // Extract patches
        std::vector<Domain::ColorPatch> patches = extractPatches(processedImage, checker);
        
        if (patches.size() < Types::COLORCHECKER_PATCHES * 0.8f) {
            return Domain::DetectionResult::createFailure(
                Types::DetectionMethod::MCC_DETECTOR, 
                "Insufficient patches detected");
        }

        // Extract corners
        std::vector<Types::Point2D> corners = extractCorners(checker);
        
        // Calculate confidence based on detection quality
        float confidence = calculateConfidence(checker, patches);
        
        if (confidence < confidenceThreshold_) {
            return Domain::DetectionResult::createFailure(
                Types::DetectionMethod::MCC_DETECTOR, 
                "Detection confidence too low");
        }

        LOG_INFO("MCC detection successful: ", patches.size(), " patches, confidence: ", 
                 confidence);
        
        return Domain::DetectionResult(
            Types::DetectionMethod::MCC_DETECTOR,
            patches,
            Types::ConfidenceScore::fromValue(confidence),
            corners
        );

    } catch (const cv::Exception& e) {
        LOG_ERROR("OpenCV exception in MCC detection: ", e.what());
        return Domain::DetectionResult::createFailure(
            Types::DetectionMethod::MCC_DETECTOR, 
            "OpenCV exception: " + std::string(e.what()));
    }
}

Types::DetectionMethod MCCDetector::getMethod() const {
    return Types::DetectionMethod::MCC_DETECTOR;
}

Types::ConfidenceScore MCCDetector::getExpectedConfidence(const Types::Image& image) const {
    if (!isAvailable_ || image.empty()) {
        return Types::ConfidenceScore::fromValue(0.0f);
    }

    // Heuristic based on image quality
    float score = 0.5f; // base score for MCC detector
    
    // Prefer larger images
    int pixels = image.cols * image.rows;
    if (pixels > 1920 * 1080) score += 0.2f;
    else if (pixels > 1280 * 720) score += 0.1f;
    
    // Prefer color images
    if (image.channels() >= 3) score += 0.1f;
    
    // Check basic image quality (contrast)
    cv::Scalar meanVal, stdVal;
    cv::meanStdDev(image, meanVal, stdVal);
    double contrast = stdVal[0];
    if (image.channels() >= 3) {
        contrast = (stdVal[0] + stdVal[1] + stdVal[2]) / 3.0;
    }
    
    if (contrast > 30.0) score += 0.2f;
    else if (contrast < 10.0) score -= 0.2f;
    
    return Types::ConfidenceScore::fromValue(std::clamp(score, 0.0f, 1.0f));
}

bool MCCDetector::isCapable(const Types::Image& image) const {
    return isAvailable_ && !image.empty() && 
           image.cols >= 200 && image.rows >= 150; // Minimum size requirements
}

std::string MCCDetector::getName() const {
    return "OpenCV MCC Detector";
}

void MCCDetector::setConfidenceThreshold(float threshold) {
    confidenceThreshold_ = std::clamp(threshold, 0.0f, 1.0f);
}

float MCCDetector::getConfidenceThreshold() const {
    return confidenceThreshold_;
}

bool MCCDetector::isAvailable() const {
    return isAvailable_;
}

std::vector<Domain::ColorPatch> MCCDetector::extractPatches(const Types::Image& image, 
                                              cv::Ptr<cv::mcc::CChecker> checker) {
    std::vector<Domain::ColorPatch> patches;
    
    // Get reference colors for ColorChecker Classic
    auto referencePatches = Domain::ColorPatch::createStandardColorChecker();
    
    try {
        // Get the ColorChecker chart colors
        cv::Mat chartsRGB = checker->getChartsRGB();
        
        if (chartsRGB.rows != Types::COLORCHECKER_PATCHES) {
            LOG_WARN("Unexpected patch count from MCC detector: ", chartsRGB.rows);
            return patches;
        }

        // Get the chart corners for calculating patch positions
        std::vector<cv::Point2f> chartCorners = checker->getBox();
        cv::Point2f center = checker->getCenter();

        for (int i = 0; i < Types::COLORCHECKER_PATCHES; ++i) {
            // Extract measured color (convert from 0-255 to 0-1 range)
            cv::Vec3f measuredColor;
            if (chartsRGB.type() == CV_64F) {
                cv::Vec3d colorDouble = chartsRGB.at<cv::Vec3d>(i, 0);
                measuredColor = cv::Vec3f(colorDouble[0] / 255.0f, 
                                        colorDouble[1] / 255.0f, 
                                        colorDouble[2] / 255.0f);
            } else if (chartsRGB.type() == CV_32F) {
                measuredColor = chartsRGB.at<cv::Vec3f>(i, 0) / 255.0f;
            } else {
                cv::Vec3b colorByte = chartsRGB.at<cv::Vec3b>(i, 0);
                measuredColor = cv::Vec3f(colorByte[0] / 255.0f, 
                                        colorByte[1] / 255.0f, 
                                        colorByte[2] / 255.0f);
            }

            // Use chart center as patch position approximation
            // (More sophisticated position calculation could be added here)
            cv::Point2f patchCenter = center;
            
            // Create patch with measured and reference colors
            Domain::ColorPatch patch(
                i,
                patchCenter,
                measuredColor,
                referencePatches[i].getReferenceColor(),
                Types::ConfidenceScore::fromValue(0.9f) // High confidence for MCC detector
            );

            patches.push_back(patch);
        }

        LOG_DEBUG("Extracted ", patches.size(), " patches from MCC detection");
        
    } catch (const cv::Exception& e) {
        LOG_ERROR("Error extracting patches from MCC detection: ", e.what());
    }

    return patches;
}

std::vector<Types::Point2D> MCCDetector::extractCorners(cv::Ptr<cv::mcc::CChecker> checker) {
    std::vector<Types::Point2D> corners;
    
    try {
        // Get the box (bounding quadrilateral) of the ColorChecker
        std::vector<cv::Point2f> box = checker->getBox();
        
        if (box.size() == 4) {
            corners.assign(box.begin(), box.end());
        } else {
            LOG_WARN("MCC detector returned ", box.size(), " corners instead of 4");
        }
        
    } catch (const cv::Exception& e) {
        LOG_ERROR("Error extracting corners from MCC detection: ", e.what());
    }
    
    return corners;
}

float MCCDetector::calculateConfidence(cv::Ptr<cv::mcc::CChecker> checker, 
                        const std::vector<Domain::ColorPatch>& patches) {
    float confidence = 0.8f; // Base confidence for successful MCC detection
    
    try {
        // Factor 1: Number of patches detected
        float patchCompleteness = static_cast<float>(patches.size()) / Types::COLORCHECKER_PATCHES;
        confidence *= patchCompleteness;
        
        // Factor 2: Geometry quality (if corners are available)
        std::vector<cv::Point2f> box = checker->getBox();
        if (box.size() == 4) {
            // Check if the box forms a reasonable quadrilateral
            float area = cv::contourArea(box);
            if (area > 10000.0f) { // Reasonable minimum area
                confidence += 0.1f;
            }
            
            // Check aspect ratio (ColorChecker is roughly 1.5:1)
            cv::Rect boundingRect = cv::boundingRect(box);
            float aspectRatio = static_cast<float>(boundingRect.width) / boundingRect.height;
            if (aspectRatio > 1.3f && aspectRatio < 1.7f) {
                confidence += 0.05f;
            }
        }
        
        // Factor 3: Color consistency (basic check)
        if (!patches.empty()) {
            float avgDeltaE = 0.0f;
            int validPatches = 0;
            
            for (const auto& patch : patches) {
                if (patch.isValid()) {
                    avgDeltaE += patch.calculateDeltaE();
                    validPatches++;
                }
            }
            
            if (validPatches > 0) {
                avgDeltaE /= validPatches;
                // Lower Delta E = higher confidence
                if (avgDeltaE < 10.0f) confidence += 0.05f;
                else if (avgDeltaE > 30.0f) confidence -= 0.1f;
            }
        }
        
    } catch (const cv::Exception& e) {
        LOG_WARN("Error calculating MCC confidence: ", e.what());
        confidence *= 0.8f; // Reduce confidence due to calculation error
    }
    
    return std::clamp(confidence, 0.0f, 1.0f);
}

}  // namespace ColorCorrection::Internal::Detection