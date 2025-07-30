#include "ContourDetector.hpp"

namespace ColorCorrection::Internal::Detection {

ContourDetector::ContourSettings::ContourSettings() 
    : minContourArea(1000.0)
    , maxContourArea(50000.0)
    , approxEpsilonFactor(0.02)
    , minAspectRatio(1.2)
    , maxAspectRatio(1.8)
    , gaussianBlurKernel(5)
    , morphologyKernel(3)
    , cannyLowerThreshold(50.0)
    , cannyUpperThreshold(150.0) {}

ContourDetector::ContourDetector(const ContourSettings& settings) 
    : settings_(settings) {
    LOG_INFO("Contour-based detector initialized");
}

Domain::DetectionResult ContourDetector::detect(const Types::Image& image) {
    if (image.empty()) {
        return Domain::DetectionResult::createFailure(
            Types::DetectionMethod::CONTOUR_BASED, 
            "Input image is empty");
    }

    try {
        LOG_DEBUG("Starting contour detection on image ", image.cols, "x", image.rows);

        // Preprocessing
        Types::Image processed = preprocessImage(image);
        
        // Find contours
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(processed, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        LOG_DEBUG("Found ", contours.size(), " contours");

        // Find ColorChecker candidate
        std::vector<cv::Point2f> colorCheckerCorners;
        std::vector<std::vector<cv::Point2f>> patchCenters;
        
        if (!findColorCheckerContour(contours, image, colorCheckerCorners, patchCenters)) {
            return Domain::DetectionResult::createFailure(
                Types::DetectionMethod::CONTOUR_BASED, 
                "No ColorChecker pattern found in contours");
        }

        // Extract color patches
        std::vector<Domain::ColorPatch> patches = extractColorPatches(image, patchCenters[0]);
        
        if (patches.size() < Types::COLORCHECKER_PATCHES * 0.7f) {
            return Domain::DetectionResult::createFailure(
                Types::DetectionMethod::CONTOUR_BASED, 
                "Insufficient patches detected");
        }

        // Calculate confidence
        float confidence = calculateConfidence(colorCheckerCorners, patches, image);
        
        if (confidence < confidenceThreshold_) {
            return Domain::DetectionResult::createFailure(
                Types::DetectionMethod::CONTOUR_BASED, 
                "Detection confidence too low");
        }

        LOG_INFO("Contour detection successful: ", patches.size(), " patches, confidence: ", 
                 confidence);

        return Domain::DetectionResult(
            Types::DetectionMethod::CONTOUR_BASED,
            patches,
            Types::ConfidenceScore::fromValue(confidence),
            colorCheckerCorners
        );

    } catch (const cv::Exception& e) {
        LOG_ERROR("OpenCV exception in contour detection: ", e.what());
        return Domain::DetectionResult::createFailure(
            Types::DetectionMethod::CONTOUR_BASED, 
            "OpenCV exception: " + std::string(e.what()));
    }
}

Types::DetectionMethod ContourDetector::getMethod() const {
    return Types::DetectionMethod::CONTOUR_BASED;
}

Types::ConfidenceScore ContourDetector::getExpectedConfidence(const Types::Image& image) const {
    if (image.empty()) {
        return Types::ConfidenceScore::fromValue(0.0f);
    }

    float score = 0.4f; // Base score for contour detection
    
    // Prefer high contrast images
    cv::Scalar meanVal, stdVal;
    cv::meanStdDev(image, meanVal, stdVal);
    double contrast = stdVal[0];
    if (image.channels() >= 3) {
        contrast = (stdVal[0] + stdVal[1] + stdVal[2]) / 3.0;
    }
    
    if (contrast > 40.0) score += 0.3f;
    else if (contrast > 25.0) score += 0.2f;
    else if (contrast < 15.0) score -= 0.2f;
    
    // Prefer medium-sized images (not too small, not too large)
    int pixels = image.cols * image.rows;
    if (pixels > 640 * 480 && pixels < 1920 * 1080) score += 0.1f;
    
    return Types::ConfidenceScore::fromValue(std::clamp(score, 0.0f, 1.0f));
}

bool ContourDetector::isCapable(const Types::Image& image) const {
    return !image.empty() && image.cols >= 300 && image.rows >= 200;
}

std::string ContourDetector::getName() const {
    return "Contour-based Detector";
}

void ContourDetector::setContourSettings(const ContourSettings& settings) {
    settings_ = settings;
}

ContourDetector::ContourSettings ContourDetector::getContourSettings() const {
    return settings_;
}

void ContourDetector::setConfidenceThreshold(float threshold) {
    confidenceThreshold_ = std::clamp(threshold, 0.0f, 1.0f);
}

float ContourDetector::getConfidenceThreshold() const {
    return confidenceThreshold_;
}

Types::Image ContourDetector::preprocessImage(const Types::Image& image) {
    Types::Image processed;
    
    // Convert to grayscale if needed
    if (image.channels() > 1) {
        cv::cvtColor(image, processed, cv::COLOR_BGR2GRAY);
    } else {
        processed = image.clone();
    }

    // Gaussian blur to reduce noise
    cv::GaussianBlur(processed, processed, cv::Size(settings_.gaussianBlurKernel, 
                    settings_.gaussianBlurKernel), 0);

    // Adaptive histogram equalization for better contrast
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(processed, processed);

    // Edge detection
    cv::Canny(processed, processed, settings_.cannyLowerThreshold, 
             settings_.cannyUpperThreshold);

    // Morphological operations to close gaps
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, 
        cv::Size(settings_.morphologyKernel, settings_.morphologyKernel));
    cv::morphologyEx(processed, processed, cv::MORPH_CLOSE, kernel);

    return processed;
}

bool ContourDetector::findColorCheckerContour(const std::vector<std::vector<cv::Point>>& contours,
                            const Types::Image& originalImage,
                            std::vector<cv::Point2f>& corners,
                            std::vector<std::vector<cv::Point2f>>& patchCenters) {
    
    std::vector<std::pair<double, size_t>> candidates; // score, index
    
    for (size_t i = 0; i < contours.size(); ++i) {
        const auto& contour = contours[i];
        
        // Basic area filtering
        double area = cv::contourArea(contour);
        if (area < settings_.minContourArea || area > settings_.maxContourArea) {
            continue;
        }

        // Approximate to polygon
        std::vector<cv::Point> approx;
        double epsilon = settings_.approxEpsilonFactor * cv::arcLength(contour, true);
        cv::approxPolyDP(contour, approx, epsilon, true);

        // Look for rectangular shapes (4 corners)
        if (approx.size() != 4) {
            continue;
        }

        // Check aspect ratio
        cv::Rect boundingRect = cv::boundingRect(approx);
        double aspectRatio = static_cast<double>(boundingRect.width) / boundingRect.height;
        
        if (aspectRatio < settings_.minAspectRatio || aspectRatio > settings_.maxAspectRatio) {
            continue;
        }

        // Calculate quality score
        double score = calculateContourScore(contour, approx, originalImage);
        candidates.emplace_back(score, i);
    }

    if (candidates.empty()) {
        LOG_DEBUG("No suitable rectangular contours found");
        return false;
    }

    // Sort by score (highest first)
    std::sort(candidates.begin(), candidates.end(), std::greater<>());

    // Try the best candidates
    for (const auto& candidate : candidates) {
        size_t contourIdx = candidate.second;
        const auto& contour = contours[contourIdx];
        
        std::vector<cv::Point> approx;
        double epsilon = settings_.approxEpsilonFactor * cv::arcLength(contour, true);
        cv::approxPolyDP(contour, approx, epsilon, true);

        // Convert to Point2f
        corners.clear();
        for (const auto& point : approx) {
            corners.emplace_back(point.x, point.y);
        }

        // Order corners (top-left, top-right, bottom-right, bottom-left)
        orderCorners(corners);

        // Generate patch centers grid
        std::vector<cv::Point2f> gridCenters;
        if (generatePatchGrid(corners, gridCenters)) {
            patchCenters.clear();
            patchCenters.push_back(gridCenters);
            LOG_DEBUG("Found ColorChecker candidate with score: ", candidate.first);
            return true;
        }
    }

    return false;
}

double ContourDetector::calculateContourScore(const std::vector<cv::Point>& contour,
                           const std::vector<cv::Point>& approx,
                           const Types::Image& image) {
    double score = 0.0;

    // Factor 1: How well the approximation fits the contour
    double approxError = cv::matchShapes(contour, approx, cv::CONTOURS_MATCH_I2, 0.0);
    score += std::max(0.0, 1.0 - approxError);

    // Factor 2: Rectangularity (how close to a rectangle)
    cv::Rect boundingRect = cv::boundingRect(approx);
    double contourArea = cv::contourArea(approx);
    double rectangularityRatio = contourArea / (boundingRect.width * boundingRect.height);
    score += rectangularityRatio;

    // Factor 3: Aspect ratio closeness to ColorChecker (1.5:1)
    double aspectRatio = static_cast<double>(boundingRect.width) / boundingRect.height;
    double aspectScore = 1.0 - std::abs(aspectRatio - 1.5) / 1.5;
    score += aspectScore;

    // Factor 4: Position (prefer center of image)
    cv::Point2f center = (approx[0] + approx[1] + approx[2] + approx[3]) * 0.25f;
    cv::Point2f imageCenter(image.cols * 0.5f, image.rows * 0.5f);
    double distanceFromCenter = cv::norm(center - imageCenter);
    double maxDistance = cv::norm(cv::Point2f(image.cols, image.rows));
    double positionScore = 1.0 - (distanceFromCenter / maxDistance);
    score += positionScore * 0.5; // Lower weight for position

    return score;
}

void ContourDetector::orderCorners(std::vector<cv::Point2f>& corners) {
    if (corners.size() != 4) return;

    // Calculate centroid
    cv::Point2f center(0, 0);
    for (const auto& corner : corners) {
        center += corner;
    }
    center *= 0.25f;

    // Sort by angle from center
    std::sort(corners.begin(), corners.end(), [&center](const cv::Point2f& a, const cv::Point2f& b) {
        return std::atan2(a.y - center.y, a.x - center.x) < std::atan2(b.y - center.y, b.x - center.x);
    });

    // Ensure proper order: top-left, top-right, bottom-right, bottom-left
    std::vector<cv::Point2f> ordered(4);
    
    // Find top-left (minimum sum of coordinates)
    int topLeftIdx = 0;
    float minSum = corners[0].x + corners[0].y;
    for (int i = 1; i < 4; ++i) {
        float sum = corners[i].x + corners[i].y;
        if (sum < minSum) {
            minSum = sum;
            topLeftIdx = i;
        }
    }

    // Assign corners in clockwise order starting from top-left
    for (int i = 0; i < 4; ++i) {
        ordered[i] = corners[(topLeftIdx + i) % 4];
    }

    corners = ordered;
}

bool ContourDetector::generatePatchGrid(const std::vector<cv::Point2f>& corners, 
                      std::vector<cv::Point2f>& gridCenters) {
    if (corners.size() != 4) {
        return false;
    }

    gridCenters.clear();
    gridCenters.reserve(Types::COLORCHECKER_PATCHES);

    // Use perspective transform to create regular grid
    cv::Point2f topLeft = corners[0];
    cv::Point2f topRight = corners[1];
    cv::Point2f bottomRight = corners[2];
    cv::Point2f bottomLeft = corners[3];

    // Generate 6x4 grid (ColorChecker Classic layout)
    for (int row = 0; row < Types::COLORCHECKER_ROWS; ++row) {
        for (int col = 0; col < Types::COLORCHECKER_COLS; ++col) {
            // Normalized coordinates within the ColorChecker (0 to 1)
            float u = (col + 0.5f) / Types::COLORCHECKER_COLS;
            float v = (row + 0.5f) / Types::COLORCHECKER_ROWS;

            // Bilinear interpolation to get actual coordinates
            cv::Point2f top = topLeft + u * (topRight - topLeft);
            cv::Point2f bottom = bottomLeft + u * (bottomRight - bottomLeft);
            cv::Point2f center = top + v * (bottom - top);

            gridCenters.push_back(center);
        }
    }

    return true;
}

std::vector<Domain::ColorPatch> ContourDetector::extractColorPatches(const Types::Image& image,
                                                   const std::vector<cv::Point2f>& centers) {
    std::vector<Domain::ColorPatch> patches;
    auto referencePatches = Domain::ColorPatch::createStandardColorChecker();

    if (centers.size() != Types::COLORCHECKER_PATCHES) {
        LOG_WARN("Expected ", Types::COLORCHECKER_PATCHES, " patch centers, got ", centers.size());
        return patches;
    }

    // Estimate patch size from grid spacing
    float avgSpacing = 0.0f;
    int spacingCount = 0;
    
    for (int i = 0; i < Types::COLORCHECKER_ROWS; ++i) {
        for (int j = 0; j < Types::COLORCHECKER_COLS - 1; ++j) {
            int idx1 = i * Types::COLORCHECKER_COLS + j;
            int idx2 = i * Types::COLORCHECKER_COLS + j + 1;
            avgSpacing += cv::norm(centers[idx1] - centers[idx2]);
            spacingCount++;
        }
    }
    avgSpacing /= spacingCount;
    
    int patchRadius = static_cast<int>(avgSpacing * 0.3f); // 30% of spacing

    for (int i = 0; i < Types::COLORCHECKER_PATCHES; ++i) {
        cv::Point2f center = centers[i];
        
        // Extract color from circular region around center
        cv::Point centerInt(static_cast<int>(center.x), static_cast<int>(center.y));
        
        // Bounds checking
        if (centerInt.x - patchRadius < 0 || centerInt.x + patchRadius >= image.cols ||
            centerInt.y - patchRadius < 0 || centerInt.y + patchRadius >= image.rows) {
            continue;
        }

        // Extract region of interest
        cv::Rect roi(centerInt.x - patchRadius, centerInt.y - patchRadius,
                    2 * patchRadius + 1, 2 * patchRadius + 1);
        
        Types::Image patchRegion = image(roi);
        
        // Calculate mean color (simple average)
        cv::Scalar meanColor = cv::mean(patchRegion);
        
        Types::ColorValue measuredColor;
        if (image.channels() >= 3) {
            measuredColor = Types::ColorValue(
                meanColor[2] / 255.0f, // R
                meanColor[1] / 255.0f, // G
                meanColor[0] / 255.0f  // B
            );
        } else {
            float gray = meanColor[0] / 255.0f;
            measuredColor = Types::ColorValue(gray, gray, gray);
        }

        // Create patch
        Domain::ColorPatch patch(
            i,
            center,
            measuredColor,
            referencePatches[i].getReferenceColor(),
            Types::ConfidenceScore::fromValue(0.7f) // Moderate confidence for contour detection
        );

        patches.push_back(patch);
    }

    return patches;
}

float ContourDetector::calculateConfidence(const std::vector<cv::Point2f>& corners,
                        const std::vector<Domain::ColorPatch>& patches,
                        const Types::Image& image) {
    float confidence = 0.6f; // Base confidence for contour detection

    // Factor 1: Geometry quality
    if (corners.size() == 4) {
        double area = cv::contourArea(corners);
        if (area > 10000.0) confidence += 0.1f;

        // Check if corners form a reasonable quadrilateral
        cv::Rect boundingRect = cv::boundingRect(corners);
        double aspectRatio = static_cast<double>(boundingRect.width) / boundingRect.height;
        if (aspectRatio > 1.3 && aspectRatio < 1.7) confidence += 0.1f;
    }

    // Factor 2: Patch completeness
    float completeness = static_cast<float>(patches.size()) / Types::COLORCHECKER_PATCHES;
    confidence *= completeness;

    // Factor 3: Color consistency
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
            if (avgDeltaE < 15.0f) confidence += 0.1f;
            else if (avgDeltaE > 40.0f) confidence -= 0.15f;
        }
    }

    return std::clamp(confidence, 0.0f, 1.0f);
}

}  // namespace ColorCorrection::Internal::Detection