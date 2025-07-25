#pragma once

#include "ColorPatch.hpp"
#include <shared/types/Common.hpp>
#include <shared/utils/Logger.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>

namespace ColorCorrection::Domain {

class DetectionResult {
  public:
    DetectionResult() : method_(Types::DetectionMethod::MCC_DETECTOR), 
                       overallConfidence_(Types::ConfidenceScore::fromValue(0.0f)),
                       isSuccess_(false) {}

    DetectionResult(Types::DetectionMethod method, 
                   const std::vector<ColorPatch>& patches,
                   Types::ConfidenceScore confidence,
                   const std::vector<Types::Point2D>& corners = {})
        : method_(method), patches_(patches), overallConfidence_(confidence),
          corners_(corners), isSuccess_(true) {
        
        detectionTime_ = std::chrono::steady_clock::now();
        validateResult();
    }

    Types::DetectionMethod getMethod() const { return method_; }
    const std::vector<ColorPatch>& getPatches() const { return patches_; }
    Types::ConfidenceScore getOverallConfidence() const { return overallConfidence_; }
    const std::vector<Types::Point2D>& getCorners() const { return corners_; }
    bool isSuccess() const { return isSuccess_; }
    
    std::chrono::steady_clock::time_point getDetectionTime() const { return detectionTime_; }

    void setPatches(const std::vector<ColorPatch>& patches) {
        patches_ = patches;
        validateResult();
    }

    void addPatch(const ColorPatch& patch) {
        patches_.push_back(patch);
        recalculateConfidence();
    }

    void setCorners(const std::vector<Types::Point2D>& corners) {
        if (corners.size() != 4) {
            LOG_WARN("ColorChecker should have exactly 4 corners, got ", corners.size());
        }
        corners_ = corners;
    }

    bool hasValidGeometry() const {
        if (corners_.size() != 4) return false;
        
        // Check if corners form a reasonable quadrilateral
        // Calculate area using shoelace formula
        float area = 0.0f;
        for (size_t i = 0; i < corners_.size(); ++i) {
            size_t j = (i + 1) % corners_.size();
            area += corners_[i].x * corners_[j].y - corners_[j].x * corners_[i].y;
        }
        area = std::abs(area) / 2.0f;
        
        return area > 1000.0f;  // Minimum reasonable area
    }

    cv::Rect getBoundingRect() const {
        if (corners_.empty()) {
            if (patches_.empty()) return cv::Rect();
            
            // Calculate from patch centers
            float minX = std::numeric_limits<float>::max();
            float minY = std::numeric_limits<float>::max();
            float maxX = std::numeric_limits<float>::min();
            float maxY = std::numeric_limits<float>::min();
            
            for (const auto& patch : patches_) {
                Types::Point2D center = patch.getCenter();
                minX = std::min(minX, center.x);
                minY = std::min(minY, center.y);
                maxX = std::max(maxX, center.x);
                maxY = std::max(maxY, center.y);
            }
            
            return cv::Rect(static_cast<int>(minX), static_cast<int>(minY),
                          static_cast<int>(maxX - minX), static_cast<int>(maxY - minY));
        }
        
        return cv::boundingRect(corners_);
    }

    std::string getMethodName() const {
        switch (method_) {
            case Types::DetectionMethod::MCC_DETECTOR:
                return "OpenCV MCC Detector";
            case Types::DetectionMethod::CONTOUR_BASED:
                return "Contour-based Detection";
            case Types::DetectionMethod::TEMPLATE_MATCHING:
                return "Template Matching";
            default:
                return "Unknown Method";
        }
    }

    float calculateAverageConfidence() const {
        if (patches_.empty()) return 0.0f;
        
        float sum = 0.0f;
        for (const auto& patch : patches_) {
            sum += patch.getConfidence().value;
        }
        return sum / patches_.size();
    }

    size_t getValidPatchCount() const {
        size_t count = 0;
        for (const auto& patch : patches_) {
            if (patch.isValid()) ++count;
        }
        return count;
    }

    void markAsFailure(const std::string& reason = "") {
        isSuccess_ = false;
        overallConfidence_ = Types::ConfidenceScore::fromValue(0.0f);
        
        if (!reason.empty()) {
            LOG_WARN("Detection marked as failure: ", reason);
        }
    }

    static DetectionResult createFailure(Types::DetectionMethod method, 
                                       const std::string& reason = "") {
        DetectionResult result;
        result.method_ = method;
        result.isSuccess_ = false;
        result.overallConfidence_ = Types::ConfidenceScore::fromValue(0.0f);
        
        if (!reason.empty()) {
            LOG_WARN("Detection failed (", result.getMethodName(), "): ", reason);
        }
        
        return result;
    }

  private:
    Types::DetectionMethod method_;
    std::vector<ColorPatch> patches_;
    Types::ConfidenceScore overallConfidence_;
    std::vector<Types::Point2D> corners_;
    bool isSuccess_;
    std::chrono::steady_clock::time_point detectionTime_;

    void validateResult() {
        if (patches_.size() != Types::COLORCHECKER_PATCHES) {
            LOG_WARN("Expected ", Types::COLORCHECKER_PATCHES, " patches, got ", patches_.size());
            if (patches_.size() < Types::COLORCHECKER_PATCHES / 2) {
                markAsFailure("Too few patches detected");
                return;
            }
        }

        size_t validPatches = getValidPatchCount();
        if (validPatches < Types::COLORCHECKER_PATCHES * 0.8f) { // 80% threshold
            markAsFailure("Too many invalid patches");
            return;
        }

        recalculateConfidence();
    }

    void recalculateConfidence() {
        if (patches_.empty()) {
            overallConfidence_ = Types::ConfidenceScore::fromValue(0.0f);
            return;
        }

        float avgConfidence = calculateAverageConfidence();
        float geometryBonus = hasValidGeometry() ? 0.1f : 0.0f;
        float completenessBonus = (patches_.size() == Types::COLORCHECKER_PATCHES) ? 0.1f : 0.0f;
        
        float finalConfidence = std::min(1.0f, avgConfidence + geometryBonus + completenessBonus);
        overallConfidence_ = Types::ConfidenceScore::fromValue(finalConfidence);
    }
};

}  // namespace ColorCorrection::Domain