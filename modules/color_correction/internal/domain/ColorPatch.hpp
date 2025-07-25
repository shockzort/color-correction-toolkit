#pragma once

#include <shared/types/Common.hpp>
#include <shared/utils/Logger.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include <vector>

namespace ColorCorrection::Domain {

class ColorPatch {
  public:
    ColorPatch() = default;
    
    ColorPatch(int patchId, const Types::Point2D& center, const Types::ColorValue& measuredColor,
               const Types::ColorValue& referenceColor, Types::ConfidenceScore confidence)
        : patchId_(patchId), center_(center), measuredColor_(measuredColor),
          referenceColor_(referenceColor), confidence_(confidence) {
        validatePatch();
    }

    int getPatchId() const { return patchId_; }
    Types::Point2D getCenter() const { return center_; }
    Types::ColorValue getMeasuredColor() const { return measuredColor_; }
    Types::ColorValue getReferenceColor() const { return referenceColor_; }
    Types::ConfidenceScore getConfidence() const { return confidence_; }

    void setMeasuredColor(const Types::ColorValue& color) {
        measuredColor_ = color;
        validateColor(color);
    }

    void setReferenceColor(const Types::ColorValue& color) {
        referenceColor_ = color;
        validateColor(color);
    }

    void setCenter(const Types::Point2D& center) { center_ = center; }

    void setConfidence(Types::ConfidenceScore confidence) {
        if (!confidence.isValid()) {
            LOG_WARN("Invalid confidence score for patch ", patchId_, ": ", confidence.value);
        }
        confidence_ = confidence;
    }

    bool isValid() const {
        return patchId_ >= 0 && patchId_ < Types::COLORCHECKER_PATCHES && confidence_.isValid() &&
               isColorValid(measuredColor_) && isColorValid(referenceColor_);
    }

    float calculateDeltaE() const {
        if (!isValid()) return std::numeric_limits<float>::max();
        
        // Simplified Delta E calculation (should be replaced with proper CIE Delta E 2000)
        Types::ColorValue diff = measuredColor_ - referenceColor_;
        return cv::norm(diff, cv::NORM_L2);
    }

    static std::vector<ColorPatch> createStandardColorChecker() {
        std::vector<ColorPatch> patches;
        patches.reserve(Types::COLORCHECKER_PATCHES);

        // Standard ColorChecker Classic reference values (sRGB)
        const std::array<Types::ColorValue, Types::COLORCHECKER_PATCHES> referenceColors = {{
            {115, 82, 68},   // Dark Skin
            {194, 150, 130}, // Light Skin
            {98, 122, 157},  // Blue Sky
            {87, 108, 67},   // Foliage
            {133, 128, 177}, // Blue Flower
            {103, 189, 170}, // Bluish Green
            {214, 126, 44},  // Orange
            {80, 91, 166},   // Purplish Blue
            {193, 90, 99},   // Moderate Red
            {94, 60, 108},   // Purple
            {157, 188, 64},  // Yellow Green
            {224, 163, 46},  // Orange Yellow
            {56, 61, 150},   // Blue
            {70, 148, 73},   // Green
            {175, 54, 60},   // Red
            {231, 199, 31},  // Yellow
            {187, 86, 149},  // Magenta
            {8, 133, 161},   // Cyan
            {243, 243, 242}, // White 9.5
            {200, 200, 200}, // Neutral 8
            {160, 160, 160}, // Neutral 6.5
            {122, 122, 121}, // Neutral 5
            {85, 85, 85},    // Neutral 3.5
            {52, 52, 52}     // Black 2
        }};

        for (int i = 0; i < Types::COLORCHECKER_PATCHES; ++i) {
            Types::Point2D center(0, 0);  // Will be set during detection
            Types::ColorValue measured(0, 0, 0);  // Will be measured from image
            Types::ColorValue reference = referenceColors[i] / 255.0f;  // Normalize to [0,1]
            
            patches.emplace_back(i, center, measured, reference, 
                               Types::ConfidenceScore::fromValue(1.0f));
        }

        return patches;
    }

  private:
    int patchId_ = -1;
    Types::Point2D center_;
    Types::ColorValue measuredColor_;
    Types::ColorValue referenceColor_;
    Types::ConfidenceScore confidence_;

    void validatePatch() {
        if (patchId_ < 0 || patchId_ >= Types::COLORCHECKER_PATCHES) {
            LOG_ERROR("Invalid patch ID: ", patchId_);
        }
        validateColor(measuredColor_);
        validateColor(referenceColor_);
    }

    void validateColor(const Types::ColorValue& color) {
        if (!isColorValid(color)) {
            LOG_WARN("Color values out of expected range: [", color[0], ", ", color[1], ", ", 
                     color[2], "]");
        }
    }

    bool isColorValid(const Types::ColorValue& color) const {
        return color[0] >= 0.0f && color[0] <= 1.0f && 
               color[1] >= 0.0f && color[1] <= 1.0f &&
               color[2] >= 0.0f && color[2] <= 1.0f;
    }
};

}  // namespace ColorCorrection::Domain