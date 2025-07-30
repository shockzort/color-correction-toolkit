#pragma once

#include <opencv2/opencv.hpp>
#include <array>
#include <string>
#include <vector>
#include <algorithm>

namespace ColorCorrection::Types {

using Matrix3x3 = cv::Matx33f;
using Vector3 = cv::Vec3f;
using Point2D = cv::Point2f;
using Point3D = cv::Point3f;
using Image = cv::Mat;
using ColorValue = cv::Vec3f;  // RGB or Lab values

constexpr int COLORCHECKER_PATCHES = 24;
constexpr int COLORCHECKER_ROWS = 4;
constexpr int COLORCHECKER_COLS = 6;

enum class ColorSpace { SRGB, LINEAR_RGB, LAB, XYZ };

enum class DetectionMethod { MCC_DETECTOR, CONTOUR_BASED, TEMPLATE_MATCHING };

enum class ProcessingQuality { FAST, BALANCED, HIGH_QUALITY };

struct ConfidenceScore {
    float value = 0.0f;
    bool isValid() const { return value >= 0.0f && value <= 1.0f; }
    
    static ConfidenceScore fromValue(float v) {
        ConfidenceScore score;
        score.value = std::clamp(v, 0.0f, 1.0f);
        return score;
    }
};

}  // namespace ColorCorrection::Types