#pragma once

#include "../domain/DetectionResult.hpp"
#include "../domain/ColorPatch.hpp"
#include <shared/types/Common.hpp>
#include <shared/utils/Logger.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

namespace ColorCorrection::Internal::Detection {

class DetectionValidator {
  public:
    struct ValidationSettings {
        // Geometric validation
        float minColorCheckerArea = 10000.0f;
        float maxColorCheckerArea = 500000.0f;
        float minAspectRatio = 1.2f;
        float maxAspectRatio = 1.8f;
        float maxSkewAngle = 15.0f; // degrees
        
        // Patch validation
        int minRequiredPatches = 20; // 80% of 24 patches
        float maxPatchDistanceVariation = 0.3f; // 30% variation in spacing
        float minPatchConfidence = 0.3f;
        
        // Color validation
        float maxDeltaE = 50.0f; // Very lenient for initial detection
        float maxBrightnessVariation = 0.8f; // Max variation in brightness
        bool validateColorConsistency = true;
        
        // Statistical validation
        float confidenceThreshold = 0.6f;
        int minInlierPatches = 18; // For RANSAC-style validation
        float inlierThreshold = 10.0f; // Delta E threshold for inliers
        
        // Temporal validation (for video sequences)
        bool enableTemporalValidation = false;
        float maxFrameToFrameMovement = 50.0f; // pixels
        int temporalConsistencyFrames = 3;
    };

    DetectionValidator(const ValidationSettings& settings = ValidationSettings{})
        : settings_(settings) {
        LOG_INFO("Detection validator initialized");
    }

    struct ValidationResult {
        bool isValid = false;
        float overallScore = 0.0f;
        
        // Detailed scores
        float geometricScore = 0.0f;
        float patchQualityScore = 0.0f;
        float colorConsistencyScore = 0.0f;
        float spatialConsistencyScore = 0.0f;
        
        // Specific issues found
        std::vector<std::string> warnings;
        std::vector<std::string> errors;
        
        // Statistics
        int validPatchCount = 0;
        int totalPatchCount = 0;
        float averagePatchConfidence = 0.0f;
        float averageDeltaE = 0.0f;
        float colorCheckerArea = 0.0f;
        float aspectRatio = 0.0f;
    };

    ValidationResult validate(const Domain::DetectionResult& detection) {
        ValidationResult result;
        
        if (!detection.isSuccess()) {
            result.errors.push_back("Detection was not successful");
            return result;
        }

        LOG_DEBUG("Validating detection with ", detection.getPatches().size(), " patches");

        try {
            // Geometric validation
            result.geometricScore = validateGeometry(detection, result);
            
            // Patch quality validation
            result.patchQualityScore = validatePatchQuality(detection, result);
            
            // Color consistency validation
            if (settings_.validateColorConsistency) {
                result.colorConsistencyScore = validateColorConsistency(detection, result);
            } else {
                result.colorConsistencyScore = 1.0f;
            }
            
            // Spatial consistency validation
            result.spatialConsistencyScore = validateSpatialConsistency(detection, result);
            
            // Calculate overall score
            result.overallScore = calculateOverallScore(result);
            
            // Determine if valid
            result.isValid = (result.overallScore >= settings_.confidenceThreshold) &&
                           (result.validPatchCount >= settings_.minRequiredPatches) &&
                           result.errors.empty();
            
            // Populate statistics
            populateStatistics(detection, result);
            
            LOG_DEBUG("Validation completed. Score: ", result.overallScore, 
                     ", Valid: ", result.isValid ? "Yes" : "No");
            
        } catch (const std::exception& e) {
            result.errors.push_back("Validation exception: " + std::string(e.what()));
            LOG_ERROR("Exception during validation: ", e.what());
        }

        return result;
    }

    Types::ConfidenceScore calculateConfidenceScore(const Domain::DetectionResult& detection) {
        ValidationResult validation = validate(detection);
        return Types::ConfidenceScore::fromValue(validation.overallScore);
    }

    bool isDetectionAcceptable(const Domain::DetectionResult& detection) {
        ValidationResult validation = validate(detection);
        return validation.isValid;
    }

    ValidationSettings getSettings() const {
        return settings_;
    }

    void setSettings(const ValidationSettings& settings) {
        settings_ = settings;
    }

    // Enhanced validation for specific use cases
    ValidationResult validateForCalibration(const Domain::DetectionResult& detection) {
        // Temporarily use stricter settings for calibration
        ValidationSettings calibrationSettings = settings_;
        calibrationSettings.minRequiredPatches = 22; // 90% of patches required
        calibrationSettings.maxDeltaE = 30.0f; // Stricter color validation
        calibrationSettings.confidenceThreshold = 0.8f; // Higher confidence required
        calibrationSettings.minPatchConfidence = 0.5f;
        
        ValidationSettings originalSettings = settings_;
        settings_ = calibrationSettings;
        
        ValidationResult result = validate(detection);
        
        settings_ = originalSettings;
        return result;
    }

    ValidationResult validateForRealTime(const Domain::DetectionResult& detection) {
        // Use more lenient settings for real-time processing
        ValidationSettings realtimeSettings = settings_;
        realtimeSettings.minRequiredPatches = 16; // 70% of patches sufficient
        realtimeSettings.maxDeltaE = 60.0f; // More lenient color validation
        realtimeSettings.confidenceThreshold = 0.5f; // Lower confidence acceptable
        
        ValidationSettings originalSettings = settings_;
        settings_ = realtimeSettings;
        
        ValidationResult result = validate(detection);
        
        settings_ = originalSettings;
        return result;
    }

  private:
    ValidationSettings settings_;

    float validateGeometry(const Domain::DetectionResult& detection, ValidationResult& result) {
        float score = 0.0f;
        
        const std::vector<Types::Point2D>& corners = detection.getCorners();
        
        if (corners.size() != 4) {
            result.warnings.push_back("ColorChecker corners not properly detected");
            return 0.0f;
        }

        try {
            // Calculate area
            float area = cv::contourArea(corners);
            result.colorCheckerArea = area;
            
            if (area < settings_.minColorCheckerArea) {
                result.warnings.push_back("ColorChecker area too small: " + std::to_string(area));
                score *= 0.5f;
            } else if (area > settings_.maxColorCheckerArea) {
                result.warnings.push_back("ColorChecker area too large: " + std::to_string(area));
                score *= 0.5f;
            } else {
                score += 0.3f; // Good area
            }

            // Calculate aspect ratio
            cv::Rect boundingRect = cv::boundingRect(corners);
            float aspectRatio = static_cast<float>(boundingRect.width) / boundingRect.height;
            result.aspectRatio = aspectRatio;
            
            if (aspectRatio >= settings_.minAspectRatio && aspectRatio <= settings_.maxAspectRatio) {
                score += 0.3f; // Good aspect ratio
            } else {
                result.warnings.push_back("Aspect ratio out of range: " + std::to_string(aspectRatio));
                score += 0.1f; // Partial credit
            }

            // Check for rectangularity (how close to a rectangle)
            float rectangularityScore = calculateRectangularity(corners);
            score += rectangularityScore * 0.2f;
            
            if (rectangularityScore < 0.7f) {
                result.warnings.push_back("ColorChecker shape not sufficiently rectangular");
            }

            // Check skew angle
            float skewAngle = calculateSkewAngle(corners);
            if (skewAngle <= settings_.maxSkewAngle) {
                score += 0.2f; // Good alignment
            } else {
                result.warnings.push_back("ColorChecker too skewed: " + std::to_string(skewAngle) + " degrees");
                score += 0.05f; // Minimal credit for high skew
            }

        } catch (const cv::Exception& e) {
            result.errors.push_back("Geometry validation error: " + std::string(e.what()));
            return 0.0f;
        }

        return std::clamp(score, 0.0f, 1.0f);
    }

    float validatePatchQuality(const Domain::DetectionResult& detection, ValidationResult& result) {
        const std::vector<Domain::ColorPatch>& patches = detection.getPatches();
        
        if (patches.empty()) {
            result.errors.push_back("No patches detected");
            return 0.0f;
        }

        int validPatches = 0;
        float totalConfidence = 0.0f;
        
        for (const auto& patch : patches) {
            if (patch.isValid() && patch.getConfidence().value >= settings_.minPatchConfidence) {
                validPatches++;
                totalConfidence += patch.getConfidence().value;
            }
        }

        result.validPatchCount = validPatches;
        result.totalPatchCount = static_cast<int>(patches.size());
        result.averagePatchConfidence = validPatches > 0 ? totalConfidence / validPatches : 0.0f;

        float completenessScore = static_cast<float>(validPatches) / Types::COLORCHECKER_PATCHES;
        float qualityScore = result.averagePatchConfidence;

        if (validPatches < settings_.minRequiredPatches) {
            result.errors.push_back("Insufficient valid patches: " + std::to_string(validPatches) + 
                                  "/" + std::to_string(Types::COLORCHECKER_PATCHES));
        }

        if (result.averagePatchConfidence < 0.5f) {
            result.warnings.push_back("Low average patch confidence: " + 
                                    std::to_string(result.averagePatchConfidence));
        }

        return (completenessScore + qualityScore) * 0.5f;
    }

    float validateColorConsistency(const Domain::DetectionResult& detection, ValidationResult& result) {
        const std::vector<Domain::ColorPatch>& patches = detection.getPatches();
        
        if (patches.empty()) {
            return 0.0f;
        }

        float totalDeltaE = 0.0f;
        int validPatches = 0;
        int inlierCount = 0;

        std::vector<float> deltaEValues;
        
        for (const auto& patch : patches) {
            if (!patch.isValid()) continue;
            
            float deltaE = patch.calculateDeltaE();
            deltaEValues.push_back(deltaE);
            totalDeltaE += deltaE;
            validPatches++;
            
            if (deltaE <= settings_.inlierThreshold) {
                inlierCount++;
            }
        }

        if (validPatches == 0) {
            result.errors.push_back("No valid patches for color consistency check");
            return 0.0f;
        }

        result.averageDeltaE = totalDeltaE / validPatches;
        
        // Calculate color consistency score
        float avgDeltaEScore = std::max(0.0f, 1.0f - (result.averageDeltaE / settings_.maxDeltaE));
        float inlierRatio = static_cast<float>(inlierCount) / validPatches;
        
        if (result.averageDeltaE > settings_.maxDeltaE) {
            result.warnings.push_back("High average Delta E: " + std::to_string(result.averageDeltaE));
        }

        if (inlierCount < settings_.minInlierPatches) {
            result.warnings.push_back("Too few inlier patches: " + std::to_string(inlierCount));
        }

        // Check for outliers
        if (!deltaEValues.empty()) {
            std::sort(deltaEValues.begin(), deltaEValues.end());
            float medianDeltaE = deltaEValues[deltaEValues.size() / 2];
            
            int outlierCount = 0;
            for (float deltaE : deltaEValues) {
                if (deltaE > medianDeltaE * 3.0f) { // 3x median threshold
                    outlierCount++;
                }
            }
            
            if (outlierCount > validPatches * 0.2f) { // More than 20% outliers
                result.warnings.push_back("High number of color outliers: " + std::to_string(outlierCount));
            }
        }

        return (avgDeltaEScore + inlierRatio) * 0.5f;
    }

    float validateSpatialConsistency(const Domain::DetectionResult& detection, ValidationResult& result) {
        const std::vector<Domain::ColorPatch>& patches = detection.getPatches();
        
        if (patches.size() < 4) {
            return 0.0f;
        }

        // Check if patches form a reasonable grid
        std::vector<Types::Point2D> centers;
        for (const auto& patch : patches) {
            if (patch.isValid()) {
                centers.push_back(patch.getCenter());
            }
        }

        if (centers.size() < settings_.minRequiredPatches) {
            return 0.0f;
        }

        float score = 0.0f;

        try {
            // Calculate grid regularity
            std::vector<float> horizontalSpacings, verticalSpacings;
            
            // Expected grid: 6 columns, 4 rows
            for (int row = 0; row < Types::COLORCHECKER_ROWS; ++row) {
                for (int col = 0; col < Types::COLORCHECKER_COLS - 1; ++col) {
                    int idx1 = row * Types::COLORCHECKER_COLS + col;
                    int idx2 = row * Types::COLORCHECKER_COLS + col + 1;
                    
                    if (idx1 < centers.size() && idx2 < centers.size()) {
                        float spacing = cv::norm(centers[idx1] - centers[idx2]);
                        horizontalSpacings.push_back(spacing);
                    }
                }
            }
            
            for (int col = 0; col < Types::COLORCHECKER_COLS; ++col) {
                for (int row = 0; row < Types::COLORCHECKER_ROWS - 1; ++row) {
                    int idx1 = row * Types::COLORCHECKER_COLS + col;
                    int idx2 = (row + 1) * Types::COLORCHECKER_COLS + col;
                    
                    if (idx1 < centers.size() && idx2 < centers.size()) {
                        float spacing = cv::norm(centers[idx1] - centers[idx2]);
                        verticalSpacings.push_back(spacing);
                    }
                }
            }

            // Calculate spacing consistency
            float horizontalConsistency = calculateSpacingConsistency(horizontalSpacings);
            float verticalConsistency = calculateSpacingConsistency(verticalSpacings);
            
            score = (horizontalConsistency + verticalConsistency) * 0.5f;
            
            if (horizontalConsistency < 0.7f || verticalConsistency < 0.7f) {
                result.warnings.push_back("Poor spatial consistency (H: " + 
                                        std::to_string(horizontalConsistency) + 
                                        ", V: " + std::to_string(verticalConsistency) + ")");
            }

        } catch (const std::exception& e) {
            result.warnings.push_back("Spatial consistency calculation error: " + std::string(e.what()));
            score = 0.5f; // Default moderate score on error
        }

        return std::clamp(score, 0.0f, 1.0f);
    }

    float calculateOverallScore(const ValidationResult& result) {
        // Weighted combination of individual scores
        float weights[] = {0.3f, 0.3f, 0.2f, 0.2f}; // geometry, patch quality, color, spatial
        float scores[] = {result.geometricScore, result.patchQualityScore, 
                         result.colorConsistencyScore, result.spatialConsistencyScore};
        
        float weightedSum = 0.0f;
        for (int i = 0; i < 4; ++i) {
            weightedSum += weights[i] * scores[i];
        }
        
        // Penalty for errors
        if (!result.errors.empty()) {
            weightedSum *= 0.5f; // 50% penalty for errors
        }
        
        // Penalty for too many warnings
        if (result.warnings.size() > 3) {
            weightedSum *= 0.8f; // 20% penalty for many warnings
        }
        
        return std::clamp(weightedSum, 0.0f, 1.0f);
    }

    void populateStatistics(const Domain::DetectionResult& detection, ValidationResult& result) {
        // Statistics are mostly populated during validation
        // Add any missing statistics here
        
        if (result.validPatchCount == 0 && !detection.getPatches().empty()) {
            // Recalculate if not done in patch validation
            for (const auto& patch : detection.getPatches()) {
                if (patch.isValid()) {
                    result.validPatchCount++;
                }
            }
        }
    }

    float calculateRectangularity(const std::vector<Types::Point2D>& corners) {
        if (corners.size() != 4) return 0.0f;

        // Calculate angles at each corner
        std::vector<float> angles;
        for (int i = 0; i < 4; ++i) {
            Types::Point2D p1 = corners[i];
            Types::Point2D p2 = corners[(i + 1) % 4];
            Types::Point2D p3 = corners[(i + 2) % 4];
            
            Types::Point2D v1 = p1 - p2;
            Types::Point2D v2 = p3 - p2;
            
            float dot = v1.x * v2.x + v1.y * v2.y;
            float mag1 = std::sqrt(v1.x * v1.x + v1.y * v1.y);
            float mag2 = std::sqrt(v2.x * v2.x + v2.y * v2.y);
            
            if (mag1 > 1e-6 && mag2 > 1e-6) {
                float angle = std::acos(std::clamp(dot / (mag1 * mag2), -1.0f, 1.0f));
                angles.push_back(angle * 180.0f / CV_PI);
            }
        }

        if (angles.size() != 4) return 0.0f;

        // Score based on how close angles are to 90 degrees
        float score = 0.0f;
        for (float angle : angles) {
            float deviation = std::abs(angle - 90.0f);
            score += std::max(0.0f, 1.0f - deviation / 30.0f); // 30 degree tolerance
        }
        
        return score / 4.0f;
    }

    float calculateSkewAngle(const std::vector<Types::Point2D>& corners) {
        if (corners.size() != 4) return 90.0f; // Maximum skew

        // Calculate angle of top edge relative to horizontal
        Types::Point2D topLeft = corners[0];
        Types::Point2D topRight = corners[1];
        
        Types::Point2D edge = topRight - topLeft;
        float angle = std::atan2(edge.y, edge.x) * 180.0f / CV_PI;
        
        return std::abs(angle);
    }

    float calculateSpacingConsistency(const std::vector<float>& spacings) {
        if (spacings.empty()) return 0.0f;
        if (spacings.size() == 1) return 1.0f;

        // Calculate mean and standard deviation
        float mean = 0.0f;
        for (float spacing : spacings) {
            mean += spacing;
        }
        mean /= spacings.size();

        float variance = 0.0f;
        for (float spacing : spacings) {
            float diff = spacing - mean;
            variance += diff * diff;
        }
        variance /= spacings.size();

        float stddev = std::sqrt(variance);
        float coefficientOfVariation = mean > 1e-6 ? stddev / mean : 1.0f;

        // Good consistency if CV < maxPatchDistanceVariation
        return std::max(0.0f, 1.0f - coefficientOfVariation / settings_.maxPatchDistanceVariation);
    }
};

}  // namespace ColorCorrection::Internal::Detection