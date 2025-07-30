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
        float minColorCheckerArea;
        float maxColorCheckerArea;
        float minAspectRatio;
        float maxAspectRatio;
        float maxSkewAngle; // degrees
        
        // Patch validation
        int minRequiredPatches; // 80% of 24 patches
        float maxPatchDistanceVariation; // 30% variation in spacing
        float minPatchConfidence;
        
        // Color validation
        float maxDeltaE; // Very lenient for initial detection
        float maxBrightnessVariation; // Max variation in brightness
        bool validateColorConsistency;
        
        // Statistical validation
        float confidenceThreshold;
        int minInlierPatches; // For RANSAC-style validation
        float inlierThreshold; // Delta E threshold for inliers
        
        // Temporal validation (for video sequences)
        bool enableTemporalValidation;
        float maxFrameToFrameMovement; // pixels
        int temporalConsistencyFrames;
        
        ValidationSettings();
    };

    DetectionValidator(const ValidationSettings& settings = ValidationSettings{});

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

    ValidationResult validate(const Domain::DetectionResult& detection);

    Types::ConfidenceScore calculateConfidenceScore(const Domain::DetectionResult& detection);

    bool isDetectionAcceptable(const Domain::DetectionResult& detection);

    ValidationSettings getSettings() const;

    void setSettings(const ValidationSettings& settings);

    // Enhanced validation for specific use cases
    ValidationResult validateForCalibration(const Domain::DetectionResult& detection);

    ValidationResult validateForRealTime(const Domain::DetectionResult& detection);

  private:
    ValidationSettings settings_;

    float validateGeometry(const Domain::DetectionResult& detection, ValidationResult& result);
    float validatePatchQuality(const Domain::DetectionResult& detection, ValidationResult& result);
    float validateColorConsistency(const Domain::DetectionResult& detection, ValidationResult& result);
    float validateSpatialConsistency(const Domain::DetectionResult& detection, ValidationResult& result);
    float calculateOverallScore(const ValidationResult& result);
    void populateStatistics(const Domain::DetectionResult& detection, ValidationResult& result);
    float calculateRectangularity(const std::vector<Types::Point2D>& corners);
    float calculateSkewAngle(const std::vector<Types::Point2D>& corners);
    float calculateSpacingConsistency(const std::vector<float>& spacings);
};

}  // namespace ColorCorrection::Internal::Detection