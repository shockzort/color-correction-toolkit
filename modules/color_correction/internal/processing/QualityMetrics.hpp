#pragma once

#include "../domain/ColorPatch.hpp"
#include "../domain/CorrectionMatrix.hpp"
#include "ColorSpaceConverter.hpp"
#include <shared/types/Common.hpp>
#include <shared/utils/Logger.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iomanip>

namespace ColorCorrection::Internal::Processing {

class QualityMetrics {
  public:
    enum class DeltaEFormula {
        CIE76,          // Original CIE76 Delta E
        CIE94,          // CIE94 Delta E (improved)
        CIEDE2000,      // CIE DE2000 (most accurate, default)
        CMC,            // CMC l:c Delta E
        EUCLIDEAN_RGB,  // Simple Euclidean distance in RGB
        EUCLIDEAN_LAB   // Simple Euclidean distance in LAB
    };

    struct QualityReport {
        // Color accuracy metrics
        float averageDeltaE = 0.0f;
        float medianDeltaE = 0.0f;
        float maxDeltaE = 0.0f;
        float minDeltaE = 0.0f;
        float percentile95DeltaE = 0.0f;
        
        // Statistical metrics
        float rmse = 0.0f;                      // Root Mean Square Error
        float mae = 0.0f;                       // Mean Absolute Error
        float mape = 0.0f;                      // Mean Absolute Percentage Error
        float r2Score = 0.0f;                   // R-squared coefficient
        
        // Per-channel metrics
        Types::ColorValue meanError{0, 0, 0};
        Types::ColorValue stdError{0, 0, 0};
        Types::ColorValue maxAbsError{0, 0, 0};
        
        // Distribution metrics
        std::vector<float> deltaEValues;
        std::vector<float> perChannelErrors[3];  // R, G, B errors
        
        // Matrix quality metrics
        float matrixConditionNumber = 0.0f;
        float matrixDeterminant = 0.0f;
        bool matrixIsWellConditioned = false;
        
        // Overall quality scores
        Types::ConfidenceScore colorAccuracyScore = Types::ConfidenceScore::fromValue(0.0f);
        Types::ConfidenceScore statisticalScore = Types::ConfidenceScore::fromValue(0.0f);
        Types::ConfidenceScore overallScore = Types::ConfidenceScore::fromValue(0.0f);
        
        // Classification
        std::string qualityGrade = "Unknown";   // Excellent, Good, Fair, Poor
        std::vector<std::string> recommendations;
        
        int totalPatches = 0;
        int validPatches = 0;
        
        bool isAcceptable(float deltaEThreshold = 2.0f) const;
        std::string getSummary() const;
    };

    struct ComparisonSettings {
        DeltaEFormula deltaEFormula;
        Types::ColorSpace workingColorSpace;
        
        // CIE94 parameters
        float cie94_kL;
        float cie94_kC;
        float cie94_kH;
        float cie94_K1;
        float cie94_K2;
        
        // CIEDE2000 parameters
        float ciede2000_kL;
        float ciede2000_kC;
        float ciede2000_kH;
        
        // CMC parameters
        float cmc_l;  // Lightness weight
        float cmc_c;  // Chroma weight
        
        // Quality thresholds
        float excellentThreshold;    // ΔE < 1.0 = excellent
        float goodThreshold;         // ΔE < 2.0 = good
        float fairThreshold;         // ΔE < 4.0 = fair
        
        // Statistical settings
        bool includeOutliers;
        float outlierThreshold;      // Standard deviations for outlier detection
        bool weightByConfidence;
        
        ComparisonSettings();
    };

    explicit QualityMetrics(const ComparisonSettings& settings = ComparisonSettings{});

    // Calculate comprehensive quality report
    QualityReport calculateQualityReport(const std::vector<Domain::ColorPatch>& patches,
                                       const Domain::CorrectionMatrix& correctionMatrix);

    // Calculate Delta E between two colors
    float calculateDeltaE(const Types::ColorValue& color1, const Types::ColorValue& color2,
                         DeltaEFormula formula = DeltaEFormula::CIEDE2000) const;

    // Batch Delta E calculation
    std::vector<float> calculateDeltaEBatch(const std::vector<Types::ColorValue>& colors1,
                                           const std::vector<Types::ColorValue>& colors2) const;

    // Compare two quality reports
    struct ComparisonResult {
        float deltaEImprovement = 0.0f;
        float rmseImprovement = 0.0f;
        float r2Improvement = 0.0f;
        bool isSignificantImprovement = false;
        std::string summary;
    };

    ComparisonResult compareQualityReports(const QualityReport& before, 
                                         const QualityReport& after) const;

    void setSettings(const ComparisonSettings& settings);
    ComparisonSettings getSettings() const;

  private:
    ComparisonSettings settings_;
    ColorSpaceConverter colorConverter_;

    void calculateDeltaEMetrics(const std::vector<Types::ColorValue>& corrected,
                               const std::vector<Types::ColorValue>& reference,
                               const std::vector<float>& confidences,
                               QualityReport& report);

    void calculateStatisticalMetrics(const std::vector<Types::ColorValue>& corrected,
                                   const std::vector<Types::ColorValue>& reference,
                                   const std::vector<float>& confidences,
                                   QualityReport& report);

    void calculatePerChannelMetrics(const std::vector<Types::ColorValue>& corrected,
                                  const std::vector<Types::ColorValue>& reference,
                                  QualityReport& report);

    void evaluateMatrixQuality(const Domain::CorrectionMatrix& matrix, QualityReport& report);

    void calculateOverallScores(QualityReport& report);

    void generateRecommendations(QualityReport& report);

    // Delta E calculation methods
    float calculateDeltaE76(const Types::ColorValue& color1, const Types::ColorValue& color2) const;

    float calculateDeltaE94(const Types::ColorValue& color1, const Types::ColorValue& color2) const;

    float calculateDeltaE2000(const Types::ColorValue& color1, const Types::ColorValue& color2) const;

    float calculateDeltaECMC(const Types::ColorValue& color1, const Types::ColorValue& color2) const;

    float calculateEuclideanRGB(const Types::ColorValue& color1, const Types::ColorValue& color2) const;

    float calculateEuclideanLAB(const Types::ColorValue& color1, const Types::ColorValue& color2) const;
};

}  // namespace ColorCorrection::Internal::Processing