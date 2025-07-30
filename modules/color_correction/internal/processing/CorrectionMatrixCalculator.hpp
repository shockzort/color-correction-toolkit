#pragma once

#include "../domain/ColorPatch.hpp"
#include "../domain/CorrectionMatrix.hpp"
#include "ColorSpaceConverter.hpp"
#include <shared/types/Common.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

namespace ColorCorrection::Internal::Processing {

class CorrectionMatrixCalculator {
  public:
    enum class CalculationMethod {
        LEAST_SQUARES,          // Basic least squares
        SVD_LEAST_SQUARES,      // SVD-based least squares (recommended)
        WEIGHTED_LEAST_SQUARES, // Confidence-weighted least squares
        RIDGE_REGRESSION,       // Ridge regression with regularization
        RANSAC_ROBUST          // RANSAC-based robust estimation
    };

    struct CalculationSettings {
        CalculationMethod method;
        
        // Regularization parameters
        bool enableRegularization;
        float regularizationFactor;
        float ridgeAlpha;
        
        // Robustness parameters
        bool enableOutlierRemoval;
        float outlierThreshold;      // Delta E threshold for outliers
        int minInlierPatches;          // Minimum inliers for valid solution
        
        // RANSAC parameters
        int ransacIterations;
        float ransacThreshold;       // Delta E threshold for RANSAC inliers
        float ransacConfidence;
        
        // Validation parameters
        bool enableCrossValidation;
        int crossValidationFolds;
        float maxConditionNumber;
        float minDeterminant;
        
        // Color space settings
        bool useLinearRGB;           // Work in linear RGB space
        bool normalizeInputs;       // Normalize input colors
        bool constrainPositivity;   // Constrain matrix to positive values
        
        CalculationSettings();
    };

    struct CalculationResult {
        Domain::CorrectionMatrix matrix;
        bool success = false;
        
        // Quality metrics
        float meanSquareError = 0.0f;
        float maxError = 0.0f;
        float conditionNumber = 0.0f;
        float determinant = 0.0f;
        
        // Validation results
        std::vector<float> residuals;
        std::vector<int> inlierIndices;
        std::vector<int> outlierIndices;
        float crossValidationScore = 0.0f;
        
        // Statistics
        int totalPatches = 0;
        int usedPatches = 0;
        int iterations = 0;
        
        std::string errorMessage;
        
        CalculationResult();
        bool isValid() const;
    };

    explicit CorrectionMatrixCalculator(const CalculationSettings& settings = CalculationSettings{});

    CalculationResult calculateMatrix(const std::vector<Domain::ColorPatch>& patches);

    // Calculate matrix with manual color specification (for testing)
    CalculationResult calculateMatrix(const std::vector<Types::ColorValue>& measuredColors,
                                    const std::vector<Types::ColorValue>& referenceColors);

    // Optimize existing matrix using additional data
    CalculationResult refineMatrix(const Domain::CorrectionMatrix& initialMatrix,
                                 const std::vector<Domain::ColorPatch>& newPatches);

    void setSettings(const CalculationSettings& settings);
    CalculationSettings getSettings() const;

    // Static utility for quick matrix calculation
    static Domain::CorrectionMatrix quickCalculate(const std::vector<Domain::ColorPatch>& patches);

  private:
    CalculationSettings settings_;
    ColorSpaceConverter colorConverter_;

    bool prepareDataMatrices(const std::vector<Domain::ColorPatch>& patches,
                           cv::Mat& measuredColors, cv::Mat& referenceColors, cv::Mat& weights);

    cv::Mat calculateLeastSquares(const cv::Mat& measured, const cv::Mat& reference);
    cv::Mat calculateSVDLeastSquares(const cv::Mat& measured, const cv::Mat& reference);
    cv::Mat calculateWeightedLeastSquares(const cv::Mat& measured, const cv::Mat& reference, const cv::Mat& weights);
    cv::Mat calculateRidgeRegression(const cv::Mat& measured, const cv::Mat& reference);
    cv::Mat calculateRANSAC(const cv::Mat& measured, const cv::Mat& reference, CalculationResult& result);

    void removeOutliers(cv::Mat& measured, cv::Mat& reference, cv::Mat& weights, CalculationResult& result);

    void calculateQualityMetrics(const cv::Mat& measured, const cv::Mat& reference, 
                               const cv::Mat& matrix, CalculationResult& result);

    float performCrossValidation(const std::vector<Domain::ColorPatch>& patches);

    Types::ColorValue normalizeColor(const Types::ColorValue& color);
};

}  // namespace ColorCorrection::Internal::Processing