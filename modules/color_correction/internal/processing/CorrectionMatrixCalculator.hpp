#pragma once

#include "../domain/ColorPatch.hpp"
#include "../domain/CorrectionMatrix.hpp"
#include "ColorSpaceConverter.hpp"
#include <shared/types/Common.hpp>
#include <shared/utils/Logger.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

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
        CalculationMethod method = CalculationMethod::SVD_LEAST_SQUARES;
        
        // Regularization parameters
        bool enableRegularization = true;
        float regularizationFactor = 1e-6f;
        float ridgeAlpha = 0.01f;
        
        // Robustness parameters
        bool enableOutlierRemoval = true;
        float outlierThreshold = 2.0f;      // Delta E threshold for outliers
        int minInlierPatches = 18;          // Minimum inliers for valid solution
        
        // RANSAC parameters
        int ransacIterations = 1000;
        float ransacThreshold = 1.5f;       // Delta E threshold for RANSAC inliers
        float ransacConfidence = 0.99f;
        
        // Validation parameters
        bool enableCrossValidation = true;
        int crossValidationFolds = 5;
        float maxConditionNumber = 1000.0f;
        float minDeterminant = 1e-6f;
        
        // Color space settings
        bool useLinearRGB = true;           // Work in linear RGB space
        bool normalizeInputs = false;       // Normalize input colors
        bool constrainPositivity = false;   // Constrain matrix to positive values
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
        
        bool isValid() const {
            return success && matrix.isValid() && 
                   conditionNumber < 1000.0f && std::abs(determinant) > 1e-6f;
        }
    };

    CorrectionMatrixCalculator(const CalculationSettings& settings = CalculationSettings{})
        : settings_(settings), colorConverter_() {
        LOG_INFO("Correction matrix calculator initialized with method ", 
                static_cast<int>(settings_.method));
    }

    CalculationResult calculateMatrix(const std::vector<Domain::ColorPatch>& patches) {
        CalculationResult result;
        result.totalPatches = static_cast<int>(patches.size());
        
        if (patches.empty()) {
            result.errorMessage = "No patches provided";
            LOG_ERROR(result.errorMessage);
            return result;
        }

        try {
            LOG_INFO("Calculating correction matrix from ", patches.size(), " patches");

            // Prepare data matrices
            cv::Mat measuredColors, referenceColors, weights;
            if (!prepareDataMatrices(patches, measuredColors, referenceColors, weights)) {
                result.errorMessage = "Failed to prepare data matrices";
                return result;
            }

            result.usedPatches = measuredColors.rows;
            
            if (result.usedPatches < 3) {
                result.errorMessage = "Insufficient valid patches for calculation";
                return result;
            }

            // Remove outliers if enabled
            if (settings_.enableOutlierRemoval) {
                removeOutliers(measuredColors, referenceColors, weights, result);
            }

            if (result.usedPatches < settings_.minInlierPatches) {
                result.errorMessage = "Too few inlier patches after outlier removal";
                return result;
            }

            // Calculate matrix using selected method
            cv::Mat correctionMatrix;
            switch (settings_.method) {
                case CalculationMethod::LEAST_SQUARES:
                    correctionMatrix = calculateLeastSquares(measuredColors, referenceColors);
                    break;
                case CalculationMethod::SVD_LEAST_SQUARES:
                    correctionMatrix = calculateSVDLeastSquares(measuredColors, referenceColors);
                    break;
                case CalculationMethod::WEIGHTED_LEAST_SQUARES:
                    correctionMatrix = calculateWeightedLeastSquares(measuredColors, referenceColors, weights);
                    break;
                case CalculationMethod::RIDGE_REGRESSION:
                    correctionMatrix = calculateRidgeRegression(measuredColors, referenceColors);
                    break;
                case CalculationMethod::RANSAC_ROBUST:
                    correctionMatrix = calculateRANSAC(measuredColors, referenceColors, result);
                    break;
                default:
                    result.errorMessage = "Unknown calculation method";
                    return result;
            }

            if (correctionMatrix.empty()) {
                result.errorMessage = "Matrix calculation failed";
                return result;
            }

            // Convert to domain object
            Types::Matrix3x3 matrix3x3;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    matrix3x3(i, j) = correctionMatrix.at<float>(i, j);
                }
            }
            
            result.matrix = Domain::CorrectionMatrix(matrix3x3);
            
            if (!result.matrix.isValid()) {
                result.errorMessage = "Calculated matrix failed validation";
                return result;
            }

            // Calculate quality metrics
            calculateQualityMetrics(measuredColors, referenceColors, correctionMatrix, result);
            
            // Cross-validation if enabled
            if (settings_.enableCrossValidation && patches.size() > settings_.crossValidationFolds) {
                result.crossValidationScore = performCrossValidation(patches);
            }

            result.success = true;
            LOG_INFO("Matrix calculation successful. MSE: ", result.meanSquareError, 
                    ", Condition: ", result.conditionNumber);

        } catch (const cv::Exception& e) {
            result.errorMessage = "OpenCV exception: " + std::string(e.what());
            LOG_ERROR(result.errorMessage);
        } catch (const std::exception& e) {
            result.errorMessage = "Exception: " + std::string(e.what());
            LOG_ERROR(result.errorMessage);
        }

        return result;
    }

    // Calculate matrix with manual color specification (for testing)
    CalculationResult calculateMatrix(const std::vector<Types::ColorValue>& measuredColors,
                                    const std::vector<Types::ColorValue>& referenceColors) {
        if (measuredColors.size() != referenceColors.size()) {
            CalculationResult result;
            result.errorMessage = "Measured and reference color counts don't match";
            return result;
        }

        // Convert to patches for processing
        std::vector<Domain::ColorPatch> patches;
        for (size_t i = 0; i < measuredColors.size(); ++i) {
            patches.emplace_back(
                static_cast<int>(i),
                Types::Point2D(0, 0), // Dummy center
                measuredColors[i],
                referenceColors[i],
                Types::ConfidenceScore::fromValue(1.0f)
            );
        }

        return calculateMatrix(patches);
    }

    // Optimize existing matrix using additional data
    CalculationResult refineMatrix(const Domain::CorrectionMatrix& initialMatrix,
                                 const std::vector<Domain::ColorPatch>& newPatches) {
        // This could implement iterative refinement algorithms
        // For now, just recalculate with combined data
        LOG_INFO("Refining matrix with ", newPatches.size(), " additional patches");
        return calculateMatrix(newPatches);
    }

    void setSettings(const CalculationSettings& settings) {
        settings_ = settings;
    }

    CalculationSettings getSettings() const {
        return settings_;
    }

    // Static utility for quick matrix calculation
    static Domain::CorrectionMatrix quickCalculate(const std::vector<Domain::ColorPatch>& patches) {
        CorrectionMatrixCalculator calculator;
        CalculationResult result = calculator.calculateMatrix(patches);
        return result.success ? result.matrix : Domain::CorrectionMatrix::identity();
    }

  private:
    CalculationSettings settings_;
    ColorSpaceConverter colorConverter_;

    bool prepareDataMatrices(const std::vector<Domain::ColorPatch>& patches,
                           cv::Mat& measuredColors, cv::Mat& referenceColors, cv::Mat& weights) {
        std::vector<Types::ColorValue> measuredList, referenceList;
        std::vector<float> weightList;

        for (const auto& patch : patches) {
            if (!patch.isValid()) {
                continue;
            }

            Types::ColorValue measured = patch.getMeasuredColor();
            Types::ColorValue reference = patch.getReferenceColor();

            // Convert to linear RGB if enabled
            if (settings_.useLinearRGB) {
                measured = colorConverter_.sRGBToLinear(measured);
                reference = colorConverter_.sRGBToLinear(reference);
            }

            // Normalize if enabled
            if (settings_.normalizeInputs) {
                measured = normalizeColor(measured);
                reference = normalizeColor(reference);
            }

            measuredList.push_back(measured);
            referenceList.push_back(reference);
            weightList.push_back(patch.getConfidence().value);
        }

        if (measuredList.empty()) {
            LOG_ERROR("No valid patches for matrix calculation");
            return false;
        }

        // Convert to OpenCV matrices
        measuredColors = cv::Mat(static_cast<int>(measuredList.size()), 3, CV_32F);
        referenceColors = cv::Mat(static_cast<int>(referenceList.size()), 3, CV_32F);
        weights = cv::Mat(static_cast<int>(weightList.size()), 1, CV_32F);

        for (size_t i = 0; i < measuredList.size(); ++i) {
            measuredColors.at<float>(i, 0) = measuredList[i][0];
            measuredColors.at<float>(i, 1) = measuredList[i][1];
            measuredColors.at<float>(i, 2) = measuredList[i][2];

            referenceColors.at<float>(i, 0) = referenceList[i][0];
            referenceColors.at<float>(i, 1) = referenceList[i][1];
            referenceColors.at<float>(i, 2) = referenceList[i][2];

            weights.at<float>(i, 0) = weightList[i];
        }

        LOG_DEBUG("Prepared data matrices: ", measuredColors.rows, " samples");
        return true;
    }

    cv::Mat calculateLeastSquares(const cv::Mat& measured, const cv::Mat& reference) {
        // Solve: reference = measured * matrix^T
        // matrix^T = measured^+ * reference (where ^+ is pseudoinverse)
        
        cv::Mat measuredT;
        cv::transpose(measured, measuredT);
        
        cv::Mat matrixT;
        cv::solve(measuredT, reference.t(), matrixT, cv::DECOMP_NORMAL);
        
        cv::Mat matrix;
        cv::transpose(matrixT, matrix);
        
        return matrix;
    }

    cv::Mat calculateSVDLeastSquares(const cv::Mat& measured, const cv::Mat& reference) {
        // Use SVD for more stable solution
        cv::Mat u, w, vt;
        cv::SVD::compute(measured, w, u, vt, cv::SVD::FULL_UV);
        
        // Apply regularization to singular values
        for (int i = 0; i < w.rows; ++i) {
            float& singularValue = w.at<float>(i, 0);
            if (singularValue < settings_.regularizationFactor) {
                singularValue = settings_.regularizationFactor;
            }
        }
        
        // Reconstruct pseudoinverse
        cv::Mat wInv = cv::Mat::zeros(w.size(), CV_32F);
        for (int i = 0; i < w.rows; ++i) {
            wInv.at<float>(i, 0) = 1.0f / w.at<float>(i, 0);
        }
        
        cv::Mat pseudoInverse;
        cv::Mat wInvMat = cv::Mat::diag(wInv);
        pseudoInverse = vt.t() * wInvMat * u.t();
        
        // Calculate matrix: reference = measured * matrix^T
        cv::Mat matrixT = pseudoInverse * reference;
        
        cv::Mat matrix;
        cv::transpose(matrixT, matrix);
        
        return matrix;
    }

    cv::Mat calculateWeightedLeastSquares(const cv::Mat& measured, const cv::Mat& reference, const cv::Mat& weights) {
        // Apply weights to the least squares problem
        cv::Mat weightedMeasured = cv::Mat::zeros(measured.size(), CV_32F);
        cv::Mat weightedReference = cv::Mat::zeros(reference.size(), CV_32F);
        
        for (int i = 0; i < measured.rows; ++i) {
            float weight = std::sqrt(weights.at<float>(i, 0)); // Use sqrt for proper weighting
            
            for (int j = 0; j < measured.cols; ++j) {
                weightedMeasured.at<float>(i, j) = measured.at<float>(i, j) * weight;
                weightedReference.at<float>(i, j) = reference.at<float>(i, j) * weight;
            }
        }
        
        return calculateSVDLeastSquares(weightedMeasured, weightedReference);
    }

    cv::Mat calculateRidgeRegression(const cv::Mat& measured, const cv::Mat& reference) {
        // Ridge regression: (X^T X + α I)^-1 X^T y
        cv::Mat measuredT;
        cv::transpose(measured, measuredT);
        
        cv::Mat XTX = measuredT * measured;
        
        // Add regularization term
        cv::Mat identity = cv::Mat::eye(XTX.size(), CV_32F);
        XTX += settings_.ridgeAlpha * identity;
        
        cv::Mat XTy = measuredT * reference;
        
        cv::Mat matrixT;
        cv::solve(XTX, XTy, matrixT, cv::DECOMP_CHOLESKY);
        
        cv::Mat matrix;
        cv::transpose(matrixT, matrix);
        
        return matrix;
    }

    cv::Mat calculateRANSAC(const cv::Mat& measured, const cv::Mat& reference, CalculationResult& result) {
        int numSamples = measured.rows;
        int sampleSize = std::min(9, numSamples / 2); // Use half the samples or 9, whichever is smaller
        
        cv::Mat bestMatrix;
        int bestInlierCount = 0;
        std::vector<int> bestInliers;
        
        std::random_device rd;
        std::mt19937 gen(rd());
        
        for (int iter = 0; iter < settings_.ransacIterations; ++iter) {
            // Random sample
            std::vector<int> sampleIndices;
            std::uniform_int_distribution<> dis(0, numSamples - 1);
            
            while (sampleIndices.size() < static_cast<size_t>(sampleSize)) {
                int idx = dis(gen);
                if (std::find(sampleIndices.begin(), sampleIndices.end(), idx) == sampleIndices.end()) {
                    sampleIndices.push_back(idx);
                }
            }
            
            // Create sample matrices
            cv::Mat sampleMeasured(sampleSize, 3, CV_32F);
            cv::Mat sampleReference(sampleSize, 3, CV_32F);
            
            for (int i = 0; i < sampleSize; ++i) {
                int idx = sampleIndices[i];
                for (int j = 0; j < 3; ++j) {
                    sampleMeasured.at<float>(i, j) = measured.at<float>(idx, j);
                    sampleReference.at<float>(i, j) = reference.at<float>(idx, j);
                }
            }
            
            // Calculate model
            cv::Mat candidateMatrix = calculateSVDLeastSquares(sampleMeasured, sampleReference);
            
            if (candidateMatrix.empty()) continue;
            
            // Count inliers
            std::vector<int> inliers;
            for (int i = 0; i < numSamples; ++i) {
                cv::Mat measuredRow = measured.row(i);
                cv::Mat referenceRow = reference.row(i);
                
                cv::Mat predicted = measuredRow * candidateMatrix.t();
                cv::Mat error = predicted - referenceRow;
                
                float deltaE = cv::norm(error, cv::NORM_L2);
                if (deltaE <= settings_.ransacThreshold) {
                    inliers.push_back(i);
                }
            }
            
            if (static_cast<int>(inliers.size()) > bestInlierCount) {
                bestInlierCount = static_cast<int>(inliers.size());
                bestMatrix = candidateMatrix.clone();
                bestInliers = inliers;
            }
            
            // Early termination if we have enough inliers
            float inlierRatio = static_cast<float>(inliers.size()) / numSamples;
            if (inlierRatio >= settings_.ransacConfidence) {
                break;
            }
        }
        
        // Refine using all inliers
        if (bestInlierCount >= settings_.minInlierPatches && !bestInliers.empty()) {
            cv::Mat inlierMeasured(bestInlierCount, 3, CV_32F);
            cv::Mat inlierReference(bestInlierCount, 3, CV_32F);
            
            for (int i = 0; i < bestInlierCount; ++i) {
                int idx = bestInliers[i];
                for (int j = 0; j < 3; ++j) {
                    inlierMeasured.at<float>(i, j) = measured.at<float>(idx, j);
                    inlierReference.at<float>(i, j) = reference.at<float>(idx, j);
                }
            }
            
            bestMatrix = calculateSVDLeastSquares(inlierMeasured, inlierReference);
            
            result.inlierIndices = bestInliers;
            result.iterations = settings_.ransacIterations;
        }
        
        return bestMatrix;
    }

    void removeOutliers(cv::Mat& measured, cv::Mat& reference, cv::Mat& weights, CalculationResult& result) {
        // Simple outlier removal based on color distance
        std::vector<int> validIndices;
        
        for (int i = 0; i < measured.rows; ++i) {
            cv::Mat measuredRow = measured.row(i);
            cv::Mat referenceRow = reference.row(i);
            
            float distance = cv::norm(measuredRow - referenceRow, cv::NORM_L2);
            
            if (distance <= settings_.outlierThreshold) {
                validIndices.push_back(i);
            } else {
                result.outlierIndices.push_back(i);
            }
        }
        
        if (validIndices.size() < static_cast<size_t>(measured.rows)) {
            // Create filtered matrices
            cv::Mat filteredMeasured(static_cast<int>(validIndices.size()), 3, CV_32F);
            cv::Mat filteredReference(static_cast<int>(validIndices.size()), 3, CV_32F);
            cv::Mat filteredWeights(static_cast<int>(validIndices.size()), 1, CV_32F);
            
            for (size_t i = 0; i < validIndices.size(); ++i) {
                int idx = validIndices[i];
                for (int j = 0; j < 3; ++j) {
                    filteredMeasured.at<float>(i, j) = measured.at<float>(idx, j);
                    filteredReference.at<float>(i, j) = reference.at<float>(idx, j);
                }
                filteredWeights.at<float>(i, 0) = weights.at<float>(idx, 0);
            }
            
            measured = filteredMeasured;
            reference = filteredReference;
            weights = filteredWeights;
            result.usedPatches = static_cast<int>(validIndices.size());
            
            LOG_DEBUG("Removed ", result.outlierIndices.size(), " outliers, ", 
                     result.usedPatches, " patches remaining");
        }
    }

    void calculateQualityMetrics(const cv::Mat& measured, const cv::Mat& reference, 
                               const cv::Mat& matrix, CalculationResult& result) {
        // Calculate residuals
        cv::Mat predicted = measured * matrix.t();
        cv::Mat residualMat = predicted - reference;
        
        float sumSquaredError = 0.0f;
        float maxError = 0.0f;
        
        for (int i = 0; i < residualMat.rows; ++i) {
            float error = cv::norm(residualMat.row(i), cv::NORM_L2);
            result.residuals.push_back(error);
            
            sumSquaredError += error * error;
            maxError = std::max(maxError, error);
        }
        
        result.meanSquareError = sumSquaredError / residualMat.rows;
        result.maxError = maxError;
        
        // Matrix condition number and determinant
        cv::Mat u, w, vt;
        cv::SVD::compute(matrix, w, u, vt);
        
        float maxSingular = 0.0f, minSingular = std::numeric_limits<float>::max();
        for (int i = 0; i < w.rows; ++i) {
            float sv = w.at<float>(i, 0);
            maxSingular = std::max(maxSingular, sv);
            minSingular = std::min(minSingular, sv);
        }
        
        result.conditionNumber = (minSingular > 1e-10f) ? maxSingular / minSingular : 
                                std::numeric_limits<float>::max();
        result.determinant = cv::determinant(matrix);
        
        LOG_DEBUG("Quality metrics - MSE: ", result.meanSquareError, 
                 ", Max Error: ", result.maxError, 
                 ", Condition: ", result.conditionNumber,
                 ", Det: ", result.determinant);
    }

    float performCrossValidation(const std::vector<Domain::ColorPatch>& patches) {
        int numFolds = settings_.crossValidationFolds;
        int foldSize = static_cast<int>(patches.size()) / numFolds;
        
        float totalError = 0.0f;
        int validFolds = 0;
        
        for (int fold = 0; fold < numFolds; ++fold) {
            // Split data
            std::vector<Domain::ColorPatch> trainPatches, testPatches;
            
            for (size_t i = 0; i < patches.size(); ++i) {
                if (static_cast<int>(i) >= fold * foldSize && 
                    static_cast<int>(i) < (fold + 1) * foldSize) {
                    testPatches.push_back(patches[i]);
                } else {
                    trainPatches.push_back(patches[i]);
                }
            }
            
            if (trainPatches.size() < 3 || testPatches.empty()) {
                continue;
            }
            
            // Train on training set
            CalculationResult trainResult = calculateMatrix(trainPatches);
            if (!trainResult.success) {
                continue;
            }
            
            // Test on test set
            float foldError = 0.0f;
            for (const auto& patch : testPatches) {
                Types::ColorValue measured = patch.getMeasuredColor();
                Types::ColorValue reference = patch.getReferenceColor();
                
                if (settings_.useLinearRGB) {
                    measured = colorConverter_.sRGBToLinear(measured);
                    reference = colorConverter_.sRGBToLinear(reference);
                }
                
                Types::ColorValue corrected = trainResult.matrix.applyCorrection(measured);
                float error = cv::norm(corrected - reference, cv::NORM_L2);
                foldError += error;
            }
            
            foldError /= testPatches.size();
            totalError += foldError;
            validFolds++;
        }
        
        float crossValidationError = (validFolds > 0) ? totalError / validFolds : 
                                   std::numeric_limits<float>::max();
        
        LOG_DEBUG("Cross-validation error: ", crossValidationError, " (", validFolds, " folds)");
        return crossValidationError;
    }

    Types::ColorValue normalizeColor(const Types::ColorValue& color) {
        float norm = cv::norm(color, cv::NORM_L2);
        return (norm > 1e-6f) ? color / norm : color;
    }
};

}  // namespace ColorCorrection::Internal::Processing