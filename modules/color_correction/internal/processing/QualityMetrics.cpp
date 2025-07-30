#include "QualityMetrics.hpp"

namespace ColorCorrection::Internal::Processing {

// QualityReport methods
bool QualityMetrics::QualityReport::isAcceptable(float deltaEThreshold) const {
    return averageDeltaE <= deltaEThreshold && 
           percentile95DeltaE <= deltaEThreshold * 2.0f &&
           matrixIsWellConditioned;
}

std::string QualityMetrics::QualityReport::getSummary() const {
    std::ostringstream oss;
    oss << "Quality Report Summary:\n";
    oss << "  Grade: " << qualityGrade << "\n";
    oss << "  Average ΔE: " << std::fixed << std::setprecision(2) << averageDeltaE << "\n";
    oss << "  Max ΔE: " << maxDeltaE << "\n";
    oss << "  95th percentile ΔE: " << percentile95DeltaE << "\n";
    oss << "  RMSE: " << rmse << "\n";
    oss << "  R²: " << r2Score << "\n";
    oss << "  Valid patches: " << validPatches << "/" << totalPatches << "\n";
    oss << "  Overall score: " << overallScore.value << "\n";
    return oss.str();
}

// ComparisonSettings constructor
QualityMetrics::ComparisonSettings::ComparisonSettings()
    : deltaEFormula(DeltaEFormula::CIEDE2000),
      workingColorSpace(Types::ColorSpace::LAB),
      cie94_kL(1.0f),
      cie94_kC(1.0f),
      cie94_kH(1.0f),
      cie94_K1(0.045f),
      cie94_K2(0.015f),
      ciede2000_kL(1.0f),
      ciede2000_kC(1.0f),
      ciede2000_kH(1.0f),
      cmc_l(2.0f),
      cmc_c(1.0f),
      excellentThreshold(1.0f),
      goodThreshold(2.0f),
      fairThreshold(4.0f),
      includeOutliers(true),
      outlierThreshold(3.0f),
      weightByConfidence(true) {}

// QualityMetrics constructor
QualityMetrics::QualityMetrics(const ComparisonSettings& settings)
    : settings_(settings), colorConverter_() {
    LOG_INFO("Quality metrics initialized with Delta E formula ", 
            static_cast<int>(settings_.deltaEFormula));
}

// Calculate comprehensive quality report
QualityMetrics::QualityReport QualityMetrics::calculateQualityReport(
    const std::vector<Domain::ColorPatch>& patches,
    const Domain::CorrectionMatrix& correctionMatrix) {
    QualityReport report;
    report.totalPatches = static_cast<int>(patches.size());
    
    if (patches.empty()) {
        LOG_ERROR("No patches provided for quality assessment");
        return report;
    }

    try {
        LOG_DEBUG("Calculating quality metrics for ", patches.size(), " patches");

        // Collect valid patches and calculate errors
        std::vector<Types::ColorValue> measuredColors, referenceColors, correctedColors;
        std::vector<float> confidences;
        
        for (const auto& patch : patches) {
            if (!patch.isValid()) continue;
            
            Types::ColorValue measured = patch.getMeasuredColor();
            Types::ColorValue reference = patch.getReferenceColor();
            Types::ColorValue corrected = correctionMatrix.applyCorrection(measured);
            
            measuredColors.push_back(measured);
            referenceColors.push_back(reference);
            correctedColors.push_back(corrected);
            confidences.push_back(patch.getConfidence().value);
            report.validPatches++;
        }

        if (report.validPatches == 0) {
            LOG_ERROR("No valid patches for quality assessment");
            return report;
        }

        // Calculate Delta E values
        calculateDeltaEMetrics(correctedColors, referenceColors, confidences, report);
        
        // Calculate statistical metrics
        calculateStatisticalMetrics(correctedColors, referenceColors, confidences, report);
        
        // Calculate per-channel metrics
        calculatePerChannelMetrics(correctedColors, referenceColors, report);
        
        // Evaluate matrix quality
        evaluateMatrixQuality(correctionMatrix, report);
        
        // Calculate overall scores
        calculateOverallScores(report);
        
        // Generate recommendations
        generateRecommendations(report);
        
        LOG_INFO("Quality assessment complete. Average ΔE: ", report.averageDeltaE, 
                ", Grade: ", report.qualityGrade);

    } catch (const std::exception& e) {
        LOG_ERROR("Error calculating quality metrics: ", e.what());
    }

    return report;
}

// Calculate Delta E between two colors
float QualityMetrics::calculateDeltaE(const Types::ColorValue& color1, const Types::ColorValue& color2,
                                     DeltaEFormula formula) const {
    switch (formula) {
        case DeltaEFormula::CIE76:
            return calculateDeltaE76(color1, color2);
        case DeltaEFormula::CIE94:
            return calculateDeltaE94(color1, color2);
        case DeltaEFormula::CIEDE2000:
            return calculateDeltaE2000(color1, color2);
        case DeltaEFormula::CMC:
            return calculateDeltaECMC(color1, color2);
        case DeltaEFormula::EUCLIDEAN_RGB:
            return calculateEuclideanRGB(color1, color2);
        case DeltaEFormula::EUCLIDEAN_LAB:
            return calculateEuclideanLAB(color1, color2);
        default:
            return calculateDeltaE2000(color1, color2);
    }
}

// Batch Delta E calculation
std::vector<float> QualityMetrics::calculateDeltaEBatch(const std::vector<Types::ColorValue>& colors1,
                                                       const std::vector<Types::ColorValue>& colors2) const {
    std::vector<float> deltaEs;
    
    size_t minSize = std::min(colors1.size(), colors2.size());
    deltaEs.reserve(minSize);
    
    for (size_t i = 0; i < minSize; ++i) {
        deltaEs.push_back(calculateDeltaE(colors1[i], colors2[i], settings_.deltaEFormula));
    }
    
    return deltaEs;
}

// Compare two quality reports
QualityMetrics::ComparisonResult QualityMetrics::compareQualityReports(const QualityReport& before, 
                                                                      const QualityReport& after) const {
    ComparisonResult result;
    
    result.deltaEImprovement = before.averageDeltaE - after.averageDeltaE;
    result.rmseImprovement = before.rmse - after.rmse;
    result.r2Improvement = after.r2Score - before.r2Score;
    
    // Consider improvement significant if Delta E improves by > 0.5 and R² improves by > 0.05
    result.isSignificantImprovement = (result.deltaEImprovement > 0.5f && result.r2Improvement > 0.05f);
    
    std::ostringstream oss;
    oss << "Quality Comparison:\n";
    oss << "  ΔE change: " << std::showpos << result.deltaEImprovement << "\n";
    oss << "  RMSE change: " << result.rmseImprovement << "\n";
    oss << "  R² change: " << result.r2Improvement << "\n";
    oss << "  Significant improvement: " << (result.isSignificantImprovement ? "Yes" : "No");
    
    result.summary = oss.str();
    return result;
}

void QualityMetrics::setSettings(const ComparisonSettings& settings) {
    settings_ = settings;
}

QualityMetrics::ComparisonSettings QualityMetrics::getSettings() const {
    return settings_;
}

// Private methods

void QualityMetrics::calculateDeltaEMetrics(const std::vector<Types::ColorValue>& corrected,
                                           const std::vector<Types::ColorValue>& reference,
                                           const std::vector<float>& confidences,
                                           QualityReport& report) {
    std::vector<float> deltaEs;
    float weightedSum = 0.0f;
    float totalWeight = 0.0f;
    
    for (size_t i = 0; i < corrected.size(); ++i) {
        float deltaE = calculateDeltaE(corrected[i], reference[i], settings_.deltaEFormula);
        deltaEs.push_back(deltaE);
        
        float weight = settings_.weightByConfidence ? confidences[i] : 1.0f;
        weightedSum += deltaE * weight;
        totalWeight += weight;
    }
    
    report.deltaEValues = deltaEs;
    
    // Calculate statistics
    std::sort(deltaEs.begin(), deltaEs.end());
    
    report.averageDeltaE = (totalWeight > 0.0f) ? weightedSum / totalWeight : 0.0f;
    report.medianDeltaE = deltaEs[deltaEs.size() / 2];
    report.minDeltaE = deltaEs.front();
    report.maxDeltaE = deltaEs.back();
    
    // 95th percentile
    size_t p95Index = static_cast<size_t>(deltaEs.size() * 0.95);
    report.percentile95DeltaE = deltaEs[std::min(p95Index, deltaEs.size() - 1)];
    
    LOG_DEBUG("Delta E metrics - Avg: ", report.averageDeltaE, 
             ", Median: ", report.medianDeltaE, 
             ", Max: ", report.maxDeltaE);
}

void QualityMetrics::calculateStatisticalMetrics(const std::vector<Types::ColorValue>& corrected,
                                                 const std::vector<Types::ColorValue>& reference,
                                                 const std::vector<float>& confidences,
                                                 QualityReport& report) {
    float sumSquaredError = 0.0f;
    float sumAbsoluteError = 0.0f;
    float sumPercentageError = 0.0f;
    float totalWeight = 0.0f;
    
    // Calculate mean of reference colors for R² calculation
    Types::ColorValue referenceMean(0, 0, 0);
    for (const auto& ref : reference) {
        referenceMean += ref;
    }
    referenceMean /= static_cast<float>(reference.size());
    
    float totalSumSquares = 0.0f;  // For R²
    
    for (size_t i = 0; i < corrected.size(); ++i) {
        Types::ColorValue error = corrected[i] - reference[i];
        float weight = settings_.weightByConfidence ? confidences[i] : 1.0f;
        
        // Squared error
        float squaredError = cv::norm(error, cv::NORM_L2SQR);
        sumSquaredError += squaredError * weight;
        
        // Absolute error
        float absoluteError = cv::norm(error, cv::NORM_L1);
        sumAbsoluteError += absoluteError * weight;
        
        // Percentage error (avoid division by zero)
        Types::ColorValue refNormalized = reference[i];
        for (int c = 0; c < 3; ++c) {
            if (std::abs(refNormalized[c]) > 1e-6f) {
                sumPercentageError += std::abs(error[c] / refNormalized[c]) * weight;
            }
        }
        
        // Total sum of squares for R²
        Types::ColorValue refDeviation = reference[i] - referenceMean;
        totalSumSquares += cv::norm(refDeviation, cv::NORM_L2SQR) * weight;
        
        totalWeight += weight;
    }
    
    if (totalWeight > 0.0f) {
        report.rmse = std::sqrt(sumSquaredError / totalWeight);
        report.mae = sumAbsoluteError / totalWeight;
        report.mape = (sumPercentageError / (totalWeight * 3.0f)) * 100.0f; // 3 channels
        
        // R² calculation
        if (totalSumSquares > 1e-6f) {
            report.r2Score = 1.0f - (sumSquaredError / totalSumSquares);
        } else {
            report.r2Score = 0.0f;
        }
    }
    
    LOG_DEBUG("Statistical metrics - RMSE: ", report.rmse, 
             ", MAE: ", report.mae, 
             ", R²: ", report.r2Score);
}

void QualityMetrics::calculatePerChannelMetrics(const std::vector<Types::ColorValue>& corrected,
                                               const std::vector<Types::ColorValue>& reference,
                                               QualityReport& report) {
    Types::ColorValue sumError(0, 0, 0);
    Types::ColorValue sumSquaredError(0, 0, 0);
    Types::ColorValue maxAbsError(0, 0, 0);
    
    // Collect per-channel errors
    for (int c = 0; c < 3; ++c) {
        report.perChannelErrors[c].reserve(corrected.size());
    }
    
    for (size_t i = 0; i < corrected.size(); ++i) {
        Types::ColorValue error = corrected[i] - reference[i];
        
        for (int c = 0; c < 3; ++c) {
            float absError = std::abs(error[c]);
            
            sumError[c] += error[c];
            sumSquaredError[c] += error[c] * error[c];
            maxAbsError[c] = std::max(maxAbsError[c], absError);
            
            report.perChannelErrors[c].push_back(error[c]);
        }
    }
    
    float n = static_cast<float>(corrected.size());
    
    // Mean errors
    report.meanError = sumError / n;
    
    // Standard deviation of errors
    for (int c = 0; c < 3; ++c) {
        float variance = (sumSquaredError[c] / n) - (report.meanError[c] * report.meanError[c]);
        report.stdError[c] = std::sqrt(std::max(0.0f, variance));
    }
    
    report.maxAbsError = maxAbsError;
    
    LOG_DEBUG("Per-channel metrics calculated for RGB channels");
}

void QualityMetrics::evaluateMatrixQuality(const Domain::CorrectionMatrix& matrix, QualityReport& report) {
    report.matrixConditionNumber = matrix.getConditionNumber();
    report.matrixDeterminant = matrix.getDeterminant();
    
    // Well-conditioned matrix criteria
    report.matrixIsWellConditioned = (report.matrixConditionNumber < 100.0f && 
                                    std::abs(report.matrixDeterminant) > 1e-3f);
    
    LOG_DEBUG("Matrix quality - Condition: ", report.matrixConditionNumber, 
             ", Determinant: ", report.matrixDeterminant,
             ", Well-conditioned: ", report.matrixIsWellConditioned ? "Yes" : "No");
}

void QualityMetrics::calculateOverallScores(QualityReport& report) {
    // Color accuracy score (based on Delta E)
    float colorScore = 1.0f;
    if (report.averageDeltaE > settings_.excellentThreshold) {
        colorScore = std::max(0.0f, 1.0f - (report.averageDeltaE - settings_.excellentThreshold) / 
                                           (settings_.fairThreshold - settings_.excellentThreshold));
    }
    report.colorAccuracyScore = Types::ConfidenceScore::fromValue(colorScore);
    
    // Statistical score (based on R² and RMSE)
    float statisticalScore = report.r2Score;
    if (report.rmse > 0.1f) {
        statisticalScore *= std::max(0.2f, 1.0f - report.rmse);
    }
    report.statisticalScore = Types::ConfidenceScore::fromValue(std::clamp(statisticalScore, 0.0f, 1.0f));
    
    // Overall score (weighted combination)
    float overallScore = (colorScore * 0.6f + statisticalScore * 0.3f + 
                        (report.matrixIsWellConditioned ? 0.1f : 0.0f));
    report.overallScore = Types::ConfidenceScore::fromValue(overallScore);
    
    // Quality grade
    if (report.averageDeltaE <= settings_.excellentThreshold) {
        report.qualityGrade = "Excellent";
    } else if (report.averageDeltaE <= settings_.goodThreshold) {
        report.qualityGrade = "Good";
    } else if (report.averageDeltaE <= settings_.fairThreshold) {
        report.qualityGrade = "Fair";
    } else {
        report.qualityGrade = "Poor";
    }
    
    LOG_DEBUG("Overall scores - Color: ", colorScore, 
             ", Statistical: ", statisticalScore, 
             ", Overall: ", overallScore,
             ", Grade: ", report.qualityGrade);
}

void QualityMetrics::generateRecommendations(QualityReport& report) {
    report.recommendations.clear();
    
    if (report.averageDeltaE > settings_.goodThreshold) {
        report.recommendations.push_back("Consider recalibrating with better lighting conditions");
    }
    
    if (report.maxDeltaE > settings_.fairThreshold * 2.0f) {
        report.recommendations.push_back("Check for outlier patches that may be affecting accuracy");
    }
    
    if (!report.matrixIsWellConditioned) {
        if (report.matrixConditionNumber > 1000.0f) {
            report.recommendations.push_back("Matrix is poorly conditioned - consider using regularization");
        }
        if (std::abs(report.matrixDeterminant) < 1e-6f) {
            report.recommendations.push_back("Matrix is nearly singular - check for linear dependencies in data");
        }
    }
    
    if (report.r2Score < 0.8f) {
        report.recommendations.push_back("Low correlation suggests systematic errors in calibration");
    }
    
    if (report.validPatches < 20) {
        report.recommendations.push_back("Consider using more ColorChecker patches for better accuracy");
    }
    
    // Check for systematic bias
    float meanErrorMagnitude = cv::norm(report.meanError, cv::NORM_L2);
    if (meanErrorMagnitude > 0.05f) {
        report.recommendations.push_back("Systematic bias detected - check white balance and exposure");
    }
}

// Delta E calculation methods
float QualityMetrics::calculateDeltaE76(const Types::ColorValue& color1, const Types::ColorValue& color2) const {
    Types::ColorValue lab1 = colorConverter_.sRGBToLab(color1);
    Types::ColorValue lab2 = colorConverter_.sRGBToLab(color2);
    
    return cv::norm(lab1 - lab2, cv::NORM_L2);
}

float QualityMetrics::calculateDeltaE94(const Types::ColorValue& color1, const Types::ColorValue& color2) const {
    Types::ColorValue lab1 = colorConverter_.sRGBToLab(color1);
    Types::ColorValue lab2 = colorConverter_.sRGBToLab(color2);
    
    float deltaL = lab1[0] - lab2[0];
    float deltaA = lab1[1] - lab2[1];
    float deltaB = lab1[2] - lab2[2];
    
    float C1 = std::sqrt(lab1[1] * lab1[1] + lab1[2] * lab1[2]);
    float C2 = std::sqrt(lab2[1] * lab2[1] + lab2[2] * lab2[2]);
    float deltaC = C1 - C2;
    
    float deltaH = std::sqrt(std::max(0.0f, deltaA * deltaA + deltaB * deltaB - deltaC * deltaC));
    
    float SL = 1.0f;
    float SC = 1.0f + settings_.cie94_K1 * C1;
    float SH = 1.0f + settings_.cie94_K2 * C1;
    
    float dL = deltaL / (settings_.cie94_kL * SL);
    float dC = deltaC / (settings_.cie94_kC * SC);
    float dH = deltaH / (settings_.cie94_kH * SH);
    
    return std::sqrt(dL * dL + dC * dC + dH * dH);
}

float QualityMetrics::calculateDeltaE2000(const Types::ColorValue& color1, const Types::ColorValue& color2) const {
    // Simplified CIEDE2000 implementation
    // Full implementation would be more complex with additional corrections
    Types::ColorValue lab1 = colorConverter_.sRGBToLab(color1);
    Types::ColorValue lab2 = colorConverter_.sRGBToLab(color2);
    
    float L1 = lab1[0], a1 = lab1[1], b1 = lab1[2];
    float L2 = lab2[0], a2 = lab2[1], b2 = lab2[2];
    
    float deltaL = L2 - L1;
    float Lbar = (L1 + L2) / 2.0f;
    
    float C1 = std::sqrt(a1 * a1 + b1 * b1);
    float C2 = std::sqrt(a2 * a2 + b2 * b2);
    float Cbar = (C1 + C2) / 2.0f;
    
    float G = 0.5f * (1.0f - std::sqrt(std::pow(Cbar, 7) / (std::pow(Cbar, 7) + std::pow(25.0f, 7))));
    
    float ap1 = (1.0f + G) * a1;
    float ap2 = (1.0f + G) * a2;
    
    float Cp1 = std::sqrt(ap1 * ap1 + b1 * b1);
    float Cp2 = std::sqrt(ap2 * ap2 + b2 * b2);
    float Cpbar = (Cp1 + Cp2) / 2.0f;
    
    float deltaCp = Cp2 - Cp1;
    
    // Simplified calculation (full CIEDE2000 has more terms)
    float SL = 1.0f + (0.015f * (Lbar - 50.0f) * (Lbar - 50.0f)) / std::sqrt(20.0f + (Lbar - 50.0f) * (Lbar - 50.0f));
    float SC = 1.0f + 0.045f * Cpbar;
    float SH = 1.0f + 0.015f * Cpbar;
    
    float dL = deltaL / (settings_.ciede2000_kL * SL);
    float dC = deltaCp / (settings_.ciede2000_kC * SC);
    
    // Simplified hue term
    float deltaHp = 2.0f * std::sqrt(Cp1 * Cp2) * std::sin(std::atan2(b2, ap2) - std::atan2(b1, ap1));
    float dH = deltaHp / (settings_.ciede2000_kH * SH);
    
    return std::sqrt(dL * dL + dC * dC + dH * dH);
}

float QualityMetrics::calculateDeltaECMC(const Types::ColorValue& color1, const Types::ColorValue& color2) const {
    Types::ColorValue lab1 = colorConverter_.sRGBToLab(color1);
    Types::ColorValue lab2 = colorConverter_.sRGBToLab(color2);
    
    float deltaL = lab1[0] - lab2[0];
    float deltaA = lab1[1] - lab2[1];
    float deltaB = lab1[2] - lab2[2];
    
    float C1 = std::sqrt(lab1[1] * lab1[1] + lab1[2] * lab1[2]);
    float deltaC = std::sqrt(lab2[1] * lab2[1] + lab2[2] * lab2[2]) - C1;
    
    float deltaH = std::sqrt(std::max(0.0f, deltaA * deltaA + deltaB * deltaB - deltaC * deltaC));
    
    float H1 = std::atan2(lab1[2], lab1[1]) * 180.0f / CV_PI;
    if (H1 < 0) H1 += 360.0f;
    
    float F = std::sqrt(std::pow(C1, 4) / (std::pow(C1, 4) + 1900.0f));
    float T = (H1 >= 164.0f && H1 <= 345.0f) ? 
              0.56f + std::abs(0.2f * std::cos((H1 + 168.0f) * CV_PI / 180.0f)) :
              0.36f + std::abs(0.4f * std::cos((H1 + 35.0f) * CV_PI / 180.0f));
    
    float SL = (lab1[0] < 16.0f) ? 0.511f : (0.040975f * lab1[0]) / (1.0f + 0.01765f * lab1[0]);
    float SC = ((0.0638f * C1) / (1.0f + 0.0131f * C1)) + 0.638f;
    float SH = SC * (F * T + 1.0f - F);
    
    float dL = deltaL / (settings_.cmc_l * SL);
    float dC = deltaC / (settings_.cmc_c * SC);
    float dH_term = deltaH / SH;
    
    return std::sqrt(dL * dL + dC * dC + dH_term * dH_term);
}

float QualityMetrics::calculateEuclideanRGB(const Types::ColorValue& color1, const Types::ColorValue& color2) const {
    return cv::norm(color1 - color2, cv::NORM_L2);
}

float QualityMetrics::calculateEuclideanLAB(const Types::ColorValue& color1, const Types::ColorValue& color2) const {
    Types::ColorValue lab1 = colorConverter_.sRGBToLab(color1);
    Types::ColorValue lab2 = colorConverter_.sRGBToLab(color2);
    return cv::norm(lab1 - lab2, cv::NORM_L2);
}

}  // namespace ColorCorrection::Internal::Processing