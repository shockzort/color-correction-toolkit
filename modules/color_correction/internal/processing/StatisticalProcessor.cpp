#include "StatisticalProcessor.hpp"

namespace ColorCorrection::Internal::Processing {

// StatisticalSettings constructor
StatisticalProcessor::StatisticalSettings::StatisticalSettings()
    : outlierMethod(OutlierMethod::MODIFIED_Z_SCORE)
    , outlierThreshold(2.5f)
    , robustMeanTrimRatio(0.1f)
    , ransacIterations(100)
    , ransacThreshold(0.1f)
    , enableOutlierRemoval(true)
    , minSampleSize(5)
    , useWeightedStatistics(true) {
}

// StatisticalProcessor constructor
StatisticalProcessor::StatisticalProcessor(const StatisticalSettings& settings)
    : settings_(settings) {
    LOG_INFO("Statistical processor initialized");
}

// ColorStatistics::isValid implementation
bool StatisticalProcessor::ColorStatistics::isValid() const {
    return validSamples >= 3 && reliability.value > 0.3f;
}

// Extract robust color statistics from a region of interest
StatisticalProcessor::ColorStatistics StatisticalProcessor::extractRegionStatistics(
    const Types::Image& image, 
    const Types::Point2D& center, 
    int radius) const {
    
    ColorStatistics stats;
    
    if (image.empty() || radius <= 0) {
        LOG_ERROR("Invalid input for region statistics");
        return stats;
    }

    try {
        // Extract circular region
        cv::Point centerInt(static_cast<int>(center.x), static_cast<int>(center.y));
        
        // Bounds checking
        if (centerInt.x - radius < 0 || centerInt.x + radius >= image.cols ||
            centerInt.y - radius < 0 || centerInt.y + radius >= image.rows) {
            LOG_WARN("Region extends beyond image boundaries");
            return stats;
        }

        std::vector<Types::ColorValue> colors;
        std::vector<float> weights;
        
        // Create circular mask and extract colors
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                float distance = std::sqrt(dx * dx + dy * dy);
                if (distance <= radius) {
                    int x = centerInt.x + dx;
                    int y = centerInt.y + dy;
                    
                    if (x >= 0 && x < image.cols && y >= 0 && y < image.rows) {
                        Types::ColorValue color = extractPixelColor(image, x, y);
                        colors.push_back(color);
                        
                        // Weight by distance from center (closer = higher weight)
                        float weight = 1.0f - (distance / radius);
                        weights.push_back(weight);
                    }
                }
            }
        }
        
        if (colors.empty()) {
            LOG_ERROR("No valid pixels in region");
            return stats;
        }

        stats = calculateColorStatistics(colors, weights);
        
        LOG_DEBUG("Extracted region statistics: ", stats.validSamples, " valid pixels");
        
    } catch (const cv::Exception& e) {
        LOG_ERROR("Error extracting region statistics: ", e.what());
    }
    
    return stats;
}

// Process multiple color samples with outlier removal
StatisticalProcessor::ColorStatistics StatisticalProcessor::processColorSamples(
    const std::vector<Types::ColorValue>& colors,
    const std::vector<float>& confidences) const {
    
    ColorStatistics stats;
    
    if (colors.empty()) {
        LOG_ERROR("No color samples provided");
        return stats;
    }

    try {
        std::vector<float> weights;
        if (confidences.size() == colors.size() && settings_.useWeightedStatistics) {
            weights = confidences;
        } else {
            weights.assign(colors.size(), 1.0f); // Equal weights
        }

        stats = calculateColorStatistics(colors, weights);
        
        LOG_DEBUG("Processed ", colors.size(), " color samples, ", 
                 stats.validSamples, " valid after outlier removal");
        
    } catch (const std::exception& e) {
        LOG_ERROR("Error processing color samples: ", e.what());
    }
    
    return stats;
}

// Update color patch with robust statistics
bool StatisticalProcessor::updatePatchWithStatistics(
    Domain::ColorPatch& patch, 
    const Types::Image& image, 
    int sampleRadius) const {
    
    ColorStatistics stats = extractRegionStatistics(image, patch.getCenter(), sampleRadius);
    
    if (!stats.isValid()) {
        LOG_WARN("Failed to extract valid statistics for patch ", patch.getPatchId());
        return false;
    }

    // Update patch with robust mean color
    patch.setMeasuredColor(stats.robustMean);
    
    // Update confidence based on statistical reliability
    float currentConfidence = patch.getConfidence().value;
    float statisticalConfidence = stats.reliability.value;
    float combinedConfidence = (currentConfidence + statisticalConfidence) * 0.5f;
    
    patch.setConfidence(Types::ConfidenceScore::fromValue(combinedConfidence));
    
    return true;
}

// Batch process multiple patches
std::vector<Domain::ColorPatch> StatisticalProcessor::processPatches(
    const std::vector<Domain::ColorPatch>& patches,
    const Types::Image& image,
    int sampleRadius) const {
    
    std::vector<Domain::ColorPatch> processedPatches = patches;
    
    int successCount = 0;
    for (auto& patch : processedPatches) {
        if (updatePatchWithStatistics(patch, image, sampleRadius)) {
            successCount++;
        }
    }
    
    LOG_INFO("Successfully processed ", successCount, "/", patches.size(), " patches");
    
    return processedPatches;
}

// Cross-validate color measurements using multiple methods
Types::ColorValue StatisticalProcessor::crossValidateColor(
    const std::vector<Types::ColorValue>& measurements,
    const std::vector<float>& confidences) const {
    
    if (measurements.empty()) {
        return Types::ColorValue(0, 0, 0);
    }

    if (measurements.size() == 1) {
        return measurements[0];
    }

    try {
        ColorStatistics stats = processColorSamples(measurements, confidences);
        
        if (stats.isValid()) {
            // Use robust mean for best estimate
            return stats.robustMean;
        } else {
            // Fallback to simple mean
            Types::ColorValue mean(0, 0, 0);
            for (const auto& color : measurements) {
                mean += color;
            }
            return mean / static_cast<float>(measurements.size());
        }
        
    } catch (const std::exception& e) {
        LOG_ERROR("Error in cross-validation: ", e.what());
        return measurements[0]; // Return first measurement as fallback
    }
}

void StatisticalProcessor::setSettings(const StatisticalSettings& settings) {
    settings_ = settings;
}

StatisticalProcessor::StatisticalSettings StatisticalProcessor::getSettings() const {
    return settings_;
}

// Private methods

Types::ColorValue StatisticalProcessor::extractPixelColor(const Types::Image& image, int x, int y) const {
    Types::ColorValue color;
    
    if (image.channels() >= 3) {
        cv::Vec3b bgr = image.at<cv::Vec3b>(y, x);
        color = Types::ColorValue(bgr[2] / 255.0f, bgr[1] / 255.0f, bgr[0] / 255.0f); // BGR to RGB
    } else {
        uchar gray = image.at<uchar>(y, x);
        float grayFloat = gray / 255.0f;
        color = Types::ColorValue(grayFloat, grayFloat, grayFloat);
    }
    
    return color;
}

StatisticalProcessor::ColorStatistics StatisticalProcessor::calculateColorStatistics(
    const std::vector<Types::ColorValue>& colors,
    const std::vector<float>& weights) const {
    
    ColorStatistics stats;
    stats.totalSamples = static_cast<int>(colors.size());
    
    if (colors.empty()) {
        return stats;
    }

    // Create working copy with weights
    std::vector<std::pair<Types::ColorValue, float>> colorWeightPairs;
    for (size_t i = 0; i < colors.size(); ++i) {
        float weight = (i < weights.size()) ? weights[i] : 1.0f;
        colorWeightPairs.emplace_back(colors[i], weight);
    }

    // Remove outliers if enabled
    if (settings_.enableOutlierRemoval && colors.size() > settings_.minSampleSize) {
        removeOutliers(colorWeightPairs, stats);
    }

    stats.validSamples = static_cast<int>(colorWeightPairs.size());
    
    if (colorWeightPairs.empty()) {
        LOG_WARN("All samples removed as outliers");
        return stats;
    }

    // Calculate basic statistics
    calculateMean(colorWeightPairs, stats);
    calculateMedian(colorWeightPairs, stats);
    calculateRobustMean(colorWeightPairs, stats);
    calculateStandardDeviation(colorWeightPairs, stats);
    calculateMAD(colorWeightPairs, stats);
    
    // Calculate reliability score
    stats.reliability = calculateReliabilityScore(stats);
    
    return stats;
}

void StatisticalProcessor::removeOutliers(
    std::vector<std::pair<Types::ColorValue, float>>& colorWeightPairs,
    ColorStatistics& stats) const {
    
    if (colorWeightPairs.size() < settings_.minSampleSize) {
        return;
    }

    std::vector<bool> isOutlier(colorWeightPairs.size(), false);
    
    switch (settings_.outlierMethod) {
        case OutlierMethod::Z_SCORE:
            detectOutliersZScore(colorWeightPairs, isOutlier);
            break;
        case OutlierMethod::IQR:
            detectOutliersIQR(colorWeightPairs, isOutlier);
            break;
        case OutlierMethod::MODIFIED_Z_SCORE:
            detectOutliersModifiedZScore(colorWeightPairs, isOutlier);
            break;
        case OutlierMethod::RANSAC:
            detectOutliersRANSAC(colorWeightPairs, isOutlier);
            break;
        case OutlierMethod::NONE:
        default:
            return; // No outlier removal
    }

    // Remove outliers and track indices
    std::vector<std::pair<Types::ColorValue, float>> filteredPairs;
    for (size_t i = 0; i < colorWeightPairs.size(); ++i) {
        if (isOutlier[i]) {
            stats.outlierIndices.push_back(static_cast<int>(i));
        } else {
            filteredPairs.push_back(colorWeightPairs[i]);
        }
    }

    stats.outlierSamples = static_cast<int>(stats.outlierIndices.size());
    colorWeightPairs = filteredPairs;
    
    LOG_DEBUG("Removed ", stats.outlierSamples, " outliers using ", 
             static_cast<int>(settings_.outlierMethod), " method");
}

void StatisticalProcessor::detectOutliersZScore(
    const std::vector<std::pair<Types::ColorValue, float>>& pairs,
    std::vector<bool>& isOutlier) const {
    
    if (pairs.size() < 3) return;

    // Calculate mean and standard deviation for each channel
    Types::ColorValue mean(0, 0, 0);
    float totalWeight = 0.0f;
    
    for (const auto& pair : pairs) {
        mean += pair.first * pair.second;
        totalWeight += pair.second;
    }
    mean /= totalWeight;

    Types::ColorValue variance(0, 0, 0);
    for (const auto& pair : pairs) {
        Types::ColorValue diff = pair.first - mean;
        variance += Types::ColorValue(diff[0] * diff[0], diff[1] * diff[1], diff[2] * diff[2]) * pair.second;
    }
    variance /= totalWeight;

    Types::ColorValue stddev(std::sqrt(variance[0]), std::sqrt(variance[1]), std::sqrt(variance[2]));

    // Mark outliers
    for (size_t i = 0; i < pairs.size(); ++i) {
        Types::ColorValue diff = pairs[i].first - mean;
        float maxZScore = 0.0f;
        
        for (int c = 0; c < 3; ++c) {
            if (stddev[c] > 1e-6f) {
                float zScore = std::abs(diff[c] / stddev[c]);
                maxZScore = std::max(maxZScore, zScore);
            }
        }
        
        if (maxZScore > settings_.outlierThreshold) {
            isOutlier[i] = true;
        }
    }
}

void StatisticalProcessor::detectOutliersModifiedZScore(
    const std::vector<std::pair<Types::ColorValue, float>>& pairs,
    std::vector<bool>& isOutlier) const {
    
    if (pairs.size() < 3) return;

    // Calculate median and MAD for each channel
    for (int channel = 0; channel < 3; ++channel) {
        std::vector<float> channelValues;
        for (const auto& pair : pairs) {
            channelValues.push_back(pair.first[channel]);
        }
        
        std::sort(channelValues.begin(), channelValues.end());
        float median = channelValues[channelValues.size() / 2];
        
        // Calculate MAD
        std::vector<float> deviations;
        for (float value : channelValues) {
            deviations.push_back(std::abs(value - median));
        }
        std::sort(deviations.begin(), deviations.end());
        float mad = deviations[deviations.size() / 2];
        
        if (mad < 1e-6f) mad = 1e-6f; // Avoid division by zero
        
        // Mark outliers for this channel
        for (size_t i = 0; i < pairs.size(); ++i) {
            float modifiedZScore = 0.6745f * std::abs(pairs[i].first[channel] - median) / mad;
            if (modifiedZScore > settings_.outlierThreshold) {
                isOutlier[i] = true;
            }
        }
    }
}

void StatisticalProcessor::detectOutliersIQR(
    const std::vector<std::pair<Types::ColorValue, float>>& pairs,
    std::vector<bool>& isOutlier) const {
    
    if (pairs.size() < 4) return;

    for (int channel = 0; channel < 3; ++channel) {
        std::vector<float> channelValues;
        for (const auto& pair : pairs) {
            channelValues.push_back(pair.first[channel]);
        }
        
        std::sort(channelValues.begin(), channelValues.end());
        
        size_t n = channelValues.size();
        float q1 = channelValues[n / 4];
        float q3 = channelValues[3 * n / 4];
        float iqr = q3 - q1;
        
        float lowerBound = q1 - 1.5f * iqr;
        float upperBound = q3 + 1.5f * iqr;
        
        for (size_t i = 0; i < pairs.size(); ++i) {
            float value = pairs[i].first[channel];
            if (value < lowerBound || value > upperBound) {
                isOutlier[i] = true;
            }
        }
    }
}

void StatisticalProcessor::detectOutliersRANSAC(
    const std::vector<std::pair<Types::ColorValue, float>>& pairs,
    std::vector<bool>& isOutlier) const {
    
    // Simplified RANSAC for color outlier detection
    // This is a basic implementation - could be enhanced
    
    if (pairs.size() < settings_.minSampleSize) return;

    int bestInlierCount = 0;
    Types::ColorValue bestModel(0, 0, 0);

    for (int iter = 0; iter < settings_.ransacIterations; ++iter) {
        // Randomly sample a subset
        std::vector<int> sampleIndices;
        for (int i = 0; i < settings_.minSampleSize; ++i) {
            int idx = rand() % pairs.size();
            sampleIndices.push_back(idx);
        }

        // Calculate model (mean of sample)
        Types::ColorValue model(0, 0, 0);
        float totalWeight = 0.0f;
        for (int idx : sampleIndices) {
            model += pairs[idx].first * pairs[idx].second;
            totalWeight += pairs[idx].second;
        }
        model /= totalWeight;

        // Count inliers
        int inlierCount = 0;
        for (const auto& pair : pairs) {
            float distance = cv::norm(pair.first - model);
            if (distance <= settings_.ransacThreshold) {
                inlierCount++;
            }
        }

        if (inlierCount > bestInlierCount) {
            bestInlierCount = inlierCount;
            bestModel = model;
        }
    }

    // Mark outliers based on best model
    for (size_t i = 0; i < pairs.size(); ++i) {
        float distance = cv::norm(pairs[i].first - bestModel);
        if (distance > settings_.ransacThreshold) {
            isOutlier[i] = true;
        }
    }
}

void StatisticalProcessor::calculateMean(
    const std::vector<std::pair<Types::ColorValue, float>>& pairs,
    ColorStatistics& stats) const {
    
    Types::ColorValue weightedSum(0, 0, 0);
    float totalWeight = 0.0f;
    
    for (const auto& pair : pairs) {
        weightedSum += pair.first * pair.second;
        totalWeight += pair.second;
    }
    
    if (totalWeight > 0.0f) {
        stats.mean = weightedSum / totalWeight;
    }
}

void StatisticalProcessor::calculateMedian(
    const std::vector<std::pair<Types::ColorValue, float>>& pairs,
    ColorStatistics& stats) const {
    
    if (pairs.empty()) return;

    for (int channel = 0; channel < 3; ++channel) {
        std::vector<float> channelValues;
        for (const auto& pair : pairs) {
            channelValues.push_back(pair.first[channel]);
        }
        
        std::sort(channelValues.begin(), channelValues.end());
        stats.median[channel] = channelValues[channelValues.size() / 2];
    }
}

void StatisticalProcessor::calculateRobustMean(
    const std::vector<std::pair<Types::ColorValue, float>>& pairs,
    ColorStatistics& stats) const {
    
    if (pairs.empty()) return;

    // Trimmed mean: remove extreme values
    int trimCount = static_cast<int>(pairs.size() * settings_.robustMeanTrimRatio);
    
    for (int channel = 0; channel < 3; ++channel) {
        std::vector<std::pair<float, float>> channelValueWeights;
        for (const auto& pair : pairs) {
            channelValueWeights.emplace_back(pair.first[channel], pair.second);
        }
        
        // Sort by value
        std::sort(channelValueWeights.begin(), channelValueWeights.end());
        
        // Calculate trimmed mean
        float weightedSum = 0.0f;
        float totalWeight = 0.0f;
        
        int start = trimCount;
        int end = static_cast<int>(channelValueWeights.size()) - trimCount;
        
        for (int i = start; i < end; ++i) {
            weightedSum += channelValueWeights[i].first * channelValueWeights[i].second;
            totalWeight += channelValueWeights[i].second;
        }
        
        if (totalWeight > 0.0f) {
            stats.robustMean[channel] = weightedSum / totalWeight;
        } else {
            stats.robustMean[channel] = stats.mean[channel]; // Fallback
        }
    }
}

void StatisticalProcessor::calculateStandardDeviation(
    const std::vector<std::pair<Types::ColorValue, float>>& pairs,
    ColorStatistics& stats) const {
    
    if (pairs.size() < 2) return;

    Types::ColorValue variance(0, 0, 0);
    float totalWeight = 0.0f;
    
    for (const auto& pair : pairs) {
        Types::ColorValue diff = pair.first - stats.mean;
        variance += Types::ColorValue(diff[0] * diff[0], diff[1] * diff[1], diff[2] * diff[2]) * pair.second;
        totalWeight += pair.second;
    }
    
    if (totalWeight > 0.0f) {
        variance /= totalWeight;
        stats.standardDeviation = Types::ColorValue(
            std::sqrt(variance[0]), 
            std::sqrt(variance[1]), 
            std::sqrt(variance[2])
        );
    }
}

void StatisticalProcessor::calculateMAD(
    const std::vector<std::pair<Types::ColorValue, float>>& pairs,
    ColorStatistics& stats) const {
    
    if (pairs.empty()) return;

    for (int channel = 0; channel < 3; ++channel) {
        std::vector<float> deviations;
        for (const auto& pair : pairs) {
            deviations.push_back(std::abs(pair.first[channel] - stats.median[channel]));
        }
        
        std::sort(deviations.begin(), deviations.end());
        stats.mad[channel] = deviations[deviations.size() / 2];
    }
}

Types::ConfidenceScore StatisticalProcessor::calculateReliabilityScore(const ColorStatistics& stats) const {
    float score = 1.0f;
    
    // Penalize low sample count
    if (stats.validSamples < 10) {
        score *= static_cast<float>(stats.validSamples) / 10.0f;
    }
    
    // Penalize high outlier ratio
    if (stats.totalSamples > 0) {
        float outlierRatio = static_cast<float>(stats.outlierSamples) / stats.totalSamples;
        if (outlierRatio > 0.2f) { // More than 20% outliers
            score *= std::max(0.2f, 1.0f - outlierRatio);
        }
    }
    
    // Penalize high variance
    float avgStdDev = (stats.standardDeviation[0] + stats.standardDeviation[1] + stats.standardDeviation[2]) / 3.0f;
    if (avgStdDev > 0.1f) { // High variance
        score *= std::max(0.3f, 1.0f - avgStdDev * 2.0f);
    }
    
    return Types::ConfidenceScore::fromValue(std::clamp(score, 0.0f, 1.0f));
}

}  // namespace ColorCorrection::Internal::Processing