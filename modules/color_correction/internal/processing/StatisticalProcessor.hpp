#pragma once

#include "../domain/ColorPatch.hpp"
#include <shared/types/Common.hpp>
#include <shared/utils/Logger.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <numeric>

namespace ColorCorrection::Internal::Processing {

class StatisticalProcessor {
  public:
    enum class OutlierMethod {
        NONE,
        Z_SCORE,           // Standard deviation based
        IQR,               // Interquartile range based
        MODIFIED_Z_SCORE,  // Median absolute deviation based
        RANSAC            // Random sample consensus
    };

    struct StatisticalSettings {
        OutlierMethod outlierMethod;
        float outlierThreshold;         // Threshold for outlier detection
        float robustMeanTrimRatio;      // Trim ratio for robust mean (0.1 = 10%)
        int ransacIterations;            // RANSAC iterations
        float ransacThreshold;          // RANSAC inlier threshold
        bool enableOutlierRemoval;      // Remove outliers completely
        int minSampleSize;                 // Minimum samples for statistics
        bool useWeightedStatistics;     // Use confidence-weighted statistics
        
        StatisticalSettings();
    };

    explicit StatisticalProcessor(const StatisticalSettings& settings = StatisticalSettings{});

    struct ColorStatistics {
        Types::ColorValue mean{0, 0, 0};
        Types::ColorValue median{0, 0, 0};
        Types::ColorValue robustMean{0, 0, 0};
        Types::ColorValue standardDeviation{0, 0, 0};
        Types::ColorValue mad{0, 0, 0};        // Median Absolute Deviation
        
        int totalSamples = 0;
        int validSamples = 0;
        int outlierSamples = 0;
        
        std::vector<int> outlierIndices;
        Types::ConfidenceScore reliability = Types::ConfidenceScore::fromValue(0.0f);
        
        bool isValid() const;
    };

    // Extract robust color statistics from a region of interest
    ColorStatistics extractRegionStatistics(const Types::Image& image, 
                                           const Types::Point2D& center, 
                                           int radius) const;

    // Process multiple color samples with outlier removal
    ColorStatistics processColorSamples(const std::vector<Types::ColorValue>& colors,
                                       const std::vector<float>& confidences = {}) const;

    // Update color patch with robust statistics
    bool updatePatchWithStatistics(Domain::ColorPatch& patch, 
                                 const Types::Image& image, 
                                 int sampleRadius = 10) const;

    // Batch process multiple patches
    std::vector<Domain::ColorPatch> processPatches(const std::vector<Domain::ColorPatch>& patches,
                                                  const Types::Image& image,
                                                  int sampleRadius = 10) const;

    // Cross-validate color measurements using multiple methods
    Types::ColorValue crossValidateColor(const std::vector<Types::ColorValue>& measurements,
                                        const std::vector<float>& confidences = {}) const;

    void setSettings(const StatisticalSettings& settings);
    StatisticalSettings getSettings() const;

  private:
    StatisticalSettings settings_;

    Types::ColorValue extractPixelColor(const Types::Image& image, int x, int y) const;

    ColorStatistics calculateColorStatistics(const std::vector<Types::ColorValue>& colors,
                                           const std::vector<float>& weights) const;

    void removeOutliers(std::vector<std::pair<Types::ColorValue, float>>& colorWeightPairs,
                       ColorStatistics& stats) const;

    void detectOutliersZScore(const std::vector<std::pair<Types::ColorValue, float>>& pairs,
                             std::vector<bool>& isOutlier) const;

    void detectOutliersModifiedZScore(const std::vector<std::pair<Types::ColorValue, float>>& pairs,
                                     std::vector<bool>& isOutlier) const;

    void detectOutliersIQR(const std::vector<std::pair<Types::ColorValue, float>>& pairs,
                          std::vector<bool>& isOutlier) const;

    void detectOutliersRANSAC(const std::vector<std::pair<Types::ColorValue, float>>& pairs,
                             std::vector<bool>& isOutlier) const;

    void calculateMean(const std::vector<std::pair<Types::ColorValue, float>>& pairs,
                      ColorStatistics& stats) const;

    void calculateMedian(const std::vector<std::pair<Types::ColorValue, float>>& pairs,
                        ColorStatistics& stats) const;

    void calculateRobustMean(const std::vector<std::pair<Types::ColorValue, float>>& pairs,
                           ColorStatistics& stats) const;

    void calculateStandardDeviation(const std::vector<std::pair<Types::ColorValue, float>>& pairs,
                                  ColorStatistics& stats) const;

    void calculateMAD(const std::vector<std::pair<Types::ColorValue, float>>& pairs,
                     ColorStatistics& stats) const;

    Types::ConfidenceScore calculateReliabilityScore(const ColorStatistics& stats) const;
};

}  // namespace ColorCorrection::Internal::Processing