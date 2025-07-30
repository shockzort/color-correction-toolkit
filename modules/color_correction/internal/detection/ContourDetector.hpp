#pragma once

#include "IDetector.hpp"
#include "../domain/ColorPatch.hpp"
#include <shared/utils/Logger.hpp>
#include <opencv2/opencv.hpp>
#include <algorithm>

namespace ColorCorrection::Internal::Detection {

class ContourDetector : public IDetector {
  public:
    struct ContourSettings {
        double minContourArea;
        double maxContourArea;
        double approxEpsilonFactor;
        double minAspectRatio;
        double maxAspectRatio;
        int gaussianBlurKernel;
        int morphologyKernel;
        double cannyLowerThreshold;
        double cannyUpperThreshold;
        
        ContourSettings();
    };

    ContourDetector(const ContourSettings& settings = ContourSettings{});

    Domain::DetectionResult detect(const Types::Image& image) override;
    Types::DetectionMethod getMethod() const override;
    Types::ConfidenceScore getExpectedConfidence(const Types::Image& image) const override;
    bool isCapable(const Types::Image& image) const override;
    std::string getName() const override;

    void setContourSettings(const ContourSettings& settings);
    ContourSettings getContourSettings() const;
    void setConfidenceThreshold(float threshold) override;
    float getConfidenceThreshold() const override;

  private:
    ContourSettings settings_;

    Types::Image preprocessImage(const Types::Image& image);
    bool findColorCheckerContour(const std::vector<std::vector<cv::Point>>& contours,
                                const Types::Image& originalImage,
                                std::vector<cv::Point2f>& corners,
                                std::vector<std::vector<cv::Point2f>>& patchCenters);
    double calculateContourScore(const std::vector<cv::Point>& contour,
                               const std::vector<cv::Point>& approx,
                               const Types::Image& image);
    void orderCorners(std::vector<cv::Point2f>& corners);
    bool generatePatchGrid(const std::vector<cv::Point2f>& corners, 
                          std::vector<cv::Point2f>& gridCenters);
    std::vector<Domain::ColorPatch> extractColorPatches(const Types::Image& image,
                                                       const std::vector<cv::Point2f>& centers);
    float calculateConfidence(const std::vector<cv::Point2f>& corners,
                            const std::vector<Domain::ColorPatch>& patches,
                            const Types::Image& image);
};

}  // namespace ColorCorrection::Internal::Detection