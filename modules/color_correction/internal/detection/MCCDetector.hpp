#pragma once

#include "IDetector.hpp"
#include "../domain/ColorPatch.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/mcc.hpp>

namespace ColorCorrection::Internal::Detection {

class MCCDetector : public IDetector {
  public:
    MCCDetector();

    Domain::DetectionResult detect(const Types::Image& image) override;
    Types::DetectionMethod getMethod() const override;
    Types::ConfidenceScore getExpectedConfidence(const Types::Image& image) const override;
    bool isCapable(const Types::Image& image) const override;
    std::string getName() const override;
    void setConfidenceThreshold(float threshold) override;
    float getConfidenceThreshold() const override;
    bool isAvailable() const;

  private:
    cv::Ptr<cv::mcc::CCheckerDetector> detector_;
    bool isAvailable_ = false;

    std::vector<Domain::ColorPatch> extractPatches(const Types::Image& image, 
                                                  cv::Ptr<cv::mcc::CChecker> checker);
    std::vector<Types::Point2D> extractCorners(cv::Ptr<cv::mcc::CChecker> checker);
    float calculateConfidence(cv::Ptr<cv::mcc::CChecker> checker, 
                            const std::vector<Domain::ColorPatch>& patches);
};

}  // namespace ColorCorrection::Internal::Detection