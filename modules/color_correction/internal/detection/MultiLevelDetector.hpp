#pragma once

#include "IDetector.hpp"
#include "MCCDetector.hpp"
#include "ContourDetector.hpp"
#include "TemplateDetector.hpp"
#include <shared/utils/Logger.hpp>
#include <memory>
#include <vector>
#include <chrono>

namespace ColorCorrection::Internal::Detection {

class MultiLevelDetector : public IDetector {
  public:
    struct DetectionStrategy {
        bool enableMCC;
        bool enableContour;
        bool enableTemplate;
        
        float mccWeight;
        float contourWeight;
        float templateWeight;
        
        bool useAdaptiveThresholds;
        int maxAttempts;
        bool enableFusion; // Combine results from multiple detectors
        
        // Timeout settings
        std::chrono::milliseconds perDetectorTimeout;
        std::chrono::milliseconds totalTimeout;
        
        DetectionStrategy()
            : enableMCC(true)
            , enableContour(true)
            , enableTemplate(true)
            , mccWeight(1.0f)
            , contourWeight(0.8f)
            , templateWeight(0.6f)
            , useAdaptiveThresholds(true)
            , maxAttempts(3)
            , enableFusion(true)
            , perDetectorTimeout{5000}
            , totalTimeout{15000} {}
    };

    MultiLevelDetector(const DetectionStrategy& strategy = DetectionStrategy{});

    Domain::DetectionResult detect(const Types::Image& image) override;
    Types::DetectionMethod getMethod() const override;
    Types::ConfidenceScore getExpectedConfidence(const Types::Image& image) const override;
    bool isCapable(const Types::Image& image) const override;
    std::string getName() const override;
    void setConfidenceThreshold(float threshold) override;
    float getConfidenceThreshold() const override;

    void setDetectionStrategy(const DetectionStrategy& strategy);
    DetectionStrategy getDetectionStrategy() const;
    std::vector<std::string> getAvailableDetectors() const;

  private:
    struct DetectionAttempt {
        Types::DetectionMethod method;
        std::string detectorName;
        Domain::DetectionResult result;
        std::chrono::milliseconds duration;
        bool succeeded;
    };

    DetectionStrategy strategy_;
    std::vector<std::unique_ptr<IDetector>> detectors_;

    void initializeDetectors();
    float getDetectorWeight(Types::DetectionMethod method) const;
    float calculateAdaptiveThreshold(const std::vector<DetectionAttempt>& attempts) const;
    bool shouldAttemptFusion(const Domain::DetectionResult& result, 
                           const std::vector<DetectionAttempt>& attempts) const;
    Domain::DetectionResult attemptFusion(const Types::Image& image,
                                        const std::vector<DetectionAttempt>& attempts);
};

}  // namespace ColorCorrection::Internal::Detection