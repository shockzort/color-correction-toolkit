#pragma once

#include "../domain/DetectionResult.hpp"
#include <shared/types/Common.hpp>
#include <memory>

namespace ColorCorrection::Internal::Detection {

class IDetector {
  public:
    virtual ~IDetector() = default;

    virtual Domain::DetectionResult detect(const Types::Image& image) = 0;
    
    virtual Types::DetectionMethod getMethod() const = 0;
    
    virtual Types::ConfidenceScore getExpectedConfidence(const Types::Image& image) const = 0;
    
    virtual bool isCapable(const Types::Image& image) const = 0;
    
    virtual std::string getName() const = 0;

    virtual void setConfidenceThreshold(float threshold) = 0;
    virtual float getConfidenceThreshold() const = 0;

  protected:
    float confidenceThreshold_ = 0.8f;
};

}  // namespace ColorCorrection::Internal::Detection