#pragma once

#include <shared/types/Common.hpp>
#include <string>
#include <map>
#include <memory>

namespace ColorCorrection::Interface {

class IConfiguration {
  public:
    virtual ~IConfiguration() = default;

    // Detection settings
    virtual float getDetectionConfidenceThreshold() const = 0;
    virtual void setDetectionConfidenceThreshold(float threshold) = 0;
    
    virtual int getMaxDetectionAttempts() const = 0;
    virtual void setMaxDetectionAttempts(int attempts) = 0;
    
    virtual bool isPerspectiveCorrectionEnabled() const = 0;
    virtual void setPerspectiveCorrectionEnabled(bool enabled) = 0;

    // Detection method settings
    virtual bool isMCCDetectorEnabled() const = 0;
    virtual void setMCCDetectorEnabled(bool enabled) = 0;
    
    virtual bool isContourDetectorEnabled() const = 0;
    virtual void setContourDetectorEnabled(bool enabled) = 0;
    
    virtual bool isTemplateMatchingEnabled() const = 0;
    virtual void setTemplateMatchingEnabled(bool enabled) = 0;

    // Processing settings
    virtual float getGammaCorrection() const = 0;
    virtual void setGammaCorrection(float gamma) = 0;
    
    virtual bool isOutlierFilterEnabled() const = 0;
    virtual void setOutlierFilterEnabled(bool enabled) = 0;
    
    virtual float getOutlierThreshold() const = 0;
    virtual void setOutlierThreshold(float threshold) = 0;

    // Quality metrics thresholds
    virtual float getMaxDeltaE() const = 0;
    virtual void setMaxDeltaE(float deltaE) = 0;
    
    virtual float getMaxConditionNumber() const = 0;
    virtual void setMaxConditionNumber(float conditionNumber) = 0;

    // Optimization settings
    virtual bool isGPUEnabled() const = 0;
    virtual void setGPUEnabled(bool enabled) = 0;
    
    virtual bool isLUTEnabled() const = 0;
    virtual void setLUTEnabled(bool enabled) = 0;
    
    virtual int getLUTResolution() const = 0;
    virtual void setLUTResolution(int resolution) = 0;
    
    virtual bool isParallelProcessingEnabled() const = 0;
    virtual void setParallelProcessingEnabled(bool enabled) = 0;
    
    virtual int getThreadCount() const = 0;
    virtual void setThreadCount(int count) = 0;

    // Configuration persistence
    virtual bool loadFromFile(const std::string& filename) = 0;
    virtual bool saveToFile(const std::string& filename) const = 0;
    virtual bool loadFromString(const std::string& configString) = 0;
    virtual std::string saveToString() const = 0;

    // Runtime configuration
    virtual void setParameter(const std::string& key, const std::string& value) = 0;
    virtual std::string getParameter(const std::string& key) const = 0;
    virtual std::map<std::string, std::string> getAllParameters() const = 0;
    
    virtual bool isValid() const = 0;
    virtual void reset() = 0;
    virtual void validate() = 0;
};

// Factory function for creating default configuration
std::unique_ptr<IConfiguration> createDefaultConfiguration();

}  // namespace ColorCorrection::Interface