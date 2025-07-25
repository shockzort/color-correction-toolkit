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
        bool enableMCC = true;
        bool enableContour = true;
        bool enableTemplate = true;
        
        float mccWeight = 1.0f;
        float contourWeight = 0.8f;
        float templateWeight = 0.6f;
        
        bool useAdaptiveThresholds = true;
        int maxAttempts = 3;
        bool enableFusion = true; // Combine results from multiple detectors
        
        // Timeout settings
        std::chrono::milliseconds perDetectorTimeout{5000};
        std::chrono::milliseconds totalTimeout{15000};
    };

    MultiLevelDetector(const DetectionStrategy& strategy = DetectionStrategy{}) 
        : strategy_(strategy) {
        initializeDetectors();
        LOG_INFO("Multi-level detector initialized with ", detectors_.size(), " detectors");
    }

    Domain::DetectionResult detect(const Types::Image& image) override {
        if (image.empty()) {
            return Domain::DetectionResult::createFailure(
                Types::DetectionMethod::MCC_DETECTOR, 
                "Input image is empty");
        }

        LOG_INFO("Starting multi-level detection on ", image.cols, "x", image.rows, " image");
        
        auto startTime = std::chrono::steady_clock::now();
        std::vector<DetectionAttempt> attempts;
        
        // Phase 1: Try each detector in priority order
        for (const auto& detector : detectors_) {
            if (!detector->isCapable(image)) {
                LOG_DEBUG("Detector ", detector->getName(), " not capable for this image");
                continue;
            }

            auto attemptStart = std::chrono::steady_clock::now();
            
            try {
                LOG_DEBUG("Attempting detection with ", detector->getName());
                
                // Adjust threshold based on previous attempts
                if (strategy_.useAdaptiveThresholds && !attempts.empty()) {
                    float adaptiveThreshold = calculateAdaptiveThreshold(attempts);
                    detector->setConfidenceThreshold(adaptiveThreshold);
                }

                Domain::DetectionResult result = detector->detect(image);
                
                auto attemptEnd = std::chrono::steady_clock::now();
                auto attemptDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    attemptEnd - attemptStart);

                DetectionAttempt attempt{
                    detector->getMethod(),
                    detector->getName(),
                    result,
                    attemptDuration,
                    result.isSuccess()
                };
                
                attempts.push_back(attempt);

                if (result.isSuccess()) {
                    LOG_INFO("Detection successful with ", detector->getName(), 
                            " (confidence: ", result.getOverallConfidence().value, 
                            ", time: ", attemptDuration.count(), "ms)");
                    
                    // Check if we should try fusion with other detectors
                    if (strategy_.enableFusion && shouldAttemptFusion(result, attempts)) {
                        Domain::DetectionResult fusedResult = attemptFusion(image, attempts);
                        if (fusedResult.isSuccess() && 
                            fusedResult.getOverallConfidence().value > result.getOverallConfidence().value) {
                            return fusedResult;
                        }
                    }
                    
                    return result;
                }

                LOG_DEBUG("Detection failed with ", detector->getName(), 
                         " (confidence: ", result.getOverallConfidence().value, ")");

            } catch (const std::exception& e) {
                LOG_ERROR("Exception in detector ", detector->getName(), ": ", e.what());
                
                DetectionAttempt attempt{
                    detector->getMethod(),
                    detector->getName(),
                    Domain::DetectionResult::createFailure(detector->getMethod(), e.what()),
                    std::chrono::milliseconds(0),
                    false
                };
                attempts.push_back(attempt);
            }

            // Check timeout
            auto currentTime = std::chrono::steady_clock::now();
            auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
                currentTime - startTime);
            
            if (totalDuration > strategy_.totalTimeout) {
                LOG_WARN("Multi-level detection timeout reached");
                break;
            }
        }

        // Phase 2: If no single detector succeeded, try fusion of partial results
        if (strategy_.enableFusion && !attempts.empty()) {
            Domain::DetectionResult fusedResult = attemptFusion(image, attempts);
            if (fusedResult.isSuccess()) {
                LOG_INFO("Fusion detection successful (confidence: ", 
                        fusedResult.getOverallConfidence().value, ")");
                return fusedResult;
            }
        }

        // Phase 3: Return best attempt even if failed
        if (!attempts.empty()) {
            auto bestAttempt = std::max_element(attempts.begin(), attempts.end(),
                [](const DetectionAttempt& a, const DetectionAttempt& b) {
                    return a.result.getOverallConfidence().value < 
                           b.result.getOverallConfidence().value;
                });

            LOG_WARN("All detectors failed. Best attempt: ", bestAttempt->detectorName,
                    " (confidence: ", bestAttempt->result.getOverallConfidence().value, ")");
            
            return bestAttempt->result;
        }

        LOG_ERROR("No detectors were capable of processing the image");
        return Domain::DetectionResult::createFailure(
            Types::DetectionMethod::MCC_DETECTOR, 
            "No capable detectors available");
    }

    Types::DetectionMethod getMethod() const override {
        return Types::DetectionMethod::MCC_DETECTOR; // Primary method
    }

    Types::ConfidenceScore getExpectedConfidence(const Types::Image& image) const override {
        if (image.empty()) {
            return Types::ConfidenceScore::fromValue(0.0f);
        }

        float maxConfidence = 0.0f;
        float weightedSum = 0.0f;
        float totalWeight = 0.0f;

        for (const auto& detector : detectors_) {
            if (!detector->isCapable(image)) continue;

            float confidence = detector->getExpectedConfidence(image).value;
            float weight = getDetectorWeight(detector->getMethod());
            
            maxConfidence = std::max(maxConfidence, confidence);
            weightedSum += confidence * weight;
            totalWeight += weight;
        }

        if (totalWeight > 0.0f) {
            // Combine max confidence with weighted average
            float avgConfidence = weightedSum / totalWeight;
            return Types::ConfidenceScore::fromValue((maxConfidence + avgConfidence) * 0.5f);
        }

        return Types::ConfidenceScore::fromValue(0.0f);
    }

    bool isCapable(const Types::Image& image) const override {
        for (const auto& detector : detectors_) {
            if (detector->isCapable(image)) {
                return true;
            }
        }
        return false;
    }

    std::string getName() const override {
        return "Multi-level Detector";
    }

    void setConfidenceThreshold(float threshold) override {
        confidenceThreshold_ = std::clamp(threshold, 0.0f, 1.0f);
        
        // Propagate to individual detectors
        for (auto& detector : detectors_) {
            detector->setConfidenceThreshold(threshold);
        }
    }

    float getConfidenceThreshold() const override {
        return confidenceThreshold_;
    }

    void setDetectionStrategy(const DetectionStrategy& strategy) {
        strategy_ = strategy;
        initializeDetectors(); // Reinitialize based on new strategy
    }

    DetectionStrategy getDetectionStrategy() const {
        return strategy_;
    }

    std::vector<std::string> getAvailableDetectors() const {
        std::vector<std::string> names;
        for (const auto& detector : detectors_) {
            names.push_back(detector->getName());
        }
        return names;
    }

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

    void initializeDetectors() {
        detectors_.clear();

        // Add detectors in priority order
        if (strategy_.enableMCC) {
            auto mccDetector = std::make_unique<MCCDetector>();
            if (mccDetector->isAvailable()) {
                detectors_.push_back(std::move(mccDetector));
            } else {
                LOG_WARN("MCC detector not available - OpenCV MCC module missing");
            }
        }

        if (strategy_.enableContour) {
            detectors_.push_back(std::make_unique<ContourDetector>());
        }

        if (strategy_.enableTemplate) {
            detectors_.push_back(std::make_unique<TemplateDetector>());
        }

        LOG_INFO("Initialized ", detectors_.size(), " detectors");
    }

    float getDetectorWeight(Types::DetectionMethod method) const {
        switch (method) {
            case Types::DetectionMethod::MCC_DETECTOR:
                return strategy_.mccWeight;
            case Types::DetectionMethod::CONTOUR_BASED:
                return strategy_.contourWeight;
            case Types::DetectionMethod::TEMPLATE_MATCHING:
                return strategy_.templateWeight;
            default:
                return 1.0f;
        }
    }

    float calculateAdaptiveThreshold(const std::vector<DetectionAttempt>& attempts) const {
        if (attempts.empty()) {
            return confidenceThreshold_;
        }

        // Find the best confidence so far
        float bestConfidence = 0.0f;
        for (const auto& attempt : attempts) {
            bestConfidence = std::max(bestConfidence, attempt.result.getOverallConfidence().value);
        }

        // Lower the threshold if we haven't found anything good
        if (bestConfidence < 0.5f) {
            return std::max(0.2f, confidenceThreshold_ - 0.2f);
        } else if (bestConfidence < 0.7f) {
            return std::max(0.4f, confidenceThreshold_ - 0.1f);
        }

        return confidenceThreshold_;
    }

    bool shouldAttemptFusion(const Domain::DetectionResult& result, 
                           const std::vector<DetectionAttempt>& attempts) const {
        // Only attempt fusion if:
        // 1. Current result has moderate confidence (could be improved)
        // 2. We have other partial results to work with
        // 3. We haven't exceeded time limits
        
        return strategy_.enableFusion && 
               result.getOverallConfidence().value < 0.9f &&
               attempts.size() > 1;
    }

    Domain::DetectionResult attemptFusion(const Types::Image& image,
                                        const std::vector<DetectionAttempt>& attempts) {
        LOG_DEBUG("Attempting result fusion from ", attempts.size(), " detection attempts");

        // Collect all partial results
        std::vector<Domain::DetectionResult> validResults;
        for (const auto& attempt : attempts) {
            if (attempt.result.getPatches().size() > Types::COLORCHECKER_PATCHES * 0.3f) {
                validResults.push_back(attempt.result);
            }
        }

        if (validResults.empty()) {
            return Domain::DetectionResult::createFailure(
                Types::DetectionMethod::MCC_DETECTOR, 
                "No results suitable for fusion");
        }

        // Simple fusion strategy: use the result with the most patches
        // In a more sophisticated implementation, we could:
        // - Combine patch detections from multiple sources
        // - Use consensus voting for patch positions
        // - Weight results based on detector reliability

        auto bestResult = std::max_element(validResults.begin(), validResults.end(),
            [](const Domain::DetectionResult& a, const Domain::DetectionResult& b) {
                float scoreA = a.getPatches().size() * a.getOverallConfidence().value;
                float scoreB = b.getPatches().size() * b.getOverallConfidence().value;
                return scoreA < scoreB;
            });

        if (bestResult != validResults.end()) {
            // Create a new result with enhanced confidence (fusion bonus)
            float originalConfidence = bestResult->getOverallConfidence().value;
            float fusionBonus = 0.1f * (validResults.size() - 1); // Bonus for having multiple confirmations
            float enhancedConfidence = std::min(1.0f, originalConfidence + fusionBonus);

            return Domain::DetectionResult(
                bestResult->getMethod(),
                bestResult->getPatches(),
                Types::ConfidenceScore::fromValue(enhancedConfidence),
                bestResult->getCorners()
            );
        }

        return Domain::DetectionResult::createFailure(
            Types::DetectionMethod::MCC_DETECTOR, 
            "Fusion failed to produce valid result");
    }
};

}  // namespace ColorCorrection::Internal::Detection