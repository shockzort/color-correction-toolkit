#pragma once

#include "IDetector.hpp"
#include "../domain/ColorPatch.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

namespace ColorCorrection::Internal::Detection {

class TemplateDetector : public IDetector {
  public:
    struct TemplateSettings {
        float matchThreshold;
        float minScale;
        float maxScale;
        float scaleStep;
        int templateMethod;
        bool useMultipleTemplates;
        int maxRotationDegrees;
        int rotationStep;
        
        TemplateSettings();
    };

    explicit TemplateDetector(const TemplateSettings& settings = TemplateSettings{});

    Domain::DetectionResult detect(const Types::Image& image) override;
    Types::DetectionMethod getMethod() const override;
    Types::ConfidenceScore getExpectedConfidence(const Types::Image& image) const override;
    bool isCapable(const Types::Image& image) const override;
    std::string getName() const override;

    void setTemplateSettings(const TemplateSettings& settings);
    TemplateSettings getTemplateSettings() const;

    void setConfidenceThreshold(float threshold) override;
    float getConfidenceThreshold() const override;

    bool loadCustomTemplate(const std::string& templatePath);

  private:
    struct TemplateData {
        Types::Image image;
        std::string name;
        float aspectRatio;
        
        TemplateData();
    };

    struct MatchResult {
        cv::Point location;
        float scale = 1.0f;
        float confidence = 0.0f;
        int templateIndex = -1;
        float rotation = 0.0f;
        cv::Size templateSize;
        
        MatchResult();
    };

    TemplateSettings settings_;
    std::vector<TemplateData> templates_;

    void initializeTemplates();
    void createSyntheticTemplate();
    void createVariationTemplates();
    
    MatchResult findBestMatch(const Types::Image& image);
    std::vector<cv::Point2f> calculateCorners(const MatchResult& match);
    std::vector<cv::Point2f> generatePatchCenters(const std::vector<cv::Point2f>& corners);
    std::vector<Domain::ColorPatch> extractColorPatches(const Types::Image& image,
                                                       const std::vector<cv::Point2f>& centers);
    float calculateFinalConfidence(const MatchResult& match, 
                                  const std::vector<Domain::ColorPatch>& patches);
};

}  // namespace ColorCorrection::Internal::Detection