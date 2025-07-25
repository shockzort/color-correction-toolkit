#pragma once

#include "IDetector.hpp"
#include "../domain/ColorPatch.hpp"
#include <shared/utils/Logger.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

namespace ColorCorrection::Internal::Detection {

class TemplateDetector : public IDetector {
  public:
    struct TemplateSettings {
        float matchThreshold = 0.7f;
        float minScale = 0.5f;
        float maxScale = 2.0f;
        float scaleStep = 0.1f;
        int templateMethod = cv::TM_CCOEFF_NORMED;
        bool useMultipleTemplates = true;
        int maxRotationDegrees = 15;
        int rotationStep = 5;
    };

    TemplateDetector(const TemplateSettings& settings = TemplateSettings{}) 
        : settings_(settings) {
        initializeTemplates();
        LOG_INFO("Template matching detector initialized with ", templates_.size(), " templates");
    }

    Domain::DetectionResult detect(const Types::Image& image) override {
        if (image.empty()) {
            return Domain::DetectionResult::createFailure(
                Types::DetectionMethod::TEMPLATE_MATCHING, 
                "Input image is empty");
        }

        if (templates_.empty()) {
            return Domain::DetectionResult::createFailure(
                Types::DetectionMethod::TEMPLATE_MATCHING, 
                "No templates available");
        }

        try {
            LOG_DEBUG("Starting template matching on image ", image.cols, "x", image.rows);

            // Convert to grayscale for template matching
            Types::Image grayImage;
            if (image.channels() > 1) {
                cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
            } else {
                grayImage = image.clone();
            }

            // Find best match across all templates and scales
            MatchResult bestMatch = findBestMatch(grayImage);
            
            if (bestMatch.confidence < settings_.matchThreshold) {
                return Domain::DetectionResult::createFailure(
                    Types::DetectionMethod::TEMPLATE_MATCHING, 
                    "No template match above threshold");
            }

            // Extract ColorChecker region and generate patch grid
            std::vector<cv::Point2f> corners = calculateCorners(bestMatch);
            std::vector<cv::Point2f> patchCenters = generatePatchCenters(corners);
            
            // Extract colors from patches
            std::vector<Domain::ColorPatch> patches = extractColorPatches(image, patchCenters);
            
            if (patches.size() < Types::COLORCHECKER_PATCHES * 0.6f) {
                return Domain::DetectionResult::createFailure(
                    Types::DetectionMethod::TEMPLATE_MATCHING, 
                    "Insufficient patches extracted");
            }

            // Final confidence calculation
            float finalConfidence = calculateFinalConfidence(bestMatch, patches);
            
            if (finalConfidence < confidenceThreshold_) {
                return Domain::DetectionResult::createFailure(
                    Types::DetectionMethod::TEMPLATE_MATCHING, 
                    "Final confidence too low");
            }

            LOG_INFO("Template matching successful: ", patches.size(), " patches, confidence: ", 
                     finalConfidence);

            return Domain::DetectionResult(
                Types::DetectionMethod::TEMPLATE_MATCHING,
                patches,
                Types::ConfidenceScore::fromValue(finalConfidence),
                corners
            );

        } catch (const cv::Exception& e) {
            LOG_ERROR("OpenCV exception in template matching: ", e.what());
            return Domain::DetectionResult::createFailure(
                Types::DetectionMethod::TEMPLATE_MATCHING, 
                "OpenCV exception: " + std::string(e.what()));
        }
    }

    Types::DetectionMethod getMethod() const override {
        return Types::DetectionMethod::TEMPLATE_MATCHING;
    }

    Types::ConfidenceScore getExpectedConfidence(const Types::Image& image) const override {
        if (image.empty() || templates_.empty()) {
            return Types::ConfidenceScore::fromValue(0.0f);
        }

        float score = 0.3f; // Base score for template matching (lowest priority)
        
        // Prefer images with good contrast
        cv::Scalar meanVal, stdVal;
        cv::meanStdDev(image, meanVal, stdVal);
        double contrast = stdVal[0];
        if (image.channels() >= 3) {
            contrast = (stdVal[0] + stdVal[1] + stdVal[2]) / 3.0;
        }
        
        if (contrast > 35.0) score += 0.2f;
        else if (contrast < 20.0) score -= 0.1f;
        
        // Prefer medium-sized images
        int pixels = image.cols * image.rows;
        if (pixels > 800 * 600 && pixels < 1600 * 1200) score += 0.15f;
        
        // Check for potential ColorChecker patterns with quick template match
        Types::Image grayImage;
        if (image.channels() > 1) {
            cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
        } else {
            grayImage = image.clone();
        }

        // Quick check with one template
        if (!templates_.empty()) {
            cv::Mat result;
            cv::matchTemplate(grayImage, templates_[0].image, result, cv::TM_CCOEFF_NORMED);
            
            double minVal, maxVal;
            cv::minMaxLoc(result, &minVal, &maxVal);
            
            if (maxVal > 0.5) score += 0.2f;
            else if (maxVal > 0.3) score += 0.1f;
        }
        
        return Types::ConfidenceScore::fromValue(std::clamp(score, 0.0f, 1.0f));
    }

    bool isCapable(const Types::Image& image) const override {
        return !image.empty() && !templates_.empty() && 
               image.cols >= 400 && image.rows >= 300;
    }

    std::string getName() const override {
        return "Template Matching Detector";
    }

    void setTemplateSettings(const TemplateSettings& settings) {
        settings_ = settings;
    }

    TemplateSettings getTemplateSettings() const {
        return settings_;
    }

    void setConfidenceThreshold(float threshold) override {
        confidenceThreshold_ = std::clamp(threshold, 0.0f, 1.0f);
    }

    float getConfidenceThreshold() const override {
        return confidenceThreshold_;
    }

    bool loadCustomTemplate(const std::string& templatePath) {
        try {
            Types::Image templateImage = cv::imread(templatePath, cv::IMREAD_GRAYSCALE);
            if (templateImage.empty()) {
                LOG_ERROR("Failed to load template from: ", templatePath);
                return false;
            }

            TemplateData templateData;
            templateData.image = templateImage;
            templateData.name = "custom_" + std::to_string(templates_.size());
            templateData.aspectRatio = static_cast<float>(templateImage.cols) / templateImage.rows;
            
            templates_.push_back(templateData);
            LOG_INFO("Loaded custom template: ", templatePath);
            return true;
            
        } catch (const cv::Exception& e) {
            LOG_ERROR("Error loading template: ", e.what());
            return false;
        }
    }

  private:
    struct TemplateData {
        Types::Image image;
        std::string name;
        float aspectRatio;
    };

    struct MatchResult {
        cv::Point location;
        float scale = 1.0f;
        float confidence = 0.0f;
        int templateIndex = -1;
        float rotation = 0.0f;
        cv::Size templateSize;
    };

    TemplateSettings settings_;
    std::vector<TemplateData> templates_;

    void initializeTemplates() {
        // Create synthetic ColorChecker template
        createSyntheticTemplate();
        
        // Create additional templates with different lighting conditions
        createVariationTemplates();
    }

    void createSyntheticTemplate() {
        // Create a synthetic ColorChecker Classic template (6x4 grid)
        int patchSize = 20;
        int spacing = 4;
        int width = Types::COLORCHECKER_COLS * patchSize + (Types::COLORCHECKER_COLS - 1) * spacing;
        int height = Types::COLORCHECKER_ROWS * patchSize + (Types::COLORCHECKER_ROWS - 1) * spacing;
        
        Types::Image templateImg = Types::Image::zeros(height, width, CV_8UC1);
        
        // Standard ColorChecker grayscale values (approximate)
        const std::array<uchar, Types::COLORCHECKER_PATCHES> grayValues = {{
            85, 160, 125, 105, 140, 155, 135, 95, 145, 80,
            165, 175, 75, 115, 70, 200, 130, 90, 240, 200,
            160, 120, 85, 50
        }};

        for (int row = 0; row < Types::COLORCHECKER_ROWS; ++row) {
            for (int col = 0; col < Types::COLORCHECKER_COLS; ++col) {
                int patchIdx = row * Types::COLORCHECKER_COLS + col;
                
                int x = col * (patchSize + spacing);
                int y = row * (patchSize + spacing);
                
                cv::Rect patchRect(x, y, patchSize, patchSize);
                templateImg(patchRect) = grayValues[patchIdx];
            }
        }

        // Add border
        int borderSize = 10;
        Types::Image borderedTemplate;
        cv::copyMakeBorder(templateImg, borderedTemplate, borderSize, borderSize, 
                          borderSize, borderSize, cv::BORDER_CONSTANT, cv::Scalar(128));

        TemplateData templateData;
        templateData.image = borderedTemplate;
        templateData.name = "synthetic_standard";
        templateData.aspectRatio = static_cast<float>(borderedTemplate.cols) / borderedTemplate.rows;
        
        templates_.push_back(templateData);
    }

    void createVariationTemplates() {
        if (templates_.empty()) return;

        const Types::Image& baseTemplate = templates_[0].image;
        
        // Create brightness variations
        for (int brightness = -30; brightness <= 30; brightness += 20) {
            if (brightness == 0) continue; // Skip original
            
            Types::Image brightTemplate;
            baseTemplate.convertTo(brightTemplate, -1, 1.0, brightness);
            
            TemplateData templateData;
            templateData.image = brightTemplate;
            templateData.name = "synthetic_bright_" + std::to_string(brightness);
            templateData.aspectRatio = static_cast<float>(brightTemplate.cols) / brightTemplate.rows;
            
            templates_.push_back(templateData);
        }

        // Create contrast variations
        for (float contrast = 0.8f; contrast <= 1.2f; contrast += 0.2f) {
            if (std::abs(contrast - 1.0f) < 0.01f) continue; // Skip original
            
            Types::Image contrastTemplate;
            baseTemplate.convertTo(contrastTemplate, -1, contrast, 0);
            
            TemplateData templateData;
            templateData.image = contrastTemplate;
            templateData.name = "synthetic_contrast_" + std::to_string(contrast);
            templateData.aspectRatio = static_cast<float>(contrastTemplate.cols) / contrastTemplate.rows;
            
            templates_.push_back(templateData);
        }
    }

    MatchResult findBestMatch(const Types::Image& image) {
        MatchResult bestMatch;
        bestMatch.confidence = 0.0f;

        for (size_t templateIdx = 0; templateIdx < templates_.size(); ++templateIdx) {
            const TemplateData& templateData = templates_[templateIdx];
            
            // Try different scales
            for (float scale = settings_.minScale; scale <= settings_.maxScale; scale += settings_.scaleStep) {
                Types::Image scaledTemplate;
                cv::Size newSize(
                    static_cast<int>(templateData.image.cols * scale),
                    static_cast<int>(templateData.image.rows * scale)
                );
                
                if (newSize.width > image.cols || newSize.height > image.rows) {
                    continue; // Template too large
                }
                
                cv::resize(templateData.image, scaledTemplate, newSize);
                
                // Template matching
                cv::Mat result;
                cv::matchTemplate(image, scaledTemplate, result, settings_.templateMethod);
                
                // Find best match location
                double minVal, maxVal;
                cv::Point minLoc, maxLoc;
                cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
                
                float confidence = static_cast<float>(maxVal);
                
                if (confidence > bestMatch.confidence) {
                    bestMatch.confidence = confidence;
                    bestMatch.location = maxLoc;
                    bestMatch.scale = scale;
                    bestMatch.templateIndex = static_cast<int>(templateIdx);
                    bestMatch.templateSize = scaledTemplate.size();
                }
            }
        }

        return bestMatch;
    }

    std::vector<cv::Point2f> calculateCorners(const MatchResult& match) {
        std::vector<cv::Point2f> corners;
        
        if (match.templateIndex < 0 || match.templateIndex >= static_cast<int>(templates_.size())) {
            return corners;
        }

        // Calculate corners based on template match location and size
        cv::Point2f topLeft(match.location.x, match.location.y);
        cv::Point2f topRight(match.location.x + match.templateSize.width, match.location.y);
        cv::Point2f bottomRight(match.location.x + match.templateSize.width, 
                               match.location.y + match.templateSize.height);
        cv::Point2f bottomLeft(match.location.x, match.location.y + match.templateSize.height);

        corners = {topLeft, topRight, bottomRight, bottomLeft};
        return corners;
    }

    std::vector<cv::Point2f> generatePatchCenters(const std::vector<cv::Point2f>& corners) {
        std::vector<cv::Point2f> centers;
        
        if (corners.size() != 4) {
            return centers;
        }

        centers.reserve(Types::COLORCHECKER_PATCHES);

        // Generate grid within the detected rectangle
        for (int row = 0; row < Types::COLORCHECKER_ROWS; ++row) {
            for (int col = 0; col < Types::COLORCHECKER_COLS; ++col) {
                float u = (col + 0.5f) / Types::COLORCHECKER_COLS;
                float v = (row + 0.5f) / Types::COLORCHECKER_ROWS;

                // Bilinear interpolation
                cv::Point2f top = corners[0] + u * (corners[1] - corners[0]);
                cv::Point2f bottom = corners[3] + u * (corners[2] - corners[3]);
                cv::Point2f center = top + v * (bottom - top);

                centers.push_back(center);
            }
        }

        return centers;
    }

    std::vector<Domain::ColorPatch> extractColorPatches(const Types::Image& image,
                                                       const std::vector<cv::Point2f>& centers) {
        std::vector<Domain::ColorPatch> patches;
        auto referencePatches = Domain::ColorPatch::createStandardColorChecker();

        if (centers.size() != Types::COLORCHECKER_PATCHES) {
            LOG_WARN("Expected ", Types::COLORCHECKER_PATCHES, " centers, got ", centers.size());
            return patches;
        }

        // Estimate patch size from grid spacing
        float avgSpacing = 0.0f;
        int spacingCount = 0;
        
        for (int i = 0; i < Types::COLORCHECKER_ROWS; ++i) {
            for (int j = 0; j < Types::COLORCHECKER_COLS - 1; ++j) {
                int idx1 = i * Types::COLORCHECKER_COLS + j;
                int idx2 = i * Types::COLORCHECKER_COLS + j + 1;
                avgSpacing += cv::norm(centers[idx1] - centers[idx2]);
                spacingCount++;
            }
        }
        avgSpacing /= spacingCount;
        
        int patchRadius = static_cast<int>(avgSpacing * 0.25f); // 25% of spacing

        for (int i = 0; i < Types::COLORCHECKER_PATCHES; ++i) {
            cv::Point2f center = centers[i];
            cv::Point centerInt(static_cast<int>(center.x), static_cast<int>(center.y));
            
            // Bounds checking
            if (centerInt.x - patchRadius < 0 || centerInt.x + patchRadius >= image.cols ||
                centerInt.y - patchRadius < 0 || centerInt.y + patchRadius >= image.rows) {
                continue;
            }

            // Extract region of interest
            cv::Rect roi(centerInt.x - patchRadius, centerInt.y - patchRadius,
                        2 * patchRadius + 1, 2 * patchRadius + 1);
            
            Types::Image patchRegion = image(roi);
            cv::Scalar meanColor = cv::mean(patchRegion);
            
            Types::ColorValue measuredColor;
            if (image.channels() >= 3) {
                measuredColor = Types::ColorValue(
                    meanColor[2] / 255.0f, // R
                    meanColor[1] / 255.0f, // G
                    meanColor[0] / 255.0f  // B
                );
            } else {
                float gray = meanColor[0] / 255.0f;
                measuredColor = Types::ColorValue(gray, gray, gray);
            }

            Domain::ColorPatch patch(
                i,
                center,
                measuredColor,
                referencePatches[i].getReferenceColor(),
                Types::ConfidenceScore::fromValue(0.6f) // Lower confidence for template matching
            );

            patches.push_back(patch);
        }

        return patches;
    }

    float calculateFinalConfidence(const MatchResult& match, 
                                  const std::vector<Domain::ColorPatch>& patches) {
        float confidence = match.confidence * 0.7f; // Base confidence from template match

        // Factor in patch completeness
        float completeness = static_cast<float>(patches.size()) / Types::COLORCHECKER_PATCHES;
        confidence *= completeness;

        // Factor in color consistency
        if (!patches.empty()) {
            float avgDeltaE = 0.0f;
            int validPatches = 0;
            
            for (const auto& patch : patches) {
                if (patch.isValid()) {
                    avgDeltaE += patch.calculateDeltaE();
                    validPatches++;
                }
            }
            
            if (validPatches > 0) {
                avgDeltaE /= validPatches;
                if (avgDeltaE < 20.0f) confidence += 0.1f;
                else if (avgDeltaE > 50.0f) confidence -= 0.2f;
            }
        }

        // Scale factor bonus (prefer original scale)
        if (std::abs(match.scale - 1.0f) < 0.2f) {
            confidence += 0.05f;
        }

        return std::clamp(confidence, 0.0f, 1.0f);
    }
};

}  // namespace ColorCorrection::Internal::Detection