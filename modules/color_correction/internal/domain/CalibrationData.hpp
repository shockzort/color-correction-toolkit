#pragma once

#include "CorrectionMatrix.hpp"
#include "DetectionResult.hpp"
#include <shared/types/Common.hpp>
#include <shared/utils/Logger.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include <map>

namespace ColorCorrection::Domain {

struct QualityMetrics {
    float averageDeltaE = 0.0f;
    float maxDeltaE = 0.0f;
    float rmse = 0.0f;
    float matrixConditionNumber = 0.0f;
    float matrixDeterminant = 0.0f;
    Types::ConfidenceScore overallQuality = Types::ConfidenceScore::fromValue(0.0f);

    bool isAcceptable(float maxDeltaEThreshold = 2.0f, 
                     float maxConditionNumber = 1000.0f) const {
        return averageDeltaE <= maxDeltaEThreshold && 
               maxDeltaE <= maxDeltaEThreshold * 2.0f &&
               matrixConditionNumber <= maxConditionNumber &&
               std::abs(matrixDeterminant) > 1e-6f;
    }
};

class CalibrationData {
  public:
    CalibrationData() : timestamp_(std::chrono::system_clock::now()), isValid_(false) {}

    CalibrationData(const DetectionResult& detectionResult, 
                   const CorrectionMatrix& correctionMatrix)
        : detectionResult_(detectionResult), correctionMatrix_(correctionMatrix),
          timestamp_(std::chrono::system_clock::now()) {
        calculateQualityMetrics();
        validate();
    }

    const DetectionResult& getDetectionResult() const { return detectionResult_; }
    const CorrectionMatrix& getCorrectionMatrix() const { return correctionMatrix_; }
    const QualityMetrics& getQualityMetrics() const { return qualityMetrics_; }
    std::chrono::system_clock::time_point getTimestamp() const { return timestamp_; }
    const std::map<std::string, std::string>& getMetadata() const { return metadata_; }
    
    bool isValid() const { return isValid_; }

    void setDetectionResult(const DetectionResult& result) {
        detectionResult_ = result;
        calculateQualityMetrics();
        validate();
    }

    void setCorrectionMatrix(const CorrectionMatrix& matrix) {
        correctionMatrix_ = matrix;
        calculateQualityMetrics();
        validate();
    }

    void addMetadata(const std::string& key, const std::string& value) {
        metadata_[key] = value;
    }

    void setCalibrationImage(const Types::Image& image) {
        if (!image.empty()) {
            addMetadata("image_width", std::to_string(image.cols));
            addMetadata("image_height", std::to_string(image.rows));
            addMetadata("image_channels", std::to_string(image.channels()));
            addMetadata("image_depth", std::to_string(image.depth()));
        }
    }

    void setLightingConditions(const std::string& conditions) {
        addMetadata("lighting_conditions", conditions);
    }

    void setCameraSettings(const std::string& settings) {
        addMetadata("camera_settings", settings);
    }

    bool saveToFile(const std::string& filename) const {
        try {
            cv::FileStorage fs(filename, cv::FileStorage::WRITE);
            if (!fs.isOpened()) {
                LOG_ERROR("Cannot open calibration file for writing: ", filename);
                return false;
            }

            // Save correction matrix
            cv::Mat matrixMat(correctionMatrix_.getMatrix());
            fs << "correction_matrix" << matrixMat;
            
            // Save quality metrics
            fs << "quality_metrics" << "{";
            fs << "average_delta_e" << qualityMetrics_.averageDeltaE;
            fs << "max_delta_e" << qualityMetrics_.maxDeltaE;
            fs << "rmse" << qualityMetrics_.rmse;
            fs << "matrix_condition_number" << qualityMetrics_.matrixConditionNumber;
            fs << "matrix_determinant" << qualityMetrics_.matrixDeterminant;
            fs << "overall_quality" << qualityMetrics_.overallQuality.value;
            fs << "}";

            // Save detection information
            fs << "detection_info" << "{";
            fs << "method" << static_cast<int>(detectionResult_.getMethod());
            fs << "method_name" << detectionResult_.getMethodName();
            fs << "overall_confidence" << detectionResult_.getOverallConfidence().value;
            fs << "patch_count" << static_cast<int>(detectionResult_.getPatches().size());
            fs << "valid_patch_count" << static_cast<int>(detectionResult_.getValidPatchCount());
            fs << "}";

            // Save metadata
            fs << "metadata" << "{";
            for (const auto& [key, value] : metadata_) {
                fs << key << value;
            }
            
            // Add timestamp
            auto time_t = std::chrono::system_clock::to_time_t(timestamp_);
            fs << "calibration_timestamp" << static_cast<int64_t>(time_t);
            fs << "}";

            fs << "is_valid" << isValid_;
            
            fs.release();
            LOG_INFO("Calibration data saved to: ", filename);
            return true;
            
        } catch (const cv::Exception& e) {
            LOG_ERROR("OpenCV error while saving calibration data: ", e.what());
            return false;
        }
    }

    bool loadFromFile(const std::string& filename) {
        try {
            cv::FileStorage fs(filename, cv::FileStorage::READ);
            if (!fs.isOpened()) {
                LOG_ERROR("Cannot open calibration file for reading: ", filename);
                return false;
            }

            // Load correction matrix
            cv::Mat matrixMat;
            fs["correction_matrix"] >> matrixMat;
            if (matrixMat.rows == 3 && matrixMat.cols == 3) {
                Types::Matrix3x3 matrix;
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        matrix(i, j) = matrixMat.at<float>(i, j);
                    }
                }
                correctionMatrix_ = CorrectionMatrix(matrix);
            }

            // Load quality metrics
            cv::FileNode metricsNode = fs["quality_metrics"];
            if (!metricsNode.empty()) {
                qualityMetrics_.averageDeltaE = metricsNode["average_delta_e"];
                qualityMetrics_.maxDeltaE = metricsNode["max_delta_e"];
                qualityMetrics_.rmse = metricsNode["rmse"];
                qualityMetrics_.matrixConditionNumber = metricsNode["matrix_condition_number"];
                qualityMetrics_.matrixDeterminant = metricsNode["matrix_determinant"];
                qualityMetrics_.overallQuality = 
                    Types::ConfidenceScore::fromValue(metricsNode["overall_quality"]);
            }

            // Load metadata
            cv::FileNode metadataNode = fs["metadata"];
            if (!metadataNode.empty()) {
                for (auto it = metadataNode.begin(); it != metadataNode.end(); ++it) {
                    std::string key = (*it).name();
                    std::string value;
                    (*it) >> value;
                    metadata_[key] = value;
                }

                // Restore timestamp
                int64_t timestamp_int = metadataNode["calibration_timestamp"];
                if (timestamp_int > 0) {
                    timestamp_ = std::chrono::system_clock::from_time_t(
                        static_cast<std::time_t>(timestamp_int));
                }
            }

            fs["is_valid"] >> isValid_;
            fs.release();

            LOG_INFO("Calibration data loaded from: ", filename);
            return isValid_;
            
        } catch (const cv::Exception& e) {
            LOG_ERROR("OpenCV error while loading calibration data: ", e.what());
            return false;
        }
    }

    std::string getSummary() const {
        std::ostringstream oss;
        oss << "Calibration Summary:\n";
        oss << "  Valid: " << (isValid_ ? "Yes" : "No") << "\n";
        oss << "  Detection Method: " << detectionResult_.getMethodName() << "\n";
        oss << "  Detection Confidence: " << detectionResult_.getOverallConfidence().value << "\n";
        oss << "  Patches Detected: " << detectionResult_.getPatches().size() << "/" 
            << Types::COLORCHECKER_PATCHES << "\n";
        oss << "  Quality Metrics:\n";
        oss << "    Average Delta E: " << qualityMetrics_.averageDeltaE << "\n";
        oss << "    Max Delta E: " << qualityMetrics_.maxDeltaE << "\n";
        oss << "    RMSE: " << qualityMetrics_.rmse << "\n";
        oss << "    Matrix Condition: " << qualityMetrics_.matrixConditionNumber << "\n";
        oss << "    Overall Quality: " << qualityMetrics_.overallQuality.value << "\n";
        
        auto time_t = std::chrono::system_clock::to_time_t(timestamp_);
        oss << "  Calibrated: " << std::ctime(&time_t);
        
        return oss.str();
    }

  private:
    DetectionResult detectionResult_;
    CorrectionMatrix correctionMatrix_;
    QualityMetrics qualityMetrics_;
    std::chrono::system_clock::time_point timestamp_;
    std::map<std::string, std::string> metadata_;
    bool isValid_;

    void calculateQualityMetrics() {
        if (!detectionResult_.isSuccess() || !correctionMatrix_.isValid()) {
            qualityMetrics_ = QualityMetrics{};
            return;
        }

        const auto& patches = detectionResult_.getPatches();
        if (patches.empty()) {
            qualityMetrics_ = QualityMetrics{};
            return;
        }

        // Calculate Delta E statistics
        float deltaESum = 0.0f;
        float maxDeltaE = 0.0f;
        float rmseSum = 0.0f;
        size_t validPatches = 0;

        for (const auto& patch : patches) {
            if (!patch.isValid()) continue;
            
            float deltaE = patch.calculateDeltaE();
            deltaESum += deltaE;
            maxDeltaE = std::max(maxDeltaE, deltaE);
            rmseSum += deltaE * deltaE;
            validPatches++;
        }

        if (validPatches > 0) {
            qualityMetrics_.averageDeltaE = deltaESum / validPatches;
            qualityMetrics_.maxDeltaE = maxDeltaE;
            qualityMetrics_.rmse = std::sqrt(rmseSum / validPatches);
        }

        // Matrix quality metrics
        qualityMetrics_.matrixConditionNumber = correctionMatrix_.getConditionNumber();
        qualityMetrics_.matrixDeterminant = correctionMatrix_.getDeterminant();

        // Overall quality score (0-1)
        float qualityScore = 1.0f;
        
        // Penalize high Delta E
        if (qualityMetrics_.averageDeltaE > 1.0f) {
            qualityScore *= std::max(0.0f, 1.0f - (qualityMetrics_.averageDeltaE - 1.0f) / 2.0f);
        }
        
        // Penalize poor matrix conditioning
        if (qualityMetrics_.matrixConditionNumber > 100.0f) {
            qualityScore *= std::max(0.0f, 1.0f - 
                (qualityMetrics_.matrixConditionNumber - 100.0f) / 900.0f);
        }
        
        // Incorporate detection confidence
        qualityScore *= detectionResult_.getOverallConfidence().value;
        
        qualityMetrics_.overallQuality = Types::ConfidenceScore::fromValue(qualityScore);
    }

    void validate() {
        isValid_ = detectionResult_.isSuccess() && 
                  correctionMatrix_.isValid() && 
                  qualityMetrics_.isAcceptable();
        
        if (!isValid_) {
            LOG_WARN("Calibration data validation failed");
        }
    }
};

}  // namespace ColorCorrection::Domain