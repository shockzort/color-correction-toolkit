#pragma once

#include <shared/types/Common.hpp>
#include <shared/utils/Logger.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <string>

namespace ColorCorrection::Domain {

class CorrectionMatrix {
  public:
    CorrectionMatrix() : matrix_(Types::Matrix3x3::eye()), isValid_(false) {}

    explicit CorrectionMatrix(const Types::Matrix3x3& matrix) 
        : matrix_(matrix), isValid_(false) {
        validate();
    }

    static CorrectionMatrix identity() {
        CorrectionMatrix correctionMatrix;
        correctionMatrix.matrix_ = Types::Matrix3x3::eye();
        correctionMatrix.isValid_ = true;
        return correctionMatrix;
    }

    const Types::Matrix3x3& getMatrix() const { return matrix_; }

    void setMatrix(const Types::Matrix3x3& matrix) {
        matrix_ = matrix;
        validate();
    }

    bool isValid() const { return isValid_; }

    Types::ColorValue applyCorrection(const Types::ColorValue& inputColor) const {
        if (!isValid_) {
            LOG_ERROR("Attempting to apply invalid correction matrix");
            return inputColor;
        }

        // Apply matrix transformation: output = matrix * input
        Types::ColorValue result;
        result[0] = matrix_(0, 0) * inputColor[0] + matrix_(0, 1) * inputColor[1] + 
                   matrix_(0, 2) * inputColor[2];
        result[1] = matrix_(1, 0) * inputColor[0] + matrix_(1, 1) * inputColor[1] + 
                   matrix_(1, 2) * inputColor[2];
        result[2] = matrix_(2, 0) * inputColor[0] + matrix_(2, 1) * inputColor[1] + 
                   matrix_(2, 2) * inputColor[2];

        // Clamp values to valid range [0, 1]
        result[0] = std::clamp(result[0], 0.0f, 1.0f);
        result[1] = std::clamp(result[1], 0.0f, 1.0f);
        result[2] = std::clamp(result[2], 0.0f, 1.0f);

        return result;
    }

    void applyToImage(const Types::Image& input, Types::Image& output) const {
        if (!isValid_) {
            LOG_ERROR("Cannot apply invalid correction matrix to image");
            input.copyTo(output);
            return;
        }

        if (input.empty()) {
            LOG_ERROR("Input image is empty");
            return;
        }

        // Convert matrix to OpenCV format for efficient processing
        cv::Mat_<float> cvMatrix(3, 3);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                cvMatrix(i, j) = matrix_(i, j);
            }
        }

        // Apply color correction
        input.convertTo(output, CV_32F, 1.0 / 255.0);  // Normalize to [0,1]
        cv::transform(output, output, cvMatrix);
        output.convertTo(output, input.type(), 255.0);  // Convert back to original range
        
        // Clamp values
        cv::threshold(output, output, 255, 255, cv::THRESH_TRUNC);
        cv::threshold(output, output, 0, 0, cv::THRESH_TOZERO);
    }

    float getDeterminant() const {
        return cv::determinant(cv::Mat(matrix_));
    }

    float getConditionNumber() const {
        cv::Mat matrixMat(matrix_);
        cv::Mat eigenvalues;
        cv::eigen(matrixMat.t() * matrixMat, eigenvalues);
        
        if (eigenvalues.rows < 3) return std::numeric_limits<float>::max();
        
        float maxEigen = eigenvalues.at<float>(0);
        float minEigen = eigenvalues.at<float>(2);
        
        if (minEigen <= 0) return std::numeric_limits<float>::max();
        
        return maxEigen / minEigen;
    }

    bool saveToFile(const std::string& filename) const {
        try {
            cv::FileStorage fs(filename, cv::FileStorage::WRITE);
            if (!fs.isOpened()) {
                LOG_ERROR("Cannot open file for writing: ", filename);
                return false;
            }

            cv::Mat matrixMat(matrix_);
            fs << "correction_matrix" << matrixMat;
            fs << "is_valid" << isValid_;
            fs << "determinant" << getDeterminant();
            fs << "condition_number" << getConditionNumber();
            
            fs.release();
            LOG_INFO("Correction matrix saved to: ", filename);
            return true;
        } catch (const cv::Exception& e) {
            LOG_ERROR("OpenCV error while saving matrix: ", e.what());
            return false;
        }
    }

    bool loadFromFile(const std::string& filename) {
        try {
            cv::FileStorage fs(filename, cv::FileStorage::READ);
            if (!fs.isOpened()) {
                LOG_ERROR("Cannot open file for reading: ", filename);
                return false;
            }

            cv::Mat matrixMat;
            fs["correction_matrix"] >> matrixMat;
            
            if (matrixMat.rows != 3 || matrixMat.cols != 3) {
                LOG_ERROR("Invalid matrix dimensions in file: ", filename);
                return false;
            }

            // Convert to Types::Matrix3x3
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    matrix_(i, j) = matrixMat.at<float>(i, j);
                }
            }

            fs.release();
            validate();
            LOG_INFO("Correction matrix loaded from: ", filename);
            return isValid_;
        } catch (const cv::Exception& e) {
            LOG_ERROR("OpenCV error while loading matrix: ", e.what());
            return false;
        }
    }

  private:
    Types::Matrix3x3 matrix_;
    bool isValid_;

    void validate() {
        // Check for NaN or infinity values
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                if (!std::isfinite(matrix_(i, j))) {
                    LOG_ERROR("Matrix contains non-finite values");
                    isValid_ = false;
                    return;
                }
            }
        }

        // Check determinant (should not be zero or too close to zero)
        float det = getDeterminant();
        if (std::abs(det) < 1e-6f) {
            LOG_WARN("Matrix is nearly singular (determinant: ", det, ")");
            isValid_ = false;
            return;
        }

        // Check condition number (should not be too large)
        float conditionNumber = getConditionNumber();
        if (conditionNumber > 1000.0f) {
            LOG_WARN("Matrix is poorly conditioned (condition number: ", conditionNumber, ")");
        }

        isValid_ = true;
        LOG_DEBUG("Matrix validation passed (det: ", det, ", cond: ", conditionNumber, ")");
    }
};

}  // namespace ColorCorrection::Domain