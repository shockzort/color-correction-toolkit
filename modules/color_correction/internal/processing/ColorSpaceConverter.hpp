#pragma once

#include <shared/types/Common.hpp>
#include <shared/utils/Logger.hpp>
#include <opencv2/opencv.hpp>
#include <cmath>

namespace ColorCorrection::Internal::Processing {

class ColorSpaceConverter {
  public:
    // Standard sRGB gamma correction parameters
    static constexpr float SRGB_GAMMA = 2.4f;
    static constexpr float SRGB_ALPHA = 1.055f;
    static constexpr float SRGB_BETA = 0.04045f;
    static constexpr float SRGB_LINEAR_THRESHOLD = 0.003131f;

    // CIE illuminants (D65 white point)
    static constexpr float D65_X = 0.95047f;
    static constexpr float D65_Y = 1.00000f;
    static constexpr float D65_Z = 1.08883f;

    ColorSpaceConverter() {
        LOG_DEBUG("Color space converter initialized");
    }

    // sRGB to Linear RGB conversion
    Types::ColorValue sRGBToLinear(const Types::ColorValue& srgb) const {
        Types::ColorValue linear;
        
        for (int i = 0; i < 3; ++i) {
            float component = srgb[i];
            
            if (component <= SRGB_BETA) {
                linear[i] = component / 12.92f;
            } else {
                linear[i] = std::pow((component + SRGB_ALPHA - 1.0f) / SRGB_ALPHA, SRGB_GAMMA);
            }
        }
        
        return linear;
    }

    // Linear RGB to sRGB conversion
    Types::ColorValue linearToSRGB(const Types::ColorValue& linear) const {
        Types::ColorValue srgb;
        
        for (int i = 0; i < 3; ++i) {
            float component = linear[i];
            
            if (component <= SRGB_LINEAR_THRESHOLD) {
                srgb[i] = component * 12.92f;
            } else {
                srgb[i] = SRGB_ALPHA * std::pow(component, 1.0f / SRGB_GAMMA) - (SRGB_ALPHA - 1.0f);
            }
        }
        
        return srgb;
    }

    // Convert entire image from sRGB to Linear RGB
    Types::Image convertImageSRGBToLinear(const Types::Image& srgbImage) const {
        if (srgbImage.empty()) {
            LOG_ERROR("Input image is empty");
            return Types::Image();
        }

        Types::Image linearImage;
        
        try {
            // Convert to float for processing
            Types::Image floatImage;
            srgbImage.convertTo(floatImage, CV_32F, 1.0/255.0);
            
            // Apply gamma correction
            if (floatImage.channels() >= 3) {
                std::vector<Types::Image> channels;
                cv::split(floatImage, channels);
                
                for (int i = 0; i < 3; ++i) {  // Process RGB channels only
                    applySRGBToLinearLUT(channels[i]);
                }
                
                cv::merge(channels, linearImage);
            } else {
                // Grayscale
                linearImage = floatImage.clone();
                applySRGBToLinearLUT(linearImage);
            }
            
            LOG_DEBUG("Converted ", srgbImage.cols, "x", srgbImage.rows, " image from sRGB to linear");
            
        } catch (const cv::Exception& e) {
            LOG_ERROR("Error converting sRGB to linear: ", e.what());
            return srgbImage;
        }
        
        return linearImage;
    }

    // Convert entire image from Linear RGB to sRGB
    Types::Image convertImageLinearToSRGB(const Types::Image& linearImage) const {
        if (linearImage.empty()) {
            LOG_ERROR("Input image is empty");
            return Types::Image();
        }

        Types::Image srgbImage;
        
        try {
            Types::Image processedImage = linearImage.clone();
            
            // Ensure float format
            if (processedImage.type() != CV_32F) {
                processedImage.convertTo(processedImage, CV_32F);
            }
            
            // Apply inverse gamma correction
            if (processedImage.channels() >= 3) {
                std::vector<Types::Image> channels;
                cv::split(processedImage, channels);
                
                for (int i = 0; i < 3; ++i) {  // Process RGB channels only
                    applyLinearToSRGBLUT(channels[i]);
                }
                
                cv::merge(channels, processedImage);
            } else {
                // Grayscale
                applyLinearToSRGBLUT(processedImage);
            }
            
            // Convert back to 8-bit
            processedImage.convertTo(srgbImage, CV_8U, 255.0);
            
            LOG_DEBUG("Converted ", linearImage.cols, "x", linearImage.rows, " image from linear to sRGB");
            
        } catch (const cv::Exception& e) {
            LOG_ERROR("Error converting linear to sRGB: ", e.what());
            return linearImage;
        }
        
        return srgbImage;
    }

    // Linear RGB to XYZ conversion (using sRGB matrix)
    Types::ColorValue linearRGBToXYZ(const Types::ColorValue& rgb) const {
        // sRGB to XYZ matrix (D65 illuminant)
        static const cv::Matx33f rgbToXyzMatrix(
            0.4124564f, 0.3575761f, 0.1804375f,
            0.2126729f, 0.7151522f, 0.0721750f,
            0.0193339f, 0.1191920f, 0.9503041f
        );
        
        cv::Vec3f rgbVec(rgb[0], rgb[1], rgb[2]);
        cv::Vec3f xyzVec = rgbToXyzMatrix * rgbVec;
        
        return Types::ColorValue(xyzVec[0], xyzVec[1], xyzVec[2]);
    }

    // XYZ to Linear RGB conversion
    Types::ColorValue xyzToLinearRGB(const Types::ColorValue& xyz) const {
        // XYZ to sRGB matrix (D65 illuminant)
        static const cv::Matx33f xyzToRgbMatrix(
             3.2404542f, -1.5371385f, -0.4985314f,
            -0.9692660f,  1.8760108f,  0.0415560f,
             0.0556434f, -0.2040259f,  1.0572252f
        );
        
        cv::Vec3f xyzVec(xyz[0], xyz[1], xyz[2]);
        cv::Vec3f rgbVec = xyzToRgbMatrix * xyzVec;
        
        return Types::ColorValue(rgbVec[0], rgbVec[1], rgbVec[2]);
    }

    // XYZ to LAB conversion
    Types::ColorValue xyzToLab(const Types::ColorValue& xyz) const {
        float x = xyz[0] / D65_X;
        float y = xyz[1] / D65_Y;
        float z = xyz[2] / D65_Z;
        
        x = (x > 0.008856f) ? std::pow(x, 1.0f/3.0f) : (7.787f * x + 16.0f/116.0f);
        y = (y > 0.008856f) ? std::pow(y, 1.0f/3.0f) : (7.787f * y + 16.0f/116.0f);
        z = (z > 0.008856f) ? std::pow(z, 1.0f/3.0f) : (7.787f * z + 16.0f/116.0f);
        
        float L = 116.0f * y - 16.0f;
        float a = 500.0f * (x - y);
        float b = 200.0f * (y - z);
        
        return Types::ColorValue(L, a, b);
    }

    // LAB to XYZ conversion
    Types::ColorValue labToXYZ(const Types::ColorValue& lab) const {
        float L = lab[0];
        float a = lab[1];
        float b = lab[2];
        
        float fy = (L + 16.0f) / 116.0f;
        float fx = a / 500.0f + fy;
        float fz = fy - b / 200.0f;
        
        float x = (fx * fx * fx > 0.008856f) ? fx * fx * fx : (fx - 16.0f/116.0f) / 7.787f;
        float y = (fy * fy * fy > 0.008856f) ? fy * fy * fy : (fy - 16.0f/116.0f) / 7.787f;
        float z = (fz * fz * fz > 0.008856f) ? fz * fz * fz : (fz - 16.0f/116.0f) / 7.787f;
        
        return Types::ColorValue(x * D65_X, y * D65_Y, z * D65_Z);
    }

    // Direct sRGB to LAB conversion (convenience function)
    Types::ColorValue sRGBToLab(const Types::ColorValue& srgb) const {
        Types::ColorValue linear = sRGBToLinear(srgb);
        Types::ColorValue xyz = linearRGBToXYZ(linear);
        return xyzToLab(xyz);
    }

    // Direct LAB to sRGB conversion (convenience function)
    Types::ColorValue labToSRGB(const Types::ColorValue& lab) const {
        Types::ColorValue xyz = labToXYZ(lab);
        Types::ColorValue linear = xyzToLinearRGB(xyz);
        return linearToSRGB(linear);
    }

    // Convert image between different color spaces
    Types::Image convertImageColorSpace(const Types::Image& input, 
                                      Types::ColorSpace from, 
                                      Types::ColorSpace to) const {
        if (input.empty() || from == to) {
            return input;
        }

        try {
            // Use OpenCV's built-in conversions when possible for better performance
            if (from == Types::ColorSpace::SRGB && to == Types::ColorSpace::LAB) {
                Types::Image result;
                cv::cvtColor(input, result, cv::COLOR_BGR2Lab);
                return result;
            } else if (from == Types::ColorSpace::LAB && to == Types::ColorSpace::SRGB) {
                Types::Image result;
                cv::cvtColor(input, result, cv::COLOR_Lab2BGR);
                return result;
            } else if (from == Types::ColorSpace::SRGB && to == Types::ColorSpace::LINEAR_RGB) {
                return convertImageSRGBToLinear(input);
            } else if (from == Types::ColorSpace::LINEAR_RGB && to == Types::ColorSpace::SRGB) {
                return convertImageLinearToSRGB(input);
            }

            // For other conversions, use manual pixel-by-pixel conversion
            return convertImageManual(input, from, to);
            
        } catch (const cv::Exception& e) {
            LOG_ERROR("Error converting color space: ", e.what());
            return input;
        }
    }

    // Validate color values are in expected range
    bool validateColorValues(const Types::ColorValue& color, Types::ColorSpace colorSpace) const {
        switch (colorSpace) {
            case Types::ColorSpace::SRGB:
            case Types::ColorSpace::LINEAR_RGB:
                return color[0] >= 0.0f && color[0] <= 1.0f &&
                       color[1] >= 0.0f && color[1] <= 1.0f &&
                       color[2] >= 0.0f && color[2] <= 1.0f;
                       
            case Types::ColorSpace::LAB:
                return color[0] >= 0.0f && color[0] <= 100.0f &&    // L: 0-100
                       color[1] >= -128.0f && color[1] <= 127.0f &&  // a: -128 to +127
                       color[2] >= -128.0f && color[2] <= 127.0f;    // b: -128 to +127
                       
            case Types::ColorSpace::XYZ:
                return color[0] >= 0.0f && color[1] >= 0.0f && color[2] >= 0.0f;
                
            default:
                return false;
        }
    }

    // Clamp color values to valid range
    Types::ColorValue clampColorValues(const Types::ColorValue& color, Types::ColorSpace colorSpace) const {
        Types::ColorValue clamped = color;
        
        switch (colorSpace) {
            case Types::ColorSpace::SRGB:
            case Types::ColorSpace::LINEAR_RGB:
                clamped[0] = std::clamp(clamped[0], 0.0f, 1.0f);
                clamped[1] = std::clamp(clamped[1], 0.0f, 1.0f);
                clamped[2] = std::clamp(clamped[2], 0.0f, 1.0f);
                break;
                
            case Types::ColorSpace::LAB:
                clamped[0] = std::clamp(clamped[0], 0.0f, 100.0f);
                clamped[1] = std::clamp(clamped[1], -128.0f, 127.0f);
                clamped[2] = std::clamp(clamped[2], -128.0f, 127.0f);
                break;
                
            case Types::ColorSpace::XYZ:
                clamped[0] = std::max(0.0f, clamped[0]);
                clamped[1] = std::max(0.0f, clamped[1]);
                clamped[2] = std::max(0.0f, clamped[2]);
                break;
        }
        
        return clamped;
    }

  private:
    void applySRGBToLinearLUT(Types::Image& channel) const {
        channel.forEach<float>([this](float& pixel, const int*) {
            pixel = sRGBToLinearSingle(pixel);
        });
    }

    void applyLinearToSRGBLUT(Types::Image& channel) const {
        channel.forEach<float>([this](float& pixel, const int*) {
            pixel = linearToSRGBSingle(pixel);
        });
    }

    float sRGBToLinearSingle(float srgb) const {
        if (srgb <= SRGB_BETA) {
            return srgb / 12.92f;
        } else {
            return std::pow((srgb + SRGB_ALPHA - 1.0f) / SRGB_ALPHA, SRGB_GAMMA);
        }
    }

    float linearToSRGBSingle(float linear) const {
        if (linear <= SRGB_LINEAR_THRESHOLD) {
            return linear * 12.92f;
        } else {
            return SRGB_ALPHA * std::pow(linear, 1.0f / SRGB_GAMMA) - (SRGB_ALPHA - 1.0f);
        }
    }

    Types::Image convertImageManual(const Types::Image& input, 
                                  Types::ColorSpace from, 
                                  Types::ColorSpace to) const {
        Types::Image output = input.clone();
        
        if (input.channels() < 3) {
            LOG_WARN("Manual color space conversion requires RGB image");
            return input;
        }

        // Convert to float for processing
        output.convertTo(output, CV_32F, 1.0/255.0);
        
        for (int y = 0; y < output.rows; ++y) {
            for (int x = 0; x < output.cols; ++x) {
                cv::Vec3f& pixel = output.at<cv::Vec3f>(y, x);
                Types::ColorValue color(pixel[2], pixel[1], pixel[0]); // BGR to RGB
                
                // Convert through the conversion chain
                Types::ColorValue converted = convertColorValue(color, from, to);
                
                pixel[0] = converted[2]; // R to B
                pixel[1] = converted[1]; // G to G
                pixel[2] = converted[0]; // B to R
            }
        }
        
        // Convert back to appropriate format
        if (to == Types::ColorSpace::SRGB) {
            output.convertTo(output, CV_8U, 255.0);
        }
        
        return output;
    }

    Types::ColorValue convertColorValue(const Types::ColorValue& input,
                                      Types::ColorSpace from,
                                      Types::ColorSpace to) const {
        if (from == to) return input;
        
        // Convert to a common intermediate format (XYZ) if needed
        Types::ColorValue xyz;
        
        switch (from) {
            case Types::ColorSpace::SRGB:
                xyz = linearRGBToXYZ(sRGBToLinear(input));
                break;
            case Types::ColorSpace::LINEAR_RGB:
                xyz = linearRGBToXYZ(input);
                break;
            case Types::ColorSpace::LAB:
                xyz = labToXYZ(input);
                break;
            case Types::ColorSpace::XYZ:
                xyz = input;
                break;
        }
        
        // Convert from XYZ to target color space
        switch (to) {
            case Types::ColorSpace::SRGB:
                return linearToSRGB(xyzToLinearRGB(xyz));
            case Types::ColorSpace::LINEAR_RGB:
                return xyzToLinearRGB(xyz);
            case Types::ColorSpace::LAB:
                return xyzToLab(xyz);
            case Types::ColorSpace::XYZ:
                return xyz;
        }
        
        return input; // Fallback
    }
};

}  // namespace ColorCorrection::Internal::Processing