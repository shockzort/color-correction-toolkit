#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

// Include implementation headers (in a real build these would be properly linked)
#include "../internal/detection/MultiLevelDetector.hpp"
#include "../internal/processing/CorrectionMatrixCalculator.hpp"
#include "../internal/processing/QualityMetrics.hpp"
#include "../internal/processing/ImagePreprocessor.hpp"
#include <shared/utils/Logger.hpp>

using namespace ColorCorrection;

struct AppSettings {
    std::string inputPath;
    std::string outputCalibration = "calibration.xml";
    bool verbose = false;
    bool showPreview = false;
    float confidenceThreshold = 0.8f;
    std::string detectionMethod = "auto"; // auto, mcc, contour, template
    bool isVideo = false;
    int maxFrames = 10;
    int frameSkip = 5;
};

void printUsage(const char* programName) {
    std::cout << "Color Correction Calibration Tool\n";
    std::cout << "Usage: " << programName << " [options] <input>\n\n";
    std::cout << "Arguments:\n";
    std::cout << "  input                   Input image or video file with ColorChecker\n\n";
    std::cout << "Options:\n";
    std::cout << "  -o, --output <file>     Output calibration file (default: calibration.xml)\n";
    std::cout << "  -v, --verbose           Enable verbose output\n";
    std::cout << "  -p, --preview           Show detection preview window\n";
    std::cout << "  -t, --threshold <value> Detection confidence threshold (default: 0.8)\n";
    std::cout << "  -m, --method <method>   Detection method: auto|mcc|contour|template (default: auto)\n";
    std::cout << "  --video                 Force video processing mode\n";
    std::cout << "  --max-frames <n>        Maximum frames to analyze for video (default: 10)\n";
    std::cout << "  --frame-skip <n>        Skip frames between analysis (default: 5)\n";
    std::cout << "  -h, --help              Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  Image:  " << programName << " -o my_calibration.xml -v colorchecker_photo.jpg\n";
    std::cout << "  Video:  " << programName << " --video --max-frames 20 colorchecker_video.mp4\n";
}

AppSettings parseArguments(int argc, char* argv[]) {
    AppSettings settings;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            exit(0);
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) {
                settings.outputCalibration = argv[++i];
            } else {
                std::cerr << "Error: " << arg << " requires a value\n";
                exit(1);
            }
        } else if (arg == "-v" || arg == "--verbose") {
            settings.verbose = true;
        } else if (arg == "-p" || arg == "--preview") {
            settings.showPreview = true;
        } else if (arg == "-t" || arg == "--threshold") {
            if (i + 1 < argc) {
                settings.confidenceThreshold = std::stof(argv[++i]);
            } else {
                std::cerr << "Error: " << arg << " requires a value\n";
                exit(1);
            }
        } else if (arg == "-m" || arg == "--method") {
            if (i + 1 < argc) {
                settings.detectionMethod = argv[++i];
            } else {
                std::cerr << "Error: " << arg << " requires a value\n";
                exit(1);
            }
        } else if (arg == "--video") {
            settings.isVideo = true;
        } else if (arg == "--max-frames") {
            if (i + 1 < argc) {
                settings.maxFrames = std::stoi(argv[++i]);
            } else {
                std::cerr << "Error: " << arg << " requires a value\n";
                exit(1);
            }
        } else if (arg == "--frame-skip") {
            if (i + 1 < argc) {
                settings.frameSkip = std::stoi(argv[++i]);
            } else {
                std::cerr << "Error: " << arg << " requires a value\n";
                exit(1);
            }
        } else if (arg[0] != '-') {
            if (settings.inputPath.empty()) {
                settings.inputPath = arg;
            } else {
                std::cerr << "Error: Multiple input files specified\n";
                exit(1);
            }
        } else {
            std::cerr << "Error: Unknown option " << arg << "\n";
            exit(1);
        }
    }
    
    if (settings.inputPath.empty()) {
        std::cerr << "Error: No input file specified\n";
        printUsage(argv[0]);
        exit(1);
    }
    
    // Auto-detect video files if not explicitly specified
    if (!settings.isVideo) {
        std::string extension = std::filesystem::path(settings.inputPath).extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
        if (extension == ".mp4" || extension == ".avi" || extension == ".mov" || 
            extension == ".mkv" || extension == ".wmv") {
            settings.isVideo = true;
        }
    }
    
    return settings;
}

cv::Mat visualizeDetection(const cv::Mat& image, const Domain::DetectionResult& result) {
    cv::Mat visualization = image.clone();
    
    // Draw ColorChecker corners if available
    const auto& corners = result.getCorners();
    if (corners.size() == 4) {
        std::vector<cv::Point> cornerPoints;
        for (const auto& corner : corners) {
            cornerPoints.emplace_back(static_cast<int>(corner.x), static_cast<int>(corner.y));
        }
        
        // Draw bounding rectangle
        cv::polylines(visualization, cornerPoints, true, cv::Scalar(0, 255, 0), 3);
    }
    
    // Draw patch centers
    const auto& patches = result.getPatches();
    for (const auto& patch : patches) {
        cv::Point center(static_cast<int>(patch.getCenter().x), 
                        static_cast<int>(patch.getCenter().y));
        
        // Color by confidence
        float confidence = patch.getConfidence().value;
        cv::Scalar color;
        if (confidence > 0.8f) {
            color = cv::Scalar(0, 255, 0); // Green for high confidence
        } else if (confidence > 0.5f) {
            color = cv::Scalar(0, 255, 255); // Yellow for medium confidence
        } else {
            color = cv::Scalar(0, 0, 255); // Red for low confidence
        }
        
        cv::circle(visualization, center, 8, color, -1);
        cv::putText(visualization, std::to_string(patch.getPatchId()), 
                   cv::Point(center.x - 10, center.y - 15), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }
    
    // Add information overlay
    std::string info = "Method: " + result.getMethodName() + 
                      " | Confidence: " + std::to_string(result.getOverallConfidence().value) +
                      " | Patches: " + std::to_string(patches.size()) + "/24";
    
    cv::putText(visualization, info, cv::Point(10, 30), 
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    return visualization;
}

bool processVideoCalibration(const AppSettings& settings) {
    std::cout << "Processing video for calibration...\n";
    
    cv::VideoCapture cap(settings.inputPath);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video file " << settings.inputPath << "\n";
        return false;
    }
    
    int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    
    std::cout << "Video info: " << totalFrames << " frames, " << fps << " FPS\n";
    
    // Setup detector
    Internal::Detection::MultiLevelDetector::DetectionStrategy strategy;
    if (settings.detectionMethod == "mcc") {
        strategy.enableContour = false;
        strategy.enableTemplate = false;
    } else if (settings.detectionMethod == "contour") {
        strategy.enableMCC = false;
        strategy.enableTemplate = false;
    } else if (settings.detectionMethod == "template") {
        strategy.enableMCC = false;
        strategy.enableContour = false;
    }
    
    Internal::Detection::MultiLevelDetector detector(strategy);
    detector.setConfidenceThreshold(settings.confidenceThreshold);
    
    Internal::Processing::ImagePreprocessor preprocessor;
    
    std::vector<Domain::DetectionResult> validDetections;
    std::vector<cv::Mat> validFrames;
    
    int frameCount = 0;
    int analyzedFrames = 0;
    cv::Mat frame;
    
    std::cout << "Analyzing frames for ColorChecker detection...\n";
    
    while (cap.read(frame) && analyzedFrames < settings.maxFrames) {
        frameCount++;
        
        // Skip frames according to settings
        if (frameCount % settings.frameSkip != 0) {
            continue;
        }
        
        analyzedFrames++;
        std::cout << "\rAnalyzing frame " << analyzedFrames << "/" << settings.maxFrames;
        std::cout.flush();
        
        // Preprocess frame
        cv::Mat processedFrame = preprocessor.preprocess(frame);
        
        // Attempt detection
        Domain::DetectionResult detection = detector.detect(processedFrame);
        
        if (detection.isSuccess() && detection.getPatches().size() >= 20) {
            validDetections.push_back(detection);
            validFrames.push_back(frame.clone());
            
            if (settings.verbose) {
                std::cout << "\n  Frame " << frameCount << ": Detection successful (confidence: " 
                         << detection.getOverallConfidence().value << ", patches: " 
                         << detection.getPatches().size() << ")\n";
            }
        } else if (settings.verbose) {
            std::cout << "\n  Frame " << frameCount << ": Detection failed\n";
        }
    }
    
    std::cout << "\n\nDetection complete. Found " << validDetections.size() 
              << " valid detections from " << analyzedFrames << " analyzed frames\n";
    
    if (validDetections.empty()) {
        std::cerr << "Error: No valid ColorChecker detections found in video\n";
        std::cerr << "Try adjusting detection parameters or ensure ColorChecker is clearly visible\n";
        return false;
    }
    
    // Select best detection based on confidence and patch count
    auto bestDetection = std::max_element(validDetections.begin(), validDetections.end(),
        [](const Domain::DetectionResult& a, const Domain::DetectionResult& b) {
            float scoreA = a.getOverallConfidence().value * a.getPatches().size();
            float scoreB = b.getOverallConfidence().value * b.getPatches().size();
            return scoreA < scoreB;
        });
    
    size_t bestIndex = std::distance(validDetections.begin(), bestDetection);
    
    std::cout << "Selected best detection from frame analysis:\n";
    std::cout << "  Confidence: " << bestDetection->getOverallConfidence().value << "\n";
    std::cout << "  Patches: " << bestDetection->getPatches().size() << "/24\n";
    std::cout << "  Method: " << bestDetection->getMethodName() << "\n";
    
    // Show preview if requested
    if (settings.showPreview && bestIndex < validFrames.size()) {
        cv::Mat visualization = visualizeDetection(validFrames[bestIndex], *bestDetection);
        cv::imshow("Best ColorChecker Detection", visualization);
        std::cout << "Press any key to continue...\n";
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    
    // Calculate correction matrix using best detection
    std::cout << "Calculating correction matrix from best detection...\n";
    Internal::Processing::CorrectionMatrixCalculator calculator;
    auto calculationResult = calculator.calculateMatrix(bestDetection->getPatches());
    
    if (!calculationResult.success) {
        std::cerr << "Error: Failed to calculate correction matrix\n";
        std::cerr << "Reason: " << calculationResult.errorMessage << "\n";
        return false;
    }
    
    std::cout << "Correction matrix calculated successfully!\n";
    std::cout << "  Used patches: " << calculationResult.usedPatches << "/" << calculationResult.totalPatches << "\n";
    std::cout << "  Mean Square Error: " << calculationResult.meanSquareError << "\n";
    std::cout << "  Condition Number: " << calculationResult.conditionNumber << "\n";
    
    // Calculate quality metrics
    std::cout << "Assessing calibration quality...\n";
    Internal::Processing::QualityMetrics qualityCalculator;
    Domain::CalibrationData calibrationData(*bestDetection, calculationResult.matrix);
    
    // Add video-specific metadata
    calibrationData.addMetadata("source_type", "video");
    calibrationData.addMetadata("source_file", settings.inputPath);
    calibrationData.addMetadata("total_frames", std::to_string(totalFrames));
    calibrationData.addMetadata("analyzed_frames", std::to_string(analyzedFrames));
    calibrationData.addMetadata("valid_detections", std::to_string(validDetections.size()));
    calibrationData.addMetadata("fps", std::to_string(fps));
    
    auto qualityReport = qualityCalculator.calculateQualityReport(bestDetection->getPatches(), 
                                                                  calculationResult.matrix);
    
    std::cout << "Quality Assessment:\n";
    std::cout << "  Grade: " << qualityReport.qualityGrade << "\n";
    std::cout << "  Average ΔE: " << qualityReport.averageDeltaE << "\n";
    std::cout << "  Max ΔE: " << qualityReport.maxDeltaE << "\n";
    std::cout << "  R² Score: " << qualityReport.r2Score << "\n";
    std::cout << "  Overall Score: " << qualityReport.overallScore.value << "\n";
    
    // Save calibration data
    std::cout << "Saving calibration data...\n";
    if (!calibrationData.saveToFile(settings.outputCalibration)) {
        std::cerr << "Error: Failed to save calibration data\n";
        return false;
    }
    
    std::cout << "Video calibration completed successfully!\n";
    std::cout << "Calibration file saved: " << settings.outputCalibration << "\n";
    
    return true;
}

bool processImageCalibration(const AppSettings& settings) {
    std::cout << "Processing image for calibration...\n";
    
    // Load input image
    cv::Mat image = cv::imread(settings.inputPath, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: Cannot load image " << settings.inputPath << "\n";
        return false;
    }
    
    std::cout << "Image loaded: " << image.cols << "x" << image.rows << " pixels\n";
    
    // Preprocess image
    std::cout << "Preprocessing image...\n";
    Internal::Processing::ImagePreprocessor preprocessor;
    cv::Mat processedImage = preprocessor.preprocess(image);
    
    // Setup detector
    Internal::Detection::MultiLevelDetector::DetectionStrategy strategy;
    if (settings.detectionMethod == "mcc") {
        strategy.enableContour = false;
        strategy.enableTemplate = false;
    } else if (settings.detectionMethod == "contour") {
        strategy.enableMCC = false;
        strategy.enableTemplate = false;
    } else if (settings.detectionMethod == "template") {
        strategy.enableMCC = false;
        strategy.enableContour = false;
    }
    
    Internal::Detection::MultiLevelDetector detector(strategy);
    detector.setConfidenceThreshold(settings.confidenceThreshold);
    
    // Perform detection
    std::cout << "Detecting ColorChecker...\n";
    Domain::DetectionResult detection = detector.detect(processedImage);
    
    if (!detection.isSuccess()) {
        std::cerr << "Error: ColorChecker detection failed\n";
        return false;
    }
    
    std::cout << "ColorChecker detected successfully!\n";
    std::cout << "  Method: " << detection.getMethodName() << "\n";
    std::cout << "  Confidence: " << detection.getOverallConfidence().value << "\n";
    std::cout << "  Patches found: " << detection.getPatches().size() << "/24\n";
    
    // Show preview if requested
    if (settings.showPreview) {
        cv::Mat visualization = visualizeDetection(image, detection);
        cv::imshow("ColorChecker Detection", visualization);
        std::cout << "Press any key to continue...\n";
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    
    // Calculate correction matrix
    std::cout << "Calculating correction matrix...\n";
    Internal::Processing::CorrectionMatrixCalculator calculator;
    auto calculationResult = calculator.calculateMatrix(detection.getPatches());
    
    if (!calculationResult.success) {
        std::cerr << "Error: Failed to calculate correction matrix\n";
        std::cerr << "Reason: " << calculationResult.errorMessage << "\n";
        return false;
    }
    
    std::cout << "Correction matrix calculated successfully!\n";
    std::cout << "  Used patches: " << calculationResult.usedPatches << "/" << calculationResult.totalPatches << "\n";
    std::cout << "  Mean Square Error: " << calculationResult.meanSquareError << "\n";
    std::cout << "  Condition Number: " << calculationResult.conditionNumber << "\n";
    
    // Calculate quality metrics
    std::cout << "Assessing calibration quality...\n";
    Internal::Processing::QualityMetrics qualityCalculator;
    Domain::CalibrationData calibrationData(detection, calculationResult.matrix);
    calibrationData.setCalibrationImage(image);
    
    auto qualityReport = qualityCalculator.calculateQualityReport(detection.getPatches(), 
                                                                  calculationResult.matrix);
    
    std::cout << "Quality Assessment:\n";
    std::cout << "  Grade: " << qualityReport.qualityGrade << "\n";
    std::cout << "  Average ΔE: " << qualityReport.averageDeltaE << "\n";
    std::cout << "  Max ΔE: " << qualityReport.maxDeltaE << "\n";
    std::cout << "  R² Score: " << qualityReport.r2Score << "\n";
    std::cout << "  Overall Score: " << qualityReport.overallScore.value << "\n";
    
    // Save calibration data
    std::cout << "Saving calibration data...\n";
    if (!calibrationData.saveToFile(settings.outputCalibration)) {
        std::cerr << "Error: Failed to save calibration data\n";
        return false;
    }
    
    std::cout << "Image calibration completed successfully!\n";
    std::cout << "Calibration file saved: " << settings.outputCalibration << "\n";
    
    return true;
}

int main(int argc, char* argv[]) {
    try {
        AppSettings settings = parseArguments(argc, argv);
        
        // Configure logging
        if (settings.verbose) {
            Shared::Logger::getInstance().setLevel(Shared::LogLevel::DEBUG);
        } else {
            Shared::Logger::getInstance().setLevel(Shared::LogLevel::INFO);
        }
        
        std::cout << "Color Correction Calibration Tool\n";
        std::cout << "==================================\n";
        std::cout << "Input: " << settings.inputPath << "\n";
        std::cout << "Output calibration: " << settings.outputCalibration << "\n";
        std::cout << "Mode: " << (settings.isVideo ? "Video" : "Image") << "\n";
        std::cout << "Detection method: " << settings.detectionMethod << "\n";
        std::cout << "Confidence threshold: " << settings.confidenceThreshold << "\n";
        
        if (settings.isVideo) {
            std::cout << "Max frames to analyze: " << settings.maxFrames << "\n";
            std::cout << "Frame skip interval: " << settings.frameSkip << "\n";
        }
        std::cout << "\n";
        
        // Process based on input type
        bool success;
        if (settings.isVideo) {
            success = processVideoCalibration(settings);
        } else {
            success = processImageCalibration(settings);
        }
        
        if (!success) {
            std::cerr << "Calibration failed. See error messages above.\n";
            return 1;
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}