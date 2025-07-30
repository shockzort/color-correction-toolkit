#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <filesystem>

// Include implementation headers
#include "../internal/correction/LinearCorrector.hpp"
#include "../internal/correction/BatchProcessor.hpp"
#include "../internal/domain/CalibrationData.hpp"
#include <shared/utils/Logger.hpp>

using namespace ColorCorrection;

struct AppSettings {
    std::string calibrationFile;
    std::string inputPath;
    std::string outputPath;
    bool batchMode = false;
    bool videoMode = false;
    bool verbose = false;
    bool showPreview = false;
    std::string outputFormat = "png";
    int jpegQuality = 95;
    bool useLinearRGB = true;
    
    // Video-specific settings
    double videoFPS = 30.0;
    int videoCodec = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    bool preserveOriginalFPS = true;
};

void printUsage(const char* programName) {
    std::cout << "Color Correction Application\n";
    std::cout << "Usage: " << programName << " [options] <calibration_file> <input> <output>\n\n";
    std::cout << "Arguments:\n";
    std::cout << "  calibration_file        Calibration file from calibration tool\n";
    std::cout << "  input                   Input image, video file, or directory\n";
    std::cout << "  output                  Output image, video file, or directory\n\n";
    std::cout << "Options:\n";
    std::cout << "  -b, --batch             Enable batch processing mode (for directories)\n";
    std::cout << "  --video                 Force video processing mode\n";
    std::cout << "  -v, --verbose           Enable verbose output\n";
    std::cout << "  -p, --preview           Show before/after preview (images only)\n";
    std::cout << "  -f, --format <format>   Output format: png|jpg|bmp|tiff|mp4|avi (default: png)\n";
    std::cout << "  -q, --quality <value>   JPEG quality 1-100 (default: 95)\n";
    std::cout << "  --fps <value>           Output video FPS (default: preserve original)\n";
    std::cout << "  --srgb                  Process in sRGB space (faster, less accurate)\n";
    std::cout << "  -h, --help              Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  Single image: " << programName << " calibration.xml input.jpg output.jpg\n";
    std::cout << "  Video:        " << programName << " calibration.xml input.mp4 output.mp4\n";
    std::cout << "  Batch mode:   " << programName << " -b calibration.xml input_dir/ output_dir/\n";
}

AppSettings parseArguments(int argc, char* argv[]) {
    AppSettings settings;
    std::vector<std::string> positionalArgs;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            exit(0);
        } else if (arg == "-b" || arg == "--batch") {
            settings.batchMode = true;
        } else if (arg == "--video") {
            settings.videoMode = true;
        } else if (arg == "-v" || arg == "--verbose") {
            settings.verbose = true;
        } else if (arg == "-p" || arg == "--preview") {
            settings.showPreview = true;
        } else if (arg == "--srgb") {
            settings.useLinearRGB = false;
        } else if (arg == "--fps") {
            if (i + 1 < argc) {
                settings.videoFPS = std::stod(argv[++i]);
                settings.preserveOriginalFPS = false;
            } else {
                std::cerr << "Error: " << arg << " requires a value\n";
                exit(1);
            }
        } else if (arg == "-f" || arg == "--format") {
            if (i + 1 < argc) {
                settings.outputFormat = argv[++i];
            } else {
                std::cerr << "Error: " << arg << " requires a value\n";
                exit(1);
            }
        } else if (arg == "-q" || arg == "--quality") {
            if (i + 1 < argc) {
                settings.jpegQuality = std::stoi(argv[++i]);
                if (settings.jpegQuality < 1 || settings.jpegQuality > 100) {
                    std::cerr << "Error: JPEG quality must be between 1 and 100\n";
                    exit(1);
                }
            } else {
                std::cerr << "Error: " << arg << " requires a value\n";
                exit(1);
            }
        } else if (arg[0] != '-') {
            positionalArgs.push_back(arg);
        } else {
            std::cerr << "Error: Unknown option " << arg << "\n";
            exit(1);
        }
    }
    
    if (positionalArgs.size() != 3) {
        std::cerr << "Error: Expected 3 arguments (calibration_file, input, output)\n";
        printUsage(argv[0]);
        exit(1);
    }
    
    settings.calibrationFile = positionalArgs[0];
    settings.inputPath = positionalArgs[1];
    settings.outputPath = positionalArgs[2];
    
    // Auto-detect video files if not explicitly specified
    if (!settings.videoMode && !settings.batchMode) {
        std::string extension = std::filesystem::path(settings.inputPath).extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
        if (extension == ".mp4" || extension == ".avi" || extension == ".mov" || 
            extension == ".mkv" || extension == ".wmv") {
            settings.videoMode = true;
        }
    }
    
    // Set video codec based on output format
    if (settings.videoMode) {
        std::string outExt = std::filesystem::path(settings.outputPath).extension().string();
        std::transform(outExt.begin(), outExt.end(), outExt.begin(), ::tolower);
        if (outExt == ".mp4") {
            settings.videoCodec = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        } else if (outExt == ".avi") {
            settings.videoCodec = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
        }
    }
    
    return settings;
}

cv::Mat createBeforeAfterComparison(const cv::Mat& before, const cv::Mat& after) {
    // Resize images to same height for comparison
    int targetHeight = 600;
    cv::Mat beforeResized, afterResized;
    
    double scale = static_cast<double>(targetHeight) / before.rows;
    cv::Size newSize(static_cast<int>(before.cols * scale), targetHeight);
    
    cv::resize(before, beforeResized, newSize);
    cv::resize(after, afterResized, newSize);
    
    // Create side-by-side comparison
    cv::Mat comparison(targetHeight, beforeResized.cols * 2 + 20, beforeResized.type(), cv::Scalar(50, 50, 50));
    
    beforeResized.copyTo(comparison(cv::Rect(0, 0, beforeResized.cols, beforeResized.rows)));
    afterResized.copyTo(comparison(cv::Rect(beforeResized.cols + 20, 0, afterResized.cols, afterResized.rows)));
    
    // Add labels
    cv::putText(comparison, "Before", cv::Point(10, 30), 
               cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
    cv::putText(comparison, "After", cv::Point(beforeResized.cols + 30, 30), 
               cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
    
    return comparison;
}

bool processSingleImage(const AppSettings& settings, const Domain::CalibrationData& calibrationData) {
    std::cout << "Processing single image...\n";
    
    // Load image
    cv::Mat inputImage = cv::imread(settings.inputPath, cv::IMREAD_COLOR);
    if (inputImage.empty()) {
        std::cerr << "Error: Cannot load image " << settings.inputPath << "\n";
        return false;
    }
    
    std::cout << "Input image: " << inputImage.cols << "x" << inputImage.rows << " pixels\n";
    
    // Setup corrector
    Internal::Correction::LinearCorrector::CorrectionSettings correctorSettings;
    correctorSettings.useLinearRGB = settings.useLinearRGB;
    correctorSettings.performQualityCheck = true;
    
    Internal::Correction::LinearCorrector corrector(correctorSettings);
    
    // Apply correction
    std::cout << "Applying color correction...\n";
    auto result = corrector.correctImage(inputImage, calibrationData);
    
    if (!result.success) {
        std::cerr << "Error: Color correction failed\n";
        std::cerr << "Reason: " << result.errorMessage << "\n";
        return false;
    }
    
    std::cout << "Color correction completed successfully!\n";
    std::cout << "  Processing time: " << result.processingTimeMs << "ms\n";
    std::cout << "  Memory usage: " << result.memoryUsageMB << "MB\n";
    std::cout << "  Estimated accuracy: " << result.estimatedAccuracy << "\n";
    
    if (result.wasClampingRequired) {
        std::cout << "  Clamped pixels: " << result.clampedPixels << "\n";
    }
    
    // Show preview if requested
    if (settings.showPreview) {
        cv::Mat comparison = createBeforeAfterComparison(inputImage, result.correctedImage);
        cv::imshow("Before/After Comparison", comparison);
        std::cout << "Press any key to continue...\n";
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    
    // Save result
    std::cout << "Saving corrected image...\n";
    
    std::vector<int> saveParams;
    if (settings.outputFormat == "jpg" || settings.outputFormat == "jpeg") {
        saveParams = {cv::IMWRITE_JPEG_QUALITY, settings.jpegQuality};
    } else if (settings.outputFormat == "png") {
        saveParams = {cv::IMWRITE_PNG_COMPRESSION, 1};
    }
    
    if (!cv::imwrite(settings.outputPath, result.correctedImage, saveParams)) {
        std::cerr << "Error: Failed to save corrected image\n";
        return false;
    }
    
    std::cout << "Corrected image saved: " << settings.outputPath << "\n";
    return true;
}

bool processBatch(const AppSettings& settings, const Domain::CalibrationData& calibrationData) {
    std::cout << "Processing batch...\n";
    
    if (!std::filesystem::is_directory(settings.inputPath)) {
        std::cerr << "Error: Input path is not a directory for batch mode\n";
        return false;
    }
    
    // Setup batch processor
    Internal::Correction::BatchProcessor::BatchSettings batchSettings;
    batchSettings.outputFormat = settings.outputFormat;
    batchSettings.jpegQuality = settings.jpegQuality;
    batchSettings.enableDetailedLogging = settings.verbose;
    batchSettings.continueOnError = true;
    
    Internal::Correction::BatchProcessor processor(batchSettings);
    
    // Setup progress callback
    processor.setProgressCallback([](int current, int total, const std::string& file, const std::string& status) {
        std::cout << "\rProgress: " << current << "/" << total << " - " << 
                    std::filesystem::path(file).filename().string() << " (" << status << ")";
        std::cout.flush();
        if (current == total) {
            std::cout << "\n";
        }
    });
    
    // Setup error callback
    processor.setErrorCallback([settings](const std::string& file, const std::string& error) {
        if (settings.verbose) {
            std::cerr << "\nError processing " << file << ": " << error << "\n";
        }
    });
    
    // Find input files
    std::vector<std::string> inputFiles = processor.findImageFiles(settings.inputPath);
    
    if (inputFiles.empty()) {
        std::cerr << "Error: No image files found in " << settings.inputPath << "\n";
        return false;
    }
    
    std::cout << "Found " << inputFiles.size() << " image files\n";
    
    // Estimate processing time
    float estimatedTime = processor.estimateBatchProcessingTime(inputFiles);
    std::cout << "Estimated processing time: " << std::fixed << std::setprecision(1) 
              << estimatedTime / 1000.0f << " seconds\n\n";
    
    // Process files
    auto batchResult = processor.processFiles(inputFiles, settings.outputPath, calibrationData);
    
    std::cout << "\nBatch processing completed!\n";
    std::cout << batchResult.getSummary() << "\n";
    
    if (batchResult.failedFiles > 0) {
        std::cout << "\nFailed files:\n";
        for (const auto& [file, error] : batchResult.failedFilesWithErrors) {
            std::cout << "  " << std::filesystem::path(file).filename().string() << ": " << error << "\n";
        }
    }
    
    return batchResult.overallSuccess || batchResult.successfulFiles > 0;
}

bool processVideo(const AppSettings& settings, const Domain::CalibrationData& calibrationData) {
    std::cout << "Processing video...\n";
    
    // Open input video
    cv::VideoCapture inputVideo(settings.inputPath);
    if (!inputVideo.isOpened()) {
        std::cerr << "Error: Cannot open input video " << settings.inputPath << "\n";
        return false;
    }
    
    // Get video properties
    int frameWidth = static_cast<int>(inputVideo.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(inputVideo.get(cv::CAP_PROP_FRAME_HEIGHT));
    double inputFPS = inputVideo.get(cv::CAP_PROP_FPS);
    int totalFrames = static_cast<int>(inputVideo.get(cv::CAP_PROP_FRAME_COUNT));
    
    double outputFPS = settings.preserveOriginalFPS ? inputFPS : settings.videoFPS;
    
    std::cout << "Input video: " << frameWidth << "x" << frameHeight << " @ " << inputFPS << " FPS\n";
    std::cout << "Total frames: " << totalFrames << "\n";
    std::cout << "Output FPS: " << outputFPS << "\n";
    
    // Setup output video writer
    cv::VideoWriter outputVideo;
    cv::Size frameSize(frameWidth, frameHeight);
    
    if (!outputVideo.open(settings.outputPath, settings.videoCodec, outputFPS, frameSize, true)) {
        std::cerr << "Error: Cannot create output video " << settings.outputPath << "\n";
        return false;
    }
    
    // Setup corrector
    Internal::Correction::LinearCorrector::CorrectionSettings correctorSettings;
    correctorSettings.useLinearRGB = settings.useLinearRGB;
    correctorSettings.enableErrorRecovery = true;
    correctorSettings.performQualityCheck = false; // Disable for performance in video processing
    
    Internal::Correction::LinearCorrector corrector(correctorSettings);
    
    // Processing loop
    cv::Mat frame, correctedFrame;
    int processedFrames = 0;
    int failedFrames = 0;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    std::cout << "Processing frames...\n";
    
    while (inputVideo.read(frame)) {
        processedFrames++;
        
        // Show progress
        if (processedFrames % 30 == 0 || processedFrames == totalFrames) {
            std::cout << "\rProgress: " << processedFrames << "/" << totalFrames 
                     << " (" << std::fixed << std::setprecision(1) 
                     << (100.0 * processedFrames / totalFrames) << "%)";
            std::cout.flush();
        }
        
        // Apply color correction
        auto result = corrector.correctImage(frame, calibrationData);
        
        if (result.success) {
            outputVideo.write(result.correctedImage);
        } else {
            // Fallback: write original frame on correction failure
            outputVideo.write(frame);
            failedFrames++;
            
            if (settings.verbose) {
                std::cerr << "\nFrame " << processedFrames << " correction failed: " 
                         << result.errorMessage << "\n";
            }
        }
    }
    
    std::cout << "\n";
    
    // Cleanup
    inputVideo.release();
    outputVideo.release();
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    // Calculate statistics
    float processingFPS = (totalTime.count() > 0) ? 
                         (processedFrames * 1000.0f / totalTime.count()) : 0.0f;
    
    std::cout << "Video processing completed!\n";
    std::cout << "  Processed frames: " << processedFrames << "\n";
    std::cout << "  Failed frames: " << failedFrames << "\n";
    std::cout << "  Processing time: " << std::fixed << std::setprecision(2) 
              << totalTime.count() / 1000.0f << " seconds\n";
    std::cout << "  Processing speed: " << std::fixed << std::setprecision(1) 
              << processingFPS << " FPS\n";
    std::cout << "  Success rate: " << std::fixed << std::setprecision(1) 
              << (100.0f * (processedFrames - failedFrames) / processedFrames) << "%\n";
    
    if (failedFrames > 0) {
        std::cout << "  Note: " << failedFrames << " frames were not corrected (original frames used)\n";
    }
    
    std::cout << "Corrected video saved: " << settings.outputPath << "\n";
    
    return failedFrames < processedFrames; // Success if at least some frames were processed
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
        
        std::cout << "Color Correction Application\n";
        std::cout << "============================\n";
        std::cout << "Calibration file: " << settings.calibrationFile << "\n";
        std::cout << "Input: " << settings.inputPath << "\n";
        std::cout << "Output: " << settings.outputPath << "\n";
        std::cout << "Mode: " << (settings.batchMode ? "Batch" : 
                                 (settings.videoMode ? "Video" : "Single image")) << "\n";
        std::cout << "Color space: " << (settings.useLinearRGB ? "Linear RGB" : "sRGB") << "\n";
        std::cout << "Output format: " << settings.outputFormat << "\n\n";
        
        // Load calibration data
        std::cout << "Loading calibration data...\n";
        Domain::CalibrationData calibrationData;
        if (!calibrationData.loadFromFile(settings.calibrationFile, true)) { // Force load
            std::cerr << "Error: Failed to load calibration file " << settings.calibrationFile << "\n";
            return 1;
        }
        
        if (!calibrationData.isValid()) {
            std::cout << "Warning: Calibration data has poor quality metrics but will be used anyway\n";
            std::cout << calibrationData.getSummary() << "\n";
        }
        
        std::cout << "Calibration data loaded successfully\n";
        if (settings.verbose) {
            std::cout << calibrationData.getSummary() << "\n";
        }
        
        // Process based on mode
        bool success;
        if (settings.batchMode) {
            success = processBatch(settings, calibrationData);
        } else if (settings.videoMode) {
            success = processVideo(settings, calibrationData);
        } else {
            success = processSingleImage(settings, calibrationData);
        }
        
        if (success) {
            std::cout << "\n✓ Color correction completed successfully!\n";
            return 0;
        } else {
            std::cout << "\n✗ Color correction failed\n";
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}