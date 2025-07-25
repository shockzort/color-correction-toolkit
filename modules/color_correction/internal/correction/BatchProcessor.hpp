#pragma once

#include "LinearCorrector.hpp"
#include "../domain/CalibrationData.hpp"
#include <shared/types/Common.hpp>
#include <shared/utils/Logger.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <functional>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <filesystem>

namespace ColorCorrection::Internal::Correction {

class BatchProcessor {
  public:
    // Progress callback: (current_item, total_items, current_file, status_message)
    using ProgressCallback = std::function<void(int, int, const std::string&, const std::string&)>;
    
    // Error callback: (filename, error_message)
    using ErrorCallback = std::function<void(const std::string&, const std::string&)>;

    struct BatchSettings {
        // Processing settings
        bool enableParallelProcessing = true;
        int numThreads = 0;                     // 0 = auto-detect
        int maxConcurrentFiles = 4;             // Limit memory usage
        
        // Error handling
        bool continueOnError = true;            // Continue processing if individual files fail
        bool enableErrorRecovery = true;       // Use fallback correction on errors
        
        // Output settings
        std::string outputFormat = "png";      // Default output format
        bool preserveOriginalFormat = false;   // Keep same format as input
        bool createOutputDirectory = true;     // Create output dir if it doesn't exist
        
        // Quality settings
        int jpegQuality = 95;                  // JPEG quality (1-100)
        int pngCompression = 1;                // PNG compression (0-9)
        bool enableOutputValidation = true;    // Validate output files
        
        // Progress reporting
        int progressUpdateInterval = 1;        // Update progress every N files
        bool enableDetailedLogging = false;    // Log processing details for each file
        
        // Memory management
        bool enableMemoryOptimization = true;  // Optimize memory usage for large batches
        size_t maxMemoryUsageMB = 2048;        // Maximum memory usage limit
        
        // File filtering
        std::vector<std::string> supportedExtensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"};
        bool recursiveDirectorySearch = false; // Search subdirectories
        
        // Performance tuning
        bool preloadCalibration = true;        // Preload calibration data
        bool enableIOOptimization = true;      // Optimize file I/O
    };

    struct BatchResult {
        bool overallSuccess = false;
        
        // Statistics
        int totalFiles = 0;
        int processedFiles = 0;
        int successfulFiles = 0;
        int failedFiles = 0;
        int skippedFiles = 0;
        
        // Performance metrics
        float totalProcessingTimeMs = 0.0f;
        float averageProcessingTimeMs = 0.0f;
        float totalIOTimeMs = 0.0f;
        float throughputFilesPerSecond = 0.0f;
        
        // Memory usage
        float peakMemoryUsageMB = 0.0f;
        float averageMemoryUsageMB = 0.0f;
        
        // Error information
        std::vector<std::pair<std::string, std::string>> failedFilesWithErrors;
        std::vector<std::string> skippedFilesWithReasons;
        
        // Output information
        std::vector<std::string> successfulOutputFiles;
        std::string outputDirectory;
        
        std::string getSummary() const {
            std::ostringstream oss;
            oss << "Batch Processing Summary:\n";
            oss << "  Total files: " << totalFiles << "\n";
            oss << "  Successful: " << successfulFiles << "\n";
            oss << "  Failed: " << failedFiles << "\n";
            oss << "  Skipped: " << skippedFiles << "\n";
            oss << "  Processing time: " << std::fixed << std::setprecision(2) 
                << totalProcessingTimeMs / 1000.0f << "s\n";
            oss << "  Throughput: " << throughputFilesPerSecond << " files/sec\n";
            oss << "  Peak memory: " << peakMemoryUsageMB << " MB\n";
            oss << "  Success rate: " << std::fixed << std::setprecision(1) 
                << (totalFiles > 0 ? (successfulFiles * 100.0f / totalFiles) : 0.0f) << "%";
            return oss.str();
        }
    };

    BatchProcessor(const BatchSettings& settings = BatchSettings{})
        : settings_(settings), corrector_(createCorrectorSettings(settings)), 
          isProcessing_(false), shouldStop_(false) {
        
        // Initialize thread count
        if (settings_.numThreads <= 0) {
            numThreads_ = std::max(1u, std::thread::hardware_concurrency());
        } else {
            numThreads_ = static_cast<unsigned int>(settings_.numThreads);
        }
        
        LOG_INFO("Batch processor initialized with ", numThreads_, " threads");
    }

    // Process multiple image files
    BatchResult processFiles(const std::vector<std::string>& inputFiles,
                           const std::string& outputDirectory,
                           const Domain::CalibrationData& calibrationData) {
        return processFiles(inputFiles, outputDirectory, calibrationData.getCorrectionMatrix());
    }

    BatchResult processFiles(const std::vector<std::string>& inputFiles,
                           const std::string& outputDirectory,
                           const Domain::CorrectionMatrix& correctionMatrix) {
        BatchResult result;
        result.outputDirectory = outputDirectory;
        result.totalFiles = static_cast<int>(inputFiles.size());

        if (inputFiles.empty()) {
            LOG_WARN("No input files provided for batch processing");
            return result;
        }

        if (!correctionMatrix.isValid()) {
            LOG_ERROR("Invalid correction matrix provided for batch processing");
            return result;
        }

        auto startTime = std::chrono::high_resolution_clock::now();
        isProcessing_ = true;
        shouldStop_ = false;

        try {
            LOG_INFO("Starting batch processing of ", inputFiles.size(), " files");

            // Create output directory if needed
            if (settings_.createOutputDirectory) {
                createOutputDirectory(outputDirectory);
            }

            // Filter and validate input files
            std::vector<std::string> validFiles = filterValidFiles(inputFiles, result);
            
            if (validFiles.empty()) {
                LOG_WARN("No valid files found for processing");
                isProcessing_ = false;
                return result;
            }

            // Process files
            if (settings_.enableParallelProcessing && validFiles.size() > 1) {
                result = processFilesParallel(validFiles, outputDirectory, correctionMatrix);
            } else {
                result = processFilesSequential(validFiles, outputDirectory, correctionMatrix);
            }

            // Calculate final statistics
            auto endTime = std::chrono::high_resolution_clock::now();
            auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
            
            result.totalProcessingTimeMs = static_cast<float>(totalDuration.count());
            result.averageProcessingTimeMs = result.processedFiles > 0 ? 
                result.totalProcessingTimeMs / result.processedFiles : 0.0f;
            
            float totalTimeSeconds = result.totalProcessingTimeMs / 1000.0f;
            result.throughputFilesPerSecond = totalTimeSeconds > 0.0f ? 
                result.processedFiles / totalTimeSeconds : 0.0f;

            result.overallSuccess = (result.failedFiles == 0);

            LOG_INFO("Batch processing completed. ", result.getSummary());

        } catch (const std::exception& e) {
            LOG_ERROR("Exception during batch processing: ", e.what());
            result.failedFilesWithErrors.emplace_back("batch_processing", e.what());
        }

        isProcessing_ = false;
        return result;
    }

    // Process directory of images
    BatchResult processDirectory(const std::string& inputDirectory,
                               const std::string& outputDirectory,
                               const Domain::CorrectionMatrix& correctionMatrix) {
        std::vector<std::string> imageFiles = findImageFiles(inputDirectory, settings_.recursiveDirectorySearch);
        return processFiles(imageFiles, outputDirectory, correctionMatrix);
    }

    // Process images in memory (no file I/O)
    struct ImageBatch {
        std::vector<Types::Image> images;
        std::vector<std::string> names;  // Optional names for tracking
    };

    std::vector<Types::Image> processImageBatch(const ImageBatch& batch,
                                               const Domain::CorrectionMatrix& correctionMatrix,
                                               std::vector<bool>& successFlags) {
        std::vector<Types::Image> correctedImages;
        successFlags.clear();
        
        correctedImages.reserve(batch.images.size());
        successFlags.reserve(batch.images.size());

        LOG_INFO("Processing batch of ", batch.images.size(), " images in memory");

        for (size_t i = 0; i < batch.images.size(); ++i) {
            try {
                LinearCorrector::CorrectionResult result = corrector_.correctImage(batch.images[i], correctionMatrix);
                
                if (result.success) {
                    correctedImages.push_back(result.correctedImage);
                    successFlags.push_back(true);
                } else {
                    correctedImages.push_back(batch.images[i]); // Fallback to original
                    successFlags.push_back(false);
                    
                    std::string name = (i < batch.names.size()) ? batch.names[i] : ("image_" + std::to_string(i));
                    LOG_WARN("Failed to process image ", name, ": ", result.errorMessage);
                }
            } catch (const std::exception& e) {
                correctedImages.push_back(batch.images[i]); // Fallback to original
                successFlags.push_back(false);
                LOG_ERROR("Exception processing image ", i, ": ", e.what());
            }
        }

        return correctedImages;
    }

    // Control methods
    void stopProcessing() {
        shouldStop_ = true;
        LOG_INFO("Batch processing stop requested");
    }

    bool isProcessing() const {
        return isProcessing_;
    }

    // Callback registration
    void setProgressCallback(ProgressCallback callback) {
        progressCallback_ = callback;
    }

    void setErrorCallback(ErrorCallback callback) {
        errorCallback_ = callback;
    }

    // Settings management
    void setSettings(const BatchSettings& settings) {
        settings_ = settings;
        corrector_.setSettings(createCorrectorSettings(settings));
    }

    BatchSettings getSettings() const {
        return settings_;
    }

    // Utility methods
    std::vector<std::string> findImageFiles(const std::string& directory, 
                                          bool recursive = false) const {
        std::vector<std::string> imageFiles;
        
        try {
            if (!std::filesystem::exists(directory)) {
                LOG_ERROR("Directory does not exist: ", directory);
                return imageFiles;
            }

            auto searchOptions = recursive ? 
                std::filesystem::directory_options::follow_directory_symlink :
                std::filesystem::directory_options::none;

            for (const auto& entry : std::filesystem::recursive_directory_iterator(directory, searchOptions)) {
                if (entry.is_regular_file()) {
                    std::string extension = entry.path().extension().string();
                    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
                    
                    if (std::find(settings_.supportedExtensions.begin(), 
                                settings_.supportedExtensions.end(), extension) != 
                        settings_.supportedExtensions.end()) {
                        imageFiles.push_back(entry.path().string());
                    }
                }
            }

            LOG_INFO("Found ", imageFiles.size(), " image files in ", directory);
            
        } catch (const std::filesystem::filesystem_error& e) {
            LOG_ERROR("Filesystem error while searching directory: ", e.what());
        }

        return imageFiles;
    }

    float estimateBatchProcessingTime(const std::vector<std::string>& inputFiles) const {
        if (inputFiles.empty()) return 0.0f;

        // Sample a few files to estimate average processing time
        int sampleSize = std::min(3, static_cast<int>(inputFiles.size()));
        float totalEstimatedTime = 0.0f;

        for (int i = 0; i < sampleSize; ++i) {
            try {
                Types::Image sampleImage = cv::imread(inputFiles[i]);
                if (!sampleImage.empty()) {
                    float estimatedTime = corrector_.estimateProcessingTime(sampleImage);
                    totalEstimatedTime += estimatedTime;
                }
            } catch (...) {
                // Ignore errors in estimation
            }
        }

        float averageTime = sampleSize > 0 ? totalEstimatedTime / sampleSize : 50.0f; // 50ms default
        float totalTime = averageTime * inputFiles.size();

        // Adjust for parallel processing
        if (settings_.enableParallelProcessing) {
            totalTime /= std::min(numThreads_, static_cast<unsigned int>(inputFiles.size()));
        }

        return totalTime;
    }

  private:
    BatchSettings settings_;
    LinearCorrector corrector_;
    unsigned int numThreads_;
    
    std::atomic<bool> isProcessing_;
    std::atomic<bool> shouldStop_;
    
    ProgressCallback progressCallback_;
    ErrorCallback errorCallback_;
    
    // Thread synchronization
    mutable std::mutex progressMutex_;
    mutable std::mutex errorMutex_;

    LinearCorrector::CorrectionSettings createCorrectorSettings(const BatchSettings& batchSettings) {
        LinearCorrector::CorrectionSettings correctorSettings;
        correctorSettings.enableParallelProcessing = batchSettings.enableParallelProcessing;
        correctorSettings.numThreads = batchSettings.numThreads;
        correctorSettings.enableErrorRecovery = batchSettings.enableErrorRecovery;
        correctorSettings.maintainInputType = true;
        correctorSettings.clampValues = true;
        return correctorSettings;
    }

    void createOutputDirectory(const std::string& outputDirectory) {
        try {
            if (!std::filesystem::exists(outputDirectory)) {
                std::filesystem::create_directories(outputDirectory);
                LOG_INFO("Created output directory: ", outputDirectory);
            }
        } catch (const std::filesystem::filesystem_error& e) {
            LOG_ERROR("Failed to create output directory: ", e.what());
            throw;
        }
    }

    std::vector<std::string> filterValidFiles(const std::vector<std::string>& inputFiles, 
                                            BatchResult& result) {
        std::vector<std::string> validFiles;
        
        for (const auto& file : inputFiles) {
            if (!std::filesystem::exists(file)) {
                result.skippedFilesWithReasons.emplace_back(file + " (file not found)");
                result.skippedFiles++;
                continue;
            }

            std::string extension = std::filesystem::path(file).extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
            
            if (std::find(settings_.supportedExtensions.begin(), settings_.supportedExtensions.end(), 
                         extension) == settings_.supportedExtensions.end()) {
                result.skippedFilesWithReasons.emplace_back(file + " (unsupported format)");
                result.skippedFiles++;
                continue;
            }

            validFiles.push_back(file);
        }

        LOG_INFO("Filtered ", validFiles.size(), " valid files from ", inputFiles.size(), " inputs");
        return validFiles;
    }

    BatchResult processFilesSequential(const std::vector<std::string>& files,
                                     const std::string& outputDirectory,
                                     const Domain::CorrectionMatrix& correctionMatrix) {
        BatchResult result;
        result.outputDirectory = outputDirectory;
        result.totalFiles = static_cast<int>(files.size());

        for (size_t i = 0; i < files.size() && !shouldStop_; ++i) {
            const std::string& inputFile = files[i];
            
            updateProgress(static_cast<int>(i + 1), result.totalFiles, inputFile, "Processing...");
            
            bool success = processSingleFile(inputFile, outputDirectory, correctionMatrix, result);
            
            result.processedFiles++;
            if (success) {
                result.successfulFiles++;
            } else {
                result.failedFiles++;
                if (!settings_.continueOnError) {
                    LOG_ERROR("Stopping batch processing due to error");
                    break;
                }
            }
        }

        return result;
    }

    BatchResult processFilesParallel(const std::vector<std::string>& files,
                                   const std::string& outputDirectory,
                                   const Domain::CorrectionMatrix& correctionMatrix) {
        BatchResult result;
        result.outputDirectory = outputDirectory;
        result.totalFiles = static_cast<int>(files.size());

        // Thread-safe counters
        std::atomic<int> processedCount(0);
        std::atomic<int> successCount(0);
        std::atomic<int> failureCount(0);
        
        // Thread pool processing
        const int numWorkers = std::min(numThreads_, static_cast<unsigned int>(files.size()));
        std::vector<std::thread> workers;
        
        // Work queue
        std::atomic<size_t> currentFileIndex(0);
        
        LOG_INFO("Starting parallel processing with ", numWorkers, " workers");

        for (int i = 0; i < numWorkers; ++i) {
            workers.emplace_back([&]() {
                while (!shouldStop_) {
                    size_t fileIndex = currentFileIndex.fetch_add(1);
                    if (fileIndex >= files.size()) {
                        break;
                    }

                    const std::string& inputFile = files[fileIndex];
                    
                    updateProgress(static_cast<int>(fileIndex + 1), result.totalFiles, 
                                 inputFile, "Processing...");
                    
                    bool success = processSingleFile(inputFile, outputDirectory, correctionMatrix, result);
                    
                    processedCount++;
                    if (success) {
                        successCount++;
                    } else {
                        failureCount++;
                        if (!settings_.continueOnError) {
                            shouldStop_ = true;
                        }
                    }
                }
            });
        }

        // Wait for all workers to complete
        for (auto& worker : workers) {
            worker.join();
        }

        result.processedFiles = processedCount.load();
        result.successfulFiles = successCount.load();
        result.failedFiles = failureCount.load();

        return result;
    }

    bool processSingleFile(const std::string& inputFile,
                          const std::string& outputDirectory,
                          const Domain::CorrectionMatrix& correctionMatrix,
                          BatchResult& result) {
        try {
            auto ioStartTime = std::chrono::high_resolution_clock::now();
            
            // Load image
            Types::Image image = cv::imread(inputFile, cv::IMREAD_UNCHANGED);
            if (image.empty()) {
                std::string error = "Failed to load image";
                reportError(inputFile, error);
                
                std::lock_guard<std::mutex> lock(errorMutex_);
                result.failedFilesWithErrors.emplace_back(inputFile, error);
                return false;
            }

            auto ioEndTime = std::chrono::high_resolution_clock::now();
            
            // Process image
            LinearCorrector::CorrectionResult correctionResult = corrector_.correctImage(image, correctionMatrix);
            
            if (!correctionResult.success) {
                reportError(inputFile, correctionResult.errorMessage);
                
                std::lock_guard<std::mutex> lock(errorMutex_);
                result.failedFilesWithErrors.emplace_back(inputFile, correctionResult.errorMessage);
                return false;
            }

            // Generate output filename
            std::filesystem::path inputPath(inputFile);
            std::string outputFilename = inputPath.stem().string() + "_corrected";
            
            std::string outputExtension;
            if (settings_.preserveOriginalFormat) {
                outputExtension = inputPath.extension().string();
            } else {
                outputExtension = "." + settings_.outputFormat;
            }
            
            std::filesystem::path outputPath = std::filesystem::path(outputDirectory) / 
                                             (outputFilename + outputExtension);
            
            // Save corrected image
            auto saveStartTime = std::chrono::high_resolution_clock::now();
            
            bool saveSuccess = saveImage(correctionResult.correctedImage, outputPath.string());
            
            auto saveEndTime = std::chrono::high_resolution_clock::now();
            
            if (!saveSuccess) {
                std::string error = "Failed to save corrected image";
                reportError(inputFile, error);
                
                std::lock_guard<std::mutex> lock(errorMutex_);
                result.failedFilesWithErrors.emplace_back(inputFile, error);
                return false;
            }

            // Update I/O timing
            auto ioTime = std::chrono::duration_cast<std::chrono::milliseconds>(
                (ioEndTime - ioStartTime) + (saveEndTime - saveStartTime));
            
            {
                std::lock_guard<std::mutex> lock(progressMutex_);
                result.totalIOTimeMs += static_cast<float>(ioTime.count());
                result.successfulOutputFiles.push_back(outputPath.string());
            }

            if (settings_.enableDetailedLogging) {
                LOG_DEBUG("Successfully processed ", inputFile, " -> ", outputPath.string(), 
                         " (", correctionResult.processingTimeMs, "ms)");
            }

            return true;

        } catch (const std::exception& e) {
            std::string error = "Exception: " + std::string(e.what());
            reportError(inputFile, error);
            
            std::lock_guard<std::mutex> lock(errorMutex_);
            result.failedFilesWithErrors.emplace_back(inputFile, error);
            return false;
        }
    }

    bool saveImage(const Types::Image& image, const std::string& outputPath) {
        try {
            std::vector<int> saveParams;
            
            std::string extension = std::filesystem::path(outputPath).extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
            
            if (extension == ".jpg" || extension == ".jpeg") {
                saveParams = {cv::IMWRITE_JPEG_QUALITY, settings_.jpegQuality};
            } else if (extension == ".png") {
                saveParams = {cv::IMWRITE_PNG_COMPRESSION, settings_.pngCompression};
            }
            
            bool success = cv::imwrite(outputPath, image, saveParams);
            
            // Validate output if enabled
            if (success && settings_.enableOutputValidation) {
                Types::Image testLoad = cv::imread(outputPath, cv::IMREAD_UNCHANGED);
                if (testLoad.empty()) {
                    LOG_ERROR("Output validation failed for ", outputPath);
                    return false;
                }
            }
            
            return success;
            
        } catch (const cv::Exception& e) {
            LOG_ERROR("OpenCV error saving image ", outputPath, ": ", e.what());
            return false;
        }
    }

    void updateProgress(int current, int total, const std::string& currentFile, 
                       const std::string& status) {
        if (progressCallback_ && (current % settings_.progressUpdateInterval == 0 || current == total)) {
            progressCallback_(current, total, currentFile, status);
        }
    }

    void reportError(const std::string& filename, const std::string& error) {
        if (errorCallback_) {
            errorCallback_(filename, error);
        }
        
        if (settings_.enableDetailedLogging) {
            LOG_ERROR("Processing error for ", filename, ": ", error);
        }
    }
};

}  // namespace ColorCorrection::Internal::Correction