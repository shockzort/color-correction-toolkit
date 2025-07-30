#pragma once

#include "LinearCorrector.hpp"
#include "../domain/CalibrationData.hpp"
#include <shared/types/Common.hpp>
#include <vector>
#include <string>
#include <functional>
#include <thread>
#include <mutex>
#include <atomic>

namespace ColorCorrection::Internal::Correction {

class BatchProcessor {
  public:
    // Progress callback: (current_item, total_items, current_file, status_message)
    using ProgressCallback = std::function<void(int, int, const std::string&, const std::string&)>;
    
    // Error callback: (filename, error_message)
    using ErrorCallback = std::function<void(const std::string&, const std::string&)>;

    struct BatchSettings {
        // Processing settings
        bool enableParallelProcessing;
        int numThreads;                     // 0 = auto-detect
        int maxConcurrentFiles;             // Limit memory usage
        
        // Error handling
        bool continueOnError;            // Continue processing if individual files fail
        bool enableErrorRecovery;       // Use fallback correction on errors
        
        // Output settings
        std::string outputFormat;      // Default output format
        bool preserveOriginalFormat;   // Keep same format as input
        bool createOutputDirectory;     // Create output dir if it doesn't exist
        
        // Quality settings
        int jpegQuality;                  // JPEG quality (1-100)
        int pngCompression;                // PNG compression (0-9)
        bool enableOutputValidation;    // Validate output files
        
        // Progress reporting
        int progressUpdateInterval;        // Update progress every N files
        bool enableDetailedLogging;    // Log processing details for each file
        
        // Memory management
        bool enableMemoryOptimization;  // Optimize memory usage for large batches
        size_t maxMemoryUsageMB;        // Maximum memory usage limit
        
        // File filtering
        std::vector<std::string> supportedExtensions;
        bool recursiveDirectorySearch; // Search subdirectories
        
        // Performance tuning
        bool preloadCalibration;        // Preload calibration data
        bool enableIOOptimization;      // Optimize file I/O
        
        BatchSettings();
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
        
        std::string getSummary() const;
    };

    // Process images in memory (no file I/O)
    struct ImageBatch {
        std::vector<Types::Image> images;
        std::vector<std::string> names;  // Optional names for tracking
    };

    explicit BatchProcessor(const BatchSettings& settings = BatchSettings{});

    // Process multiple image files
    BatchResult processFiles(const std::vector<std::string>& inputFiles,
                           const std::string& outputDirectory,
                           const Domain::CalibrationData& calibrationData);

    BatchResult processFiles(const std::vector<std::string>& inputFiles,
                           const std::string& outputDirectory,
                           const Domain::CorrectionMatrix& correctionMatrix);

    // Process directory of images
    BatchResult processDirectory(const std::string& inputDirectory,
                               const std::string& outputDirectory,
                               const Domain::CorrectionMatrix& correctionMatrix);

    std::vector<Types::Image> processImageBatch(const ImageBatch& batch,
                                               const Domain::CorrectionMatrix& correctionMatrix,
                                               std::vector<bool>& successFlags);

    // Control methods
    void stopProcessing();
    bool isProcessing() const;

    // Callback registration
    void setProgressCallback(ProgressCallback callback);
    void setErrorCallback(ErrorCallback callback);

    // Settings management
    void setSettings(const BatchSettings& settings);
    BatchSettings getSettings() const;

    // Utility methods
    std::vector<std::string> findImageFiles(const std::string& directory, 
                                          bool recursive = false) const;

    float estimateBatchProcessingTime(const std::vector<std::string>& inputFiles) const;

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

    LinearCorrector::CorrectionSettings createCorrectorSettings(const BatchSettings& batchSettings);
    void createOutputDirectory(const std::string& outputDirectory);
    std::vector<std::string> filterValidFiles(const std::vector<std::string>& inputFiles, 
                                            BatchResult& result);
    BatchResult processFilesSequential(const std::vector<std::string>& files,
                                     const std::string& outputDirectory,
                                     const Domain::CorrectionMatrix& correctionMatrix);
    BatchResult processFilesParallel(const std::vector<std::string>& files,
                                   const std::string& outputDirectory,
                                   const Domain::CorrectionMatrix& correctionMatrix);
    bool processSingleFile(const std::string& inputFile,
                          const std::string& outputDirectory,
                          const Domain::CorrectionMatrix& correctionMatrix,
                          BatchResult& result);
    bool saveImage(const Types::Image& image, const std::string& outputPath);
    void updateProgress(int current, int total, const std::string& currentFile, 
                       const std::string& status);
    void reportError(const std::string& filename, const std::string& error);
};

}  // namespace ColorCorrection::Internal::Correction