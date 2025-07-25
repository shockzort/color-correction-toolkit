# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Color Correction System** project that implements color correction algorithms using OpenCV and the ColorChecker Classic color chart. The project is documented in Russian and focuses on computer vision color calibration techniques.

### Technology Stack

- **Programming Language**: C++ 17
- **Primary Library**: OpenCV (with contrib modules)
- **Color Chart**: X-Rite ColorChecker Classic (24 patches)
- **Target Color Space**: sRGB
- **Architecture**: Modular C++ classes for calibration and correction

## Project Structure

Currently, the project contains detailed documentation and algorithm specifications:

```
├── docs/                 # Documentation
│   ├── brd/              # Business requirements
│   ├── conventions.md    # Code conventions and DDD patterns
│   ├── testing.md        # Testing guidelines
│   ├── git.md            # Git workflow
│   └── logging.md        # Logging guidelines
├── modules/              # Modules
│   ├── shared/           # General types and utilities
│   │   ├── types/        # Common types
│   │   ├── utils/        # Utility functions
│   ├── {module_name}/    # Module code
│   │   ├── interface/    # Module interface
│   │   ├── internal/     # Module implementation
│   │   ├── config/       # Module configuration
│   |   |── changelog.d   # Chanelogs
│   │   └── VERSION       # Module version
│   ├── tools/            # Development tools
├── Variant_*.md          # Different implementation approaches
└── CLAUDE.md             # This file
```

## Unified Implementation Approach

Based on analysis of 4 implementation variants, the optimal solution combines the best practices:

### Core Architecture (DDD-based)

Following Domain-Driven Design principles with clear layer separation:

```
modules/color_correction/
├── interface/              # Public API
│   ├── CalibrationManager  # Calibration interface
│   ├── ColorCorrector      # Correction interface
│   └── Configuration       # Runtime configuration
├── internal/               # Implementation details
│   ├── detection/          # Multi-level ColorChecker detection
│   ├── processing/         # Color analysis and matrix computation
│   ├── correction/         # Image correction pipeline
│   └── optimization/       # GPU/LUT optimizations
└── config/                 # Default configurations
```

### Detection Strategy (Multi-level Fallback)

1. **Primary**: OpenCV MCC module (`cv::mcc::CCheckerDetector`)
2. **Secondary**: Contour-based geometric detection
3. **Tertiary**: Template matching with perspective correction
4. **Confidence Scoring**: Each method returns confidence level for automatic fallback

### Color Processing Pipeline (Linear RGB)

1. **Preprocessing**: CLAHE enhancement, noise reduction
2. **Detection**: Multi-level ColorChecker detection with perspective correction
3. **Extraction**: Statistical color analysis (robust mean, outlier filtering)
4. **Linearization**: Proper sRGB to linear RGB conversion (gamma 2.4)
5. **Matrix Computation**: SVD-based least squares with regularization
6. **Validation**: Delta E metrics, cross-validation, numerical stability checks

### Performance Optimizations

- **GPU Acceleration**: OpenCL/CUDA for parallel processing
- **LUT Tables**: Pre-computed lookup tables for real-time video
- **Multi-threading**: Parallel processing for batch operations
- **Memory Management**: Efficient matrix operations and memory reuse

### Error Handling & Robustness

- **Comprehensive Validation**: Input validation, numerical stability checks
- **Graceful Degradation**: Automatic fallback between detection methods
- **Quality Metrics**: Delta E scoring, confidence thresholds
- **Error Recovery**: Robust against varying lighting and perspective conditions

## Development Commands

**Note**: The project currently lacks implementation files and build system. Based on documentation, the intended commands would be:

```bash
# Build the project (when implemented)
make build

# Run unit tests (when implemented)
make test-unit

# Run all tests
make test

# Run processing test data
make test-processing
```

## Core Algorithm Components

### 1. Calibration Phase

- **ColorChecker Detection**: Automatic detection of 24-patch color chart
- **Color Extraction**: Extract average colors from each patch
- **Matrix Calculation**: Compute 3x3 color correction matrix using least squares
- **Serialization**: Save calibration matrix to file (YAML/XML format)

### 2. Correction Phase

- **Matrix Loading**: Load pre-computed correction matrix
- **Color Transformation**: Apply matrix to input images
- **Optimization**: GPU acceleration and LUT-based processing for video

### Key Classes (Unified Design)

**Interface Layer:**

- `ICalibrationManager`: Calibration workflow interface
- `IColorCorrector`: Color correction interface
- `IConfiguration`: Runtime configuration interface

**Internal Layer:**

- `MultiLevelDetector`: Multi-fallback ColorChecker detection
- `LinearColorProcessor`: Linear RGB processing with gamma correction
- `CorrectionMatrixCalculator`: SVD-based matrix computation with validation
- `GPUAccelerator`: OpenCL/CUDA optimization layer
- `LUTOptimizer`: Lookup table generation and management

**Domain Objects:**

- `ColorPatch`: Individual patch representation with confidence
- `CorrectionMatrix`: 3x3 matrix with validation and serialization
- `DetectionResult`: Detection outcome with confidence scoring
- `CalibrationData`: Complete calibration parameters and metadata

## Workflow Requirements

**MANDATORY** workflow based on project conventions:

### Task Planning Rules

1. **Task ID Required**: Request unique `<task_id>` from user
2. **Russian Documentation**: Create detailed plan in `docs/tasks/<task_id>/`
3. **Sequential Execution**: Step-by-step implementation with user approval
4. **Commit Format**: `[<task_id>] description`
5. **Changelog**: Create `changelog.d/<task_id>.md` when complete

### Development Process

1. Requirements analysis (читать docs/brd/brd.md)
2. Plan creation in Russian
3. Design following conventions.md
4. Implementation (модульная архитектура)
5. Testing (см. docs/testing.md)
6. Code formatting (clang-format)
7. Documentation updates

## Code Quality Standards

Based on `docs/conventions.md`:

- **Naming Conventions**: PascalCase for classes, camelCase for variables
- **Architecture**: Follow DDD principles with clear layer separation
- **Comments**: Only where necessary, avoid code duplication
- **Error Handling**: Comprehensive error checking and logging
- **Testing**: Both unit and integration tests required
- **Formatting**: Use clang-format for consistent style

## Technical Implementation Requirements

Based on unified approach analysis:

### Core Requirements

1. **ColorChecker Classic**: 24-patch color chart (6×4 grid) with multi-level detection
2. **Linear RGB Processing**: Proper gamma correction (2.4) for mathematical accuracy
3. **sRGB Target**: Final output in sRGB color space with proper gamma encoding
4. **Multi-level Fallback**: Automatic fallback between detection methods
5. **Statistical Robustness**: Outlier filtering, robust mean calculation
6. **Numerical Stability**: SVD-based matrix computation with regularization

### Performance Requirements

1. **GPU Acceleration**: OpenCL/CUDA support for real-time video processing
2. **LUT Optimization**: Pre-computed lookup tables for batch processing
3. **Memory Efficiency**: Optimized matrix operations and memory reuse
4. **Multi-threading**: Parallel processing capabilities
5. **Real-time Processing**: Target 200+ FPS for HD video on modern hardware

### Integration Requirements

1. **Modular Design**: Clean interfaces following DDD principles
2. **Runtime Configuration**: Dynamic parameter adjustment
3. **Dual Mode Operation**: Separate calibration and correction workflows
4. **Error Recovery**: Graceful handling of detection failures
5. **Quality Metrics**: Delta E validation and confidence scoring

## Testing Strategy

Per `docs/testing.md`:

- **Unit Tests**: Black-box testing of public APIs (`*_test.cpp`)
- **Integration Tests**: With external dependencies (`Test_Integration_*`)
- **White-box Tests**: Internal implementation testing (`*_internal_test.cpp`)
- **Parallel Execution**: Tests should support parallel execution when possible

## Related Documentation

- `docs/conventions.md`: Coding standards and architectural patterns
- `docs/testing.md`: Testing methodologies and examples
- `docs/brd/brd.md`: Business requirements and domain model
- `docs/git.md`: Git workflow and practices
- `Variant_*.md`: Different implementation approaches with complete code examples
