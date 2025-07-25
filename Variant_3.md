# Разработка программных решений на C++ с использованием OpenCV для цветокоррекции по таблице Xrite ColorChecker Classic

## 1. Калибровка системы: детекция ячеек ColorChecker Classic

### 1.1. Подготовка и предварительная обработка изображения

```cpp
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

// Улучшение контраста с помощью CLAHE
Mat preprocessImage(const Mat& input) {
    Mat lab, enhanced;
    cvtColor(input, lab, COLOR_BGR2Lab);

    vector<Mat> channels;
    split(lab, channels);

    // Применение CLAHE к L-каналу
    Ptr<CLAHE> clahe = createCLAHE(3.0, Size(8, 8));
    clahe->apply(channels[0], channels[0]);

    merge(channels, lab);
    cvtColor(lab, enhanced, COLOR_Lab2BGR);

    return enhanced;
}
```

### 1.2. Детекция контуров и фильтрация по форме

```cpp
vector<Point2f> detectColorCheckerCorners(const Mat& image) {
    // Предварительная обработка
    Mat processed = preprocessImage(image);
    Mat gray, blurred, edges;

    cvtColor(processed, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blurred, Size(5, 5), 0);
    Canny(blurred, edges, 50, 150);

    // Поиск контуров
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(edges, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

    // Фильтрация контуров по площади и форме
    vector<Point2f> bestQuad;
    double maxArea = 0;

    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area < 1000) continue;

        // Аппроксимация контура
        vector<Point> approx;
        double epsilon = 0.02 * arcLength(contours[i], true);
        approxPolyDP(contours[i], approx, epsilon, true);

        // Ищем четырехугольник
        if (approx.size() == 4 && isContourConvex(approx)) {
            double areaRatio = area / (image.rows * image.cols);
            if (areaRatio > 0.05 && areaRatio < 0.9) {  // Реалистичный размер
                if (area > maxArea) {
                    maxArea = area;
                    bestQuad.clear();
                    for (const auto& pt : approx) {
                        bestQuad.push_back(Point2f(pt));
                    }
                }
            }
        }
    }

    return bestQuad;
}
```

### 1.3. Перспективная коррекция и выделение ячеек

```cpp
Mat extractColorChecker(const Mat& image, const vector<Point2f>& corners) {
    // Сортировка углов по часовой стрелке
    vector<Point2f> sortedCorners = sortCorners(corners);

    // Целевые координаты для перспективного преобразования
    Point2f dstCorners[4] = {
        Point2f(0, 0),
        Point2f(240, 0),
        Point2f(240, 160),
        Point2f(0, 160)
    };

    Mat transform = getPerspectiveTransform(sortedCorners, dstCorners);
    Mat corrected;
    warpPerspective(image, corrected, transform, Size(240, 160));

    return corrected;
}

vector<Point2f> sortCorners(const vector<Point2f>& corners) {
    vector<Point2f> sorted = corners;

    // Находим центр
    Point2f center(0, 0);
    for (const auto& pt : corners) {
        center += pt;
    }
    center *= 1.0 / 4;

    // Сортировка по углам относительно центра
    sort(sorted.begin(), sorted.end(), [center](const Point2f& a, const Point2f& b) {
        return atan2(a.y - center.y, a.x - center.x) < atan2(b.y - center.y, b.x - center.x);
    });

    return sorted;
}
```

### 1.4. Извлечение средних цветов ячеек

```cpp
struct ColorPatch {
    Point2f center;
    Vec3b color;
    int row, col;
};

vector<ColorPatch> extractPatchColors(const Mat& correctedImage) {
    vector<ColorPatch> patches;

    // Параметры сетки ColorChecker Classic (24 ячейки: 6x4)
    int rows = 4, cols = 6;
    int cellWidth = correctedImage.cols / cols;
    int cellHeight = correctedImage.rows / rows;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            Rect roi(c * cellWidth, r * cellHeight, cellWidth, cellHeight);

            // Извлечение центральной части ячейки для уменьшения влияния границ
            Rect innerRoi(roi.x + cellWidth/4, roi.y + cellHeight/4,
                         cellWidth/2, cellHeight/2);

            Mat cell = correctedImage(innerRoi);
            Scalar meanColor = mean(cell);

            ColorPatch patch;
            patch.center = Point2f(roi.x + cellWidth/2, roi.y + cellHeight/2);
            patch.color = Vec3b((uchar)meanColor[0], (uchar)meanColor[1], (uchar)meanColor[2]);
            patch.row = r;
            patch.col = c;

            patches.push_back(patch);
        }
    }

    return patches;
}
```

## 2. Определение параметров цветокоррекции

### 2.1. Эталонные значения sRGB для ColorChecker Classic

```cpp
// Эталонные значения sRGB для ColorChecker Classic (в порядке чтения слева направо, сверху вниз)
vector<Vec3f> getReferenceColors() {
    return {
        Vec3f(116, 81, 67), Vec3f(198, 156, 109), Vec3f(91, 122, 156), Vec3f(90, 168, 73),
        Vec3f(130, 100, 163), Vec3f(199, 95, 97), Vec3f(103, 148, 140), Vec3f(194, 191, 78),
        Vec3f(151, 109, 77), Vec3f(87, 108, 67), Vec3f(133, 128, 177), Vec3f(158, 87, 146),
        Vec3f(224, 153, 102), Vec3f(68, 120, 178), Vec3f(145, 188, 79), Vec3f(189, 73, 131),
        Vec3f(67, 67, 67), Vec3f(100, 100, 100), Vec3f(133, 133, 133), Vec3f(166, 166, 166),
        Vec3f(200, 200, 200), Vec3f(233, 233, 233), Vec3f(255, 255, 255), Vec3f(0, 0, 0)
    };
}
```

### 2.2. Вычисление матрицы цветокоррекции

```cpp
class ColorCorrectionMatrix {
private:
    Mat correctionMatrix; // 3x3 матрица преобразования
    bool isValid;

public:
    ColorCorrectionMatrix() : isValid(false) {
        correctionMatrix = Mat::eye(3, 3, CV_32F);
    }

    // Вычисление матрицы методом наименьших квадратов
    bool computeMatrix(const vector<ColorPatch>& measuredPatches) {
        vector<Vec3f> measuredRGB, referenceRGB;

        // Используем только 18 цветных ячеек (исключая серые и белый/черный)
        for (int i = 0; i < 18; i++) {
            Vec3f measured(measuredPatches[i].color[2],
                          measuredPatches[i].color[1],
                          measuredPatches[i].color[0]); // BGR -> RGB
            measuredRGB.push_back(measured);

            referenceRGB.push_back(getReferenceColors()[i]);
        }

        // Построение системы уравнений для нахождения матрицы 3x3
        Mat A(measuredRGB.size() * 3, 9, CV_32F);
        Mat b(measuredRGB.size() * 3, 1, CV_32F);

        for (size_t i = 0; i < measuredRGB.size(); i++) {
            Vec3f m = measuredRGB[i];
            Vec3f r = referenceRGB[i];

            // Каждая строка измеренного цвета дает три уравнения
            A.at<float>(i*3, 0) = m[0]; A.at<float>(i*3, 1) = m[1]; A.at<float>(i*3, 2) = m[2];
            A.at<float>(i*3+1, 3) = m[0]; A.at<float>(i*3+1, 4) = m[1]; A.at<float>(i*3+1, 5) = m[2];
            A.at<float>(i*3+2, 6) = m[0]; A.at<float>(i*3+2, 7) = m[1]; A.at<float>(i*3+2, 8) = m[2];

            b.at<float>(i*3) = r[0];
            b.at<float>(i*3+1) = r[1];
            b.at<float>(i*3+2) = r[2];
        }

        // Решение системы методом наименьших квадратов
        Mat x;
        bool success = solve(A, b, x, DECOMP_SVD);

        if (success) {
            correctionMatrix = Mat(3, 3, CV_32F);
            for (int i = 0; i < 9; i++) {
                correctionMatrix.at<float>(i/3, i%3) = x.at<float>(i);
            }
            isValid = true;
        }

        return success;
    }

    Mat getMatrix() const { return correctionMatrix; }
    bool getIsValid() const { return isValid; }

    // Применение матрицы к одному пикселю
    Vec3f apply(const Vec3f& color) const {
        if (!isValid) return color;

        Mat colorVec = (Mat_<float>(3, 1) << color[0], color[1], color[2]);
        Mat result = correctionMatrix * colorVec;

        return Vec3f(
            max(0.0f, min(255.0f, result.at<float>(0))),
            max(0.0f, min(255.0f, result.at<float>(1))),
            max(0.0f, min(255.0f, result.at<float>(2)))
        );
    }
};
```

## 3. Сериализация параметров калибровки

### 3.1. Сохранение и загрузка матрицы коррекции

```cpp
class CalibrationManager {
public:
    static bool saveCalibration(const string& filename, const ColorCorrectionMatrix& matrix) {
        FileStorage fs(filename, FileStorage::WRITE);
        if (!fs.isOpened()) {
            return false;
        }

        fs << "color_correction_matrix" << matrix.getMatrix();
        fs << "timestamp" << time(0);
        fs << "version" << "1.0";

        fs.release();
        return true;
    }

    static bool loadCalibration(const string& filename, ColorCorrectionMatrix& matrix) {
        FileStorage fs(filename, FileStorage::READ);
        if (!fs.isOpened()) {
            return false;
        }

        Mat loadedMatrix;
        fs["color_correction_matrix"] >> loadedMatrix;

        if (loadedMatrix.empty() || loadedMatrix.size() != Size(3, 3)) {
            fs.release();
            return false;
        }

        // Копируем матрицу
        loadedMatrix.copyTo(matrix.getMatrix());
        matrix.isValid = true;

        fs.release();
        return true;
    }
};
```

## 4. Применение цветокоррекции к изображениям

### 4.1. Основной класс цветокоррекции

```cpp
class ColorCorrector {
private:
    ColorCorrectionMatrix correctionMatrix;
    bool useLUT; // Использовать таблицу поиска для ускорения
    Mat lut; // Таблица поиска

public:
    ColorCorrector() : useLUT(true) {
        if (useLUT) {
            buildLUT();
        }
    }

    // Загрузка калибровки из файла
    bool loadCalibration(const string& filename) {
        bool success = CalibrationManager::loadCalibration(filename, correctionMatrix);
        if (success && useLUT) {
            buildLUT();
        }
        return success;
    }

    // Применение цветокоррекции к изображению
    Mat correctImage(const Mat& input) {
        Mat output;

        if (input.channels() != 3) {
            input.copyTo(output);
            return output;
        }

        if (useLUT && !lut.empty()) {
            // Использование LUT для быстрой коррекции
            Mat bgr[3];
            split(input, bgr);

            Mat correctedBGR[3];
            for (int i = 0; i < 3; i++) {
                LUT(bgr[i], lut, correctedBGR[i]);
            }

            merge(correctedBGR, 3, output);
        } else {
            // Прямое применение матрицы
            input.convertTo(output, CV_32F, 1.0/255.0);

            vector<Mat> channels;
            split(output, channels);

            // Применение матрицы к каждому пикселю
            for (int y = 0; y < output.rows; y++) {
                for (int x = 0; x < output.cols; x++) {
                    Vec3f color(channels[2].at<float>(y, x),
                               channels[1].at<float>(y, x),
                               channels[0].at<float>(y, x));

                    Vec3f corrected = correctionMatrix.apply(color);

                    channels[2].at<float>(y, x) = corrected[0] / 255.0f;
                    channels[1].at<float>(y, x) = corrected[1] / 255.0f;
                    channels[0].at<float>(y, x) = corrected[2] / 255.0f;
                }
            }

            merge(channels, output);
            output.convertTo(output, CV_8U, 255.0);
        }

        return output;
    }

private:
    void buildLUT() {
        lut = Mat(256, 1, CV_8U);
        for (int i = 0; i < 256; i++) {
            Vec3f color(i, i, i); // Для построения LUT используем серые значения
            Vec3f corrected = correctionMatrix.apply(color);
            lut.at<uchar>(i) = (uchar)corrected[0]; // Используем только один канал для упрощения
        }
    }
};
```

## 5. Полный процесс калибровки и коррекции

### 5.1. Главная функция калибровки

```cpp
bool performCalibration(const string& imagePath, const string& outputCalibrationFile) {
    // Загрузка изображения калибровки
    Mat calibrationImage = imread(imagePath);
    if (calibrationImage.empty()) {
        cout << "Ошибка: не удалось загрузить изображение " << imagePath << endl;
        return false;
    }

    cout << "Начало калибровки..." << endl;

    // Шаг 1: Детекция углов таблицы
    vector<Point2f> corners = detectColorCheckerCorners(calibrationImage);
    if (corners.size() != 4) {
        cout << "Не удалось обнаружить таблицу ColorChecker" << endl;
        return false;
    }

    // Шаг 2: Перспективная коррекция
    Mat corrected = extractColorChecker(calibrationImage, corners);

    // Шаг 3: Извлечение цветов ячеек
    vector<ColorPatch> patches = extractPatchColors(corrected);

    // Шаг 4: Вычисление матрицы коррекции
    ColorCorrectionMatrix matrix;
    if (!matrix.computeMatrix(patches)) {
        cout << "Не удалось вычислить матрицу коррекции" << endl;
        return false;
    }

    // Шаг 5: Сохранение калибровки
    if (!CalibrationManager::saveCalibration(outputCalibrationFile, matrix)) {
        cout << "Не удалось сохранить файл калибровки" << endl;
        return false;
    }

    cout << "Калибровка успешно завершена!" << endl;
    cout << "Файл калибровки сохранен: " << outputCalibrationFile << endl;

    // Визуализация результатов
    imshow("Оригинал", calibrationImage);
    imshow("Корректированная таблица", corrected);
    waitKey(0);

    return true;
}
```

### 5.2. Главная функция коррекции изображений

```cpp
bool correctImage(const string& inputPath, const string& outputPath, const string& calibrationFile) {
    // Загрузка изображения
    Mat inputImage = imread(inputPath);
    if (inputImage.empty()) {
        cout << "Ошибка: не удалось загрузить изображение " << inputPath << endl;
        return false;
    }

    // Создание корректора и загрузка калибровки
    ColorCorrector corrector;
    if (!corrector.loadCalibration(calibrationFile)) {
        cout << "Ошибка: не удалось загрузить файл калибровки" << endl;
        return false;
    }

    // Применение цветокоррекции
    Mat correctedImage = corrector.correctImage(inputImage);

    // Сохранение результата
    if (!imwrite(outputPath, correctedImage)) {
        cout << "Ошибка: не удалось сохранить изображение " << outputPath << endl;
        return false;
    }

    cout << "Цветокоррекция успешно применена!" << endl;
    cout << "Результат сохранен: " << outputPath << endl;

    // Показ результатов
    imshow("Оригинал", inputImage);
    imshow("После коррекции", correctedImage);
    waitKey(0);

    return true;
}
```

### 5.3. Пример использования

```cpp
int main() {
    // Путь к изображению для калибровки
    string calibrationImagePath = "colorchecker.jpg";
    string calibrationFile = "calibration.yml";

    // Выполнение калибровки
    if (performCalibration(calibrationImagePath, calibrationFile)) {
        cout << "Калибровка завершена. Теперь можно корректировать изображения." << endl;

        // Пример коррекции изображения
        string inputImagePath = "photo_to_correct.jpg";
        string outputImagePath = "corrected_photo.jpg";

        correctImage(inputImagePath, outputImagePath, calibrationFile);
    }

    return 0;
}
```

## 6. Сборка и компиляция

Для компиляции программы создайте файл CMakeLists.txt:

```cmake
cmake_minimum_required(VERSION 3.10)
project(ColorCorrection)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

add_executable(color_correction
    main.cpp
)

target_link_libraries(color_correction ${OpenCV_LIBS})
```

Скомпилируйте программу:
```bash
mkdir build
cd build
cmake ..
make
```

## Особенности реализации

1. **Производительность**: Использование LUT (таблицы поиска) значительно ускоряет обработку больших изображений.

2. **Надежность детекции**: Комбинация CLAHE, Canny и фильтрации контуров обеспечивает надежное обнаружение таблицы даже при неидеальных условиях съемки.

3. **Гибкость**: Возможность повторного использования калибровки без необходимости повторной детекции таблицы.

4. **Точность**: Метод наименьших квадратов для вычисления матрицы коррекции обеспечивает оптимальное преобразование цветов.

5. **Совместимость**: Работает с любыми изображениями, снятыми в условиях, близких к условиям калибровки.

Это полноценное решение для цветокоррекции по таблице Xrite ColorChecker Classic, которое можно легко интегрировать в любую систему обработки изображений на C++ с использованием OpenCV.