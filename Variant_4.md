# Проектирование алгоритма цветокоррекции с OpenCV и X-Rite ColorChecker

Этот документ описывает архитектуру и шаги разработки для создания системы цветокоррекции изображений на C++ с использованием библиотеки OpenCV. Цель — калибровка по стандартной мишени X-Rite ColorChecker Classic и последующая коррекция цветовых искажений для приведения изображений к цветовому пространству sRGB.

## Общая концепция

Алгоритм делится на два основных этапа:

Фаза калибровки (однократная): На этом этапе мы используем изображение с эталонной мишенью ColorChecker. Алгоритм находит мишень, извлекает средние цвета с ее 24 плашек и сравнивает их с известными эталонными sRGB-значениями. Результатом является матрица цветовой коррекции (CCM - Color Correction Matrix), которая математически описывает, как преобразовать "неправильные" цвета камеры в "правильные" sRGB-цвета. Эта матрица сохраняется для дальнейшего использования.

Фаза применения (многократная): На этом этапе сохраненная матрица CCM применяется к любым новым изображениям, полученным с той же камеры и при схожих условиях освещения. Каждый пиксель изображения умножается на эту матрицу, что приводит к коррекции его цвета.

## Этап 1: Калибровка системы

Этот этап выполняется один раз для определения параметров трансформации.

Шаг 1: Детекция мишени ColorChecker на изображении
Первая и самая важная задача — автоматически найти расположение мишени на калибровочном изображении. К счастью, в opencv_contrib есть специализированный модуль mcc (Macbeth Color Chart), созданный для этой цели.

Инструменты:

Модуль cv::mcc.

Класс cv::mcc::CCheckerDetector.

Процесс:

Загрузить калибровочное изображение, на котором хорошо видна мишень ColorChecker.

Создать экземпляр детектора: cv::mcc::CCheckerDetector::create().

Вызвать метод process() детектора, передав ему изображение. Этот метод ищет на изображении структуры, похожие на сетку 6x4 мишени ColorChecker.

Получить список найденных мишеней с помощью getListColorChecker(). В большинстве случаев нас будет интересовать первая и самая достоверная из найденных мишеней.

Псевдокод:

``` cpp
#include <opencv2/mcc.hpp>
#include <opencv2/imgcodecs.hpp>

// ...

// Загружаем изображение с мишенью
cv::Mat calibrationImage = cv::imread("path/to/your/colorchecker_image.jpg");

// Создаем детектор
cv::Ptr<cv::mcc::CCheckerDetector> detector = cv::mcc::CCheckerDetector::create();

// Запускаем детекцию
detector->process(calibrationImage, cv::mcc::MCC_DETECT_STANDARD, 1);

// Получаем список найденных мишеней
std::vector<cv::Ptr<cv::mcc::CChecker>> checkers = detector->getListColorChecker();

if (checkers.empty()) {
    // Обработка ошибки: мишень не найдена
    return;
}

// Берем первую найденную мишень
cv::Ptr<cv::mcc::CChecker> checker = checkers[0];
```

Шаг 2: Извлечение измеренных цветов с плашек
После того как мишень найдена, необходимо получить средний цвет каждой из 24 плашек. Модуль mcc также автоматизирует этот процесс.

Инструменты:

Объект cv::mcc::CChecker, полученный на предыдущем шаге.

Метод getChartsRGB().

Процесс:

Из объекта checker вызвать метод getChartsRGB().

Этот метод вернет матрицу cv::Mat размером 24x1 типа CV_8UC3. Каждый элемент этой матрицы — это средний BGR-цвет соответствующей плашки, измеренный на изображении.

Важно: Цвета нужно будет преобразовать в float и нормализовать в диапазон [0, 1] для дальнейших вычислений.

Псевдокод:

``` cpp
// Получаем матрицу с цветами плашек
cv::Mat measuredColorsBGR = checker->getChartsRGB();

// Конвертируем в float и нормализуем
cv::Mat measuredColorsFloat;
measuredColorsBGR.convertTo(measuredColorsFloat, CV_32F, 1.0/255.0);

// OpenCV использует порядок BGR, для стандартных вычислений часто нужен RGB.
// Поменяем каналы местами.
cv::Mat measuredColorsRGB;
cv::cvtColor(measuredColorsFloat, measuredColorsRGB, cv::COLOR_BGR2RGB);

// measuredColorsRGB - это матрица 24x1 типа CV_32FC3 с измеренными цветами.
```

Шаг 3: Определение эталонных sRGB цветов
Нам нужны целевые, "идеальные" цвета, к которым мы будем приводить наши измеренные значения. Это стандартные sRGB-значения для 24 плашек ColorChecker Classic.

Данные (Эталонные значения sRGB [0-255]):

``` cpp
// Патч | R,   G,   B   | Описание
//-------------------------------------------------
{115, 82,  68},    // 1.  Dark skin
{194, 150, 130},   // 2.  Light skin
{98,  122, 157},   // 3.  Blue sky
{87,  108, 67},    // 4.  Foliage
{133, 128, 177},   // 5.  Blue flower
{103, 189, 170},   // 6.  Bluish green
{214, 126, 44},    // 7.  Orange
{80,  91,  166},   // 8.  Purplish blue
{193, 90,  99},    // 9.  Moderate red
{94,  60,  108},   // 10. Purple
{157, 188, 64},    // 11. Yellow green
{224, 163, 46},    // 12. Orange yellow
{56,  61,  150},   // 13. Blue
{70,  148, 73},    // 14. Green
{175, 54,  60},    // 15. Red
{231, 199, 31},    // 16. Yellow
{187, 86,  149},   // 17. Magenta
{8,   133, 161},   // 18. Cyan
{243, 243, 242},   // 19. White
{200, 200, 200},   // 20. Neutral 8
{160, 160, 160},   // 21. Neutral 6.5
{122, 122, 121},   // 22. Neutral 5
{85,  85,  85},    // 23. Neutral 3.5
{52,  52,  52}     // 24. Black
```

Эти значения нужно будет объявить как cv::Mat размером 24x3 типа CV_32FC1 (или 24x1 CV_32FC3), также нормализовав в диапазон [0, 1].

Шаг 4: Расчет матрицы цветовой коррекции (CCM)
Это ядро калибровки. Мы хотим найти такую матрицу M, что:
ReferenceColor = M * MeasuredColor

Для решения этой задачи используется метод наименьших квадратов. Мы ищем 3x3 матрицу M, которая минимизирует ошибку между преобразованными измеренными цветами и эталонными цветами для всех 24 плашек.

Процесс:

Сформировать матрицу O (Observed/Measured) размером 24x3 из измеренных RGB-значений.

Сформировать матрицу P (Perfect/Reference) размером 24x3 из эталонных sRGB-значений.

Решить систему линейных уравнений P = O * M^T относительно M^T. Решение методом наименьших квадратов:
M^T = (O^T * O)^-1 * O^T * P
Где T — транспонирование, ^-1 — обращение матрицы.

Псевдокод:

``` cpp
// O - measuredColorsRGB, P - referenceColorsRGB (обе 24x3, CV_32F)
// Предполагаем, что матрицы уже созданы и заполнены

cv::Mat O = measuredColorsRGB.reshape(1, 24); // Преобразуем в 24x3
cv::Mat P = referenceColorsRGB.reshape(1, 24); // Преобразуем в 24x3

// Решаем уравнение M_transpose = (O^T * O)^-1 * O^T * P
cv::Mat M_transpose;
cv::solve(O, P, M_transpose, cv::DECOMP_SVD);

// Наша матрица CCM - это транспонированная M_transpose
cv::Mat ccm = M_transpose.t();

// ccm - это искомая матрица 3x3 типа CV_32F.
// Ее необходимо сохранить в файл для использования на следующем этапе.
```

Альтернатива: Модуль cv::ccm::ColorCorrectionModel может сделать это за вас, но понимание ручного расчета полезно.

## Этап 2: Применение цветокоррекции

После получения и сохранения матрицы ccm, ее можно применять к любым изображениям.

Шаг 5: Применение матрицы к изображению
Для коррекции нового изображения каждый его пиксель (представленный как RGB-вектор) должен быть умножен на матрицу ccm.

Инструменты:

Функция cv::transform.

Процесс:

Загрузить изображение для коррекции.

Убедиться, что оно имеет тип CV_32FC3 (3 канала, float) и значения нормализованы в [0, 1].

Применить функцию cv::transform, передав ей исходное изображение, целевое изображение и матрицу ccm.

Псевдокод:

``` cpp
// Загружаем сохраненную ранее матрицу CCM
cv::Mat ccm; // ...загрузка из файла...

// Загружаем изображение, которое нужно скорректировать
cv::Mat imageToCorrect = cv::imread("path/to/new_image.jpg");

// Конвертируем в float и нормализуем
cv::Mat imageFloat;
imageToCorrect.convertTo(imageFloat, CV_32F, 1.0/255.0);

// Важно: меняем BGR на RGB, т.к. матрица была рассчитана для RGB
cv::Mat imageRGB;
cv::cvtColor(imageFloat, imageRGB, cv::COLOR_BGR2RGB);

// Создаем пустое изображение для результата
cv::Mat correctedImageRGB = cv::Mat::zeros(imageRGB.size(), imageRGB.type());

// Применяем трансформацию
cv::transform(imageRGB, correctedImageRGB, ccm);

// Возвращаем каналы в порядок BGR для сохранения или отображения в OpenCV
cv::Mat correctedImageBGR;
cv::cvtColor(correctedImageRGB, correctedImageBGR, cv::COLOR_RGB2BGR);

// Опционально: обрезаем значения, которые могли выйти за пределы [0, 1]
cv::Mat finalImage;
correctedImageBGR.convertTo(finalImage, CV_8U, 255.0);
cv::max(0, finalImage, finalImage);
cv::min(255, finalImage, finalImage);

// Показываем или сохраняем результат
cv::imshow("Corrected Image", finalImage);
cv::waitKey(0);
```

## Ключевые соображения и дальнейшие шаги

Линеаризация (Гамма-коррекция): Для математически корректного результата цветовое преобразование должно выполняться в линейном цветовом пространстве. Изображения с камеры и sRGB-значения обычно гамма-скорректированы (гамма ≈ 2.2).

Улучшенный пайплайн:

Измерить цвета с плашек.

Применить к ним обратную гамма-коррекцию (возвести в степень ~2.2), чтобы перевести в линейное пространство.

Эталонные sRGB-значения также перевести в линейное пространство.

Рассчитать матрицу ccm на линейных данных.

При применении: взять новое изображение, линеаризовать его, применить ccm, а затем к результату применить прямую гамма-коррекцию (возвести в степень 1/2.2).

Это усложняет процесс, но значительно повышает точность.

Баланс белого и освещенность: Данный алгоритм в первую очередь исправляет оттеночные искажения. Он также частично корректирует баланс белого, так как "заставляет" серые плашки стать нейтрально-серыми. Однако для серьезных проблем с балансом белого или неравномерной освещенностью (виньетирование) потребуются отдельные алгоритмы, которые могут быть применены до или после этого шага.

Структура кода: Рекомендуется создать класс ColorCalibrator, который будет инкапсулировать логику калибровки (хранить ccm) и предоставлять метод correct(cv::Mat& image).

``` cpp
// main.cpp
//
// Пример реализации алгоритма цветокоррекции с использованием OpenCV.
//
// Программа имеет два режима работы:
// 1. calibrate: Создает матрицу цветовой коррекции (CCM) по изображению с мишенью
//    X-Rite ColorChecker и сохраняет ее в файл.
//
// 2. correct: Загружает ранее созданную матрицу CCM и применяет ее к новому
//    изображению для коррекции цветов.
//
// Зависимости:
// - OpenCV
// - OpenCV Contrib (для модуля mcc)
//
// Сборка (пример для g++):
// g++ main.cpp -o color_corrector `pkg-config --cflags --libs opencv4`
//
// Использование:
//
// Для калибровки:
// ./color_corrector calibrate <путь_к_изображению_с_мишенью> <путь_для_сохранения_ccm.yml>
//
// Для коррекции:
// ./color_corrector correct <путь_к_ccm.yml> <путь_к_входному_изображению> <путь_для_сохранения_результата.jpg>
//

#include <iostream>
#include <vector>
#include <fstream>

// Основные заголовочные файлы OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// Заголовочный файл для детекции мишени ColorChecker
#include <opencv2/mcc.hpp>

// ======================================================================================
//                          Вспомогательные функции
// ======================================================================================

/**
 * @brief Преобразует значение цвета из пространства sRGB в линейное RGB.
 * Это необходимо для математически корректных вычислений.
 * @param val Значение канала цвета в диапазоне [0, 1].
 * @return Линеаризованное значение канала.
 */
float srgbToLinear(float val) {
    if (val <= 0.04045f) {
        return val / 12.92f;
    }
    return std::pow((val + 0.055f) / 1.055f, 2.4f);
}

/**
 * @brief Преобразует значение цвета из линейного RGB в пространство sRGB.
 * @param val Линеаризованное значение канала.
 * @return Значение канала в пространстве sRGB в диапазоне [0, 1].
 */
float linearToSrgb(float val) {
    if (val <= 0.0031308f) {
        return val * 12.92f;
    }
    return 1.055f * std::pow(val, 1.0f / 2.4f) - 0.055f;
}

/**
 * @brief Применяет функцию к каждому элементу многоканальной матрицы.
 * @param mat Матрица для обработки (тип CV_32F).
 * @param func Указатель на функцию, которую нужно применить.
 */
void applyToMat(cv::Mat& mat, float (*func)(float)) {
    if (mat.type() != CV_32FC3 && mat.type() != CV_32FC1) {
        std::cerr << "Ошибка: Матрица должна быть типа CV_32F" << std::endl;
        return;
    }
    // Проходим по каждому элементу матрицы
    for (int r = 0; r < mat.rows; ++r) {
        for (int c = 0; c < mat.cols; ++c) {
            if (mat.channels() == 3) {
                cv::Vec3f& pixel = mat.at<cv::Vec3f>(r, c);
                pixel[0] = func(pixel[0]);
                pixel[1] = func(pixel[1]);
                pixel[2] = func(pixel[2]);
            } else {
                float& val = mat.at<float>(r, c);
                val = func(val);
            }
        }
    }
}


// ======================================================================================
//                          Этап 1: КАЛИБРОВКА
// ======================================================================================

/**
 * @brief Выполняет калибровку: находит мишень, извлекает цвета и вычисляет CCM.
 * @param checkerImagePath Путь к изображению с мишенью ColorChecker.
 * @param ccmOutputPath Путь для сохранения файла с матрицей CCM.
 * @return true в случае успеха, false в случае ошибки.
 */
bool performCalibration(const std::string& checkerImagePath, const std::string& ccmOutputPath) {
    std::cout << "--- Начало калибровки ---" << std::endl;

    // --- Шаг 1: Детекция мишени ColorChecker ---
    std::cout << "1. Загрузка изображения и поиск мишени..." << std::endl;
    cv::Mat calibrationImage = cv::imread(checkerImagePath);
    if (calibrationImage.empty()) {
        std::cerr << "Ошибка: не удалось загрузить изображение: " << checkerImagePath << std::endl;
        return false;
    }

    cv::Ptr<cv::mcc::CCheckerDetector> detector = cv::mcc::CCheckerDetector::create();
    if (!detector->process(calibrationImage, cv::mcc::MCC_DETECT_STANDARD, 1)) {
        std::cerr << "Ошибка: мишень ColorChecker не найдена на изображении." << std::endl;
        return false;
    }

    std::vector<cv::Ptr<cv::mcc::CChecker>> checkers = detector->getListColorChecker();
    if (checkers.empty()) {
        std::cerr << "Ошибка: список найденных мишеней пуст." << std::endl;
        return false;
    }
    cv::Ptr<cv::mcc::CChecker> checker = checkers[0];
    std::cout << "   ...Мишень успешно найдена." << std::endl;

    // --- Шаг 2: Извлечение измеренных цветов с плашек ---
    std::cout << "2. Извлечение цветов с плашек мишени..." << std::endl;
    cv::Mat measuredColorsBGR = checker->getChartsRGB();
    cv::Mat measuredColorsFloat;
    measuredColorsBGR.convertTo(measuredColorsFloat, CV_32F, 1.0 / 255.0);

    cv::Mat measuredColorsRGB;
    cv::cvtColor(measuredColorsFloat, measuredColorsRGB, cv::COLOR_BGR2RGB);

    // Линеаризация измеренных цветов
    applyToMat(measuredColorsRGB, srgbToLinear);
    std::cout << "   ...Цвета извлечены и линеаризованы." << std::endl;


    // --- Шаг 3: Определение эталонных sRGB цветов ---
    std::cout << "3. Формирование матрицы эталонных цветов..." << std::endl;
    float ref_data[24][3] = {
        {115, 82,  68},    {194, 150, 130},   {98,  122, 157},   {87,  108, 67},
        {133, 128, 177},   {103, 189, 170},   {214, 126, 44},    {80,  91,  166},
        {193, 90,  99},    {94,  60,  108},   {157, 188, 64},    {224, 163, 46},
        {56,  61,  150},   {70,  148, 73},    {175, 54,  60},    {231, 199, 31},
        {187, 86,  149},   {8,   133, 161},   {243, 243, 242},   {200, 200, 200},
        {160, 160, 160},   {122, 122, 121},   {85,  85,  85},    {52,  52,  52}
    };
    cv::Mat referenceColorsRGB(24, 1, CV_32FC3, &ref_data);
    referenceColorsRGB /= 255.0; // Нормализация в диапазон [0, 1]

    // Линеаризация эталонных цветов
    applyToMat(referenceColorsRGB, srgbToLinear);
    std::cout << "   ...Эталонные цвета сформированы и линеаризованы." << std::endl;

    // --- Шаг 4: Расчет матрицы цветовой коррекции (CCM) ---
    std::cout << "4. Расчет матрицы цветовой коррекции (CCM)..." << std::endl;
    cv::Mat O = measuredColorsRGB.reshape(1, 24);
    cv::Mat P = referenceColorsRGB.reshape(1, 24);

    cv::Mat M_transpose;
    cv::solve(O, P, M_transpose, cv::DECOMP_SVD);
    cv::Mat ccm = M_transpose.t();

    std::cout << "   ...Матрица CCM рассчитана:" << std::endl << ccm << std::endl;

    // Сохранение матрицы в файл
    cv::FileStorage fs(ccmOutputPath, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        std::cerr << "Ошибка: не удалось открыть файл для записи: " << ccmOutputPath << std::endl;
        return false;
    }
    fs << "CCM" << ccm;
    fs.release();
    std::cout << "   ...Матрица CCM сохранена в " << ccmOutputPath << std::endl;

    std::cout << "--- Калибровка успешно завершена ---" << std::endl;
    return true;
}


// ======================================================================================
//                          Этап 2: ПРИМЕНЕНИЕ КОРРЕКЦИИ
// ======================================================================================

/**
 * @brief Выполняет цветокоррекцию изображения с использованием ранее сохраненной CCM.
 * @param ccmPath Путь к файлу с матрицей CCM.
 * @param inputImagePath Путь к входному изображению для коррекции.
 * @param outputImagePath Путь для сохранения скорректированного изображения.
 * @return true в случае успеха, false в случае ошибки.
 */
bool performCorrection(const std::string& ccmPath, const std::string& inputImagePath, const std::string& outputImagePath) {
    std::cout << "--- Начало коррекции изображения ---" << std::endl;

    // --- Шаг 1: Загрузка CCM ---
    std::cout << "1. Загрузка матрицы CCM из файла..." << std::endl;
    cv::Mat ccm;
    cv::FileStorage fs(ccmPath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Ошибка: не удалось открыть файл с матрицей: " << ccmPath << std::endl;
        return false;
    }
    fs["CCM"] >> ccm;
    fs.release();

    if (ccm.empty() || ccm.rows != 3 || ccm.cols != 3) {
        std::cerr << "Ошибка: некорректная матрица CCM в файле." << std::endl;
        return false;
    }
    std::cout << "   ...Матрица CCM загружена." << std::endl;

    // --- Шаг 2: Загрузка и подготовка изображения ---
    std::cout << "2. Загрузка и подготовка изображения для коррекции..." << std::endl;
    cv::Mat imageToCorrect = cv::imread(inputImagePath);
    if (imageToCorrect.empty()) {
        std::cerr << "Ошибка: не удалось загрузить изображение: " << inputImagePath << std::endl;
        return false;
    }

    cv::Mat imageFloat;
    imageToCorrect.convertTo(imageFloat, CV_32F, 1.0 / 255.0);

    cv::Mat imageRGB;
    cv::cvtColor(imageFloat, imageRGB, cv::COLOR_BGR2RGB);

    // Линеаризация изображения
    applyToMat(imageRGB, srgbToLinear);
    std::cout << "   ...Изображение подготовлено." << std::endl;


    // --- Шаг 3: Применение матрицы к изображению ---
    std::cout << "3. Применение цветовой трансформации..." << std::endl;
    cv::Mat correctedImageRGB = cv::Mat::zeros(imageRGB.size(), imageRGB.type());
    cv::transform(imageRGB, correctedImageRGB, ccm);
    std::cout << "   ...Трансформация применена." << std::endl;


    // --- Шаг 4: Постобработка и сохранение ---
    std::cout << "4. Конвертация в sRGB и сохранение результата..." << std::endl;
    // Возврат в пространство sRGB
    applyToMat(correctedImageRGB, linearToSrgb);

    // Обрезка значений, которые могли выйти за пределы диапазона [0, 1]
    cv::max(0.0f, correctedImageRGB, correctedImageRGB);
    cv::min(1.0f, correctedImageRGB, correctedImageRGB);

    cv::Mat correctedImageBGR;
    cv::cvtColor(correctedImageRGB, correctedImageBGR, cv::COLOR_RGB2BGR);

    cv::Mat finalImage;
    correctedImageBGR.convertTo(finalImage, CV_8U, 255.0);

    if (!cv::imwrite(outputImagePath, finalImage)) {
        std::cerr << "Ошибка: не удалось сохранить результат в " << outputImagePath << std::endl;
        return false;
    }
    std::cout << "   ...Скорректированное изображение сохранено в " << outputImagePath << std::endl;

    std::cout << "--- Коррекция успешно завершена ---" << std::endl;
    return true;
}


// ======================================================================================
//                          Главная функция
// ======================================================================================

void printUsage() {
    std::cout << "Использование:" << std::endl;
    std::cout << "  Для калибровки:" << std::endl;
    std::cout << "    color_corrector calibrate <checker_image_path> <ccm_output_path.yml>" << std::endl;
    std::cout << "  Для коррекции:" << std::endl;
    std::cout << "    color_corrector correct <ccm_input_path.yml> <input_image_path> <output_image_path.jpg>" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage();
        return 1;
    }

    std::string mode = argv[1];

    if (mode == "calibrate") {
        if (argc != 4) {
            printUsage();
            return 1;
        }
        if (!performCalibration(argv[2], argv[3])) {
            std::cerr << "Произошла ошибка во время калибровки." << std::endl;
            return 1;
        }
    } else if (mode == "correct") {
        if (argc != 5) {
            printUsage();
            return 1;
        }
        if (!performCorrection(argv[2], argv[3], argv[4])) {
            std::cerr << "Произошла ошибка во время коррекции." << std::endl;
            return 1;
        }
    } else {
        std::cerr << "Неизвестный режим: " << mode << std::endl;
        printUsage();
        return 1;
    }

    return 0;
}
```
