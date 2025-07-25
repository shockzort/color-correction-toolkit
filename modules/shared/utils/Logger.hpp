#pragma once

#include <iostream>
#include <sstream>
#include <string>

namespace ColorCorrection::Shared {

enum class LogLevel { DEBUG = 0, INFO = 1, WARN = 2, ERROR = 3 };

class Logger {
  public:
    static Logger& getInstance() {
        static Logger instance;
        return instance;
    }

    void setLevel(LogLevel level) { currentLevel_ = level; }

    template <typename... Args>
    void debug(Args&&... args) {
        log(LogLevel::DEBUG, "DEBUG", std::forward<Args>(args)...);
    }

    template <typename... Args>
    void info(Args&&... args) {
        log(LogLevel::INFO, "INFO", std::forward<Args>(args)...);
    }

    template <typename... Args>
    void warn(Args&&... args) {
        log(LogLevel::WARN, "WARN", std::forward<Args>(args)...);
    }

    template <typename... Args>
    void error(Args&&... args) {
        log(LogLevel::ERROR, "ERROR", std::forward<Args>(args)...);
    }

  private:
    LogLevel currentLevel_ = LogLevel::INFO;

    template <typename... Args>
    void log(LogLevel level, const std::string& levelStr, Args&&... args) {
        if (level < currentLevel_) return;

        std::ostringstream oss;
        oss << "[" << levelStr << "] ";
        ((oss << std::forward<Args>(args)), ...);
        std::cout << oss.str() << std::endl;
    }

    Logger() = default;
    ~Logger() = default;
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
};

#define LOG_DEBUG(...) ColorCorrection::Shared::Logger::getInstance().debug(__VA_ARGS__)
#define LOG_INFO(...) ColorCorrection::Shared::Logger::getInstance().info(__VA_ARGS__)
#define LOG_WARN(...) ColorCorrection::Shared::Logger::getInstance().warn(__VA_ARGS__)
#define LOG_ERROR(...) ColorCorrection::Shared::Logger::getInstance().error(__VA_ARGS__)

}  // namespace ColorCorrection::Shared