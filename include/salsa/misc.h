#pragma once

#include <cmath>
#include <stdio.h>

#ifndef DEBUGPRINT
#define DEBUGPRINT 1
#endif

#define DEBUGPRINTLEVEL 4
#define DEBUGLOGLEVEL 0

#if DEBUGPRINT
#define SL std::cout << __LINE__ << std::endl
#define SD(level, f_, ...) do{ \
    if ((level) >= DEBUGPRINTLEVEL) {\
        printf((f_), ##__VA_ARGS__);\
        printf("\n");\
    }\
    if ((level) >= DEBUGLOGLEVEL) {\
        char buf[100];\
        int n = sprintf(buf, (f_), ##__VA_ARGS__);\
        logs_[log::Graph]->file_.write(buf, n);\
        logs_[log::Graph]->file_ << "\n";\
        logs_[log::Graph]->file_.flush();\
    }\
} while(false)

#define SD_S(level, args) do{\
    if ((level) >= DEBUGPRINTLEVEL) {\
        std::cout << args << std::endl;\
        }\
    if ((level) >= DEBUGLOGLEVEL) {\
        logs_[log::Graph]->file_ << args << std::endl;\
        logs_[log::Graph]->file_.flush();\
    }\
} while(false)
#else
#define SL
#define SD(...)
#define SD_S(...)
#endif

//#ifndef NDEBUG
#define SALSA_ASSERT(condition, ...) \
    do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << " line " << __LINE__ << ": "; \
            fprintf(stderr, __VA_ARGS__);\
            std::cerr << std::endl; \
            assert(condition); \
            throw std::runtime_error("ERROR:"); \
        } \
    } while (false)
//#else
//#   define SALSA_ASSERT(...)
//#endif

