#pragma once

#include <cmath>
#include <stdio.h>

#ifndef DEBUGPRINT
#define DEBUGPRINT 1
#endif

#define DEBUGPRINTLEVEL 4
#define DEBUGLOGLEVEL 4

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

/*************************************/
/*          Round-Off helpers        */
/*************************************/
constexpr double eps = 1e-4;
inline bool lt(double t0, double t1) { return t0 < t1-2.0*eps; }
inline bool le(double t0, double t1) { return t0-2.0*eps <= t1; }
inline bool gt(double t0, double t1) { return t0 > t1+2.0*eps; }
inline bool ge(double t0, double t1) { return t0+2.0*eps >= t1; }
inline bool eq(double t0, double t1) { return std::abs(t0 - t1) <= 2.0*eps; }
inline bool ne(double t0, double t1) { return std::abs(t0 - t1) > 2.0*eps; }

