#pragma once

#include <cmath>
#include <stdio.h>
#include <gtest/gtest.h>

//#ifndef NDEBUG
#define SALSA_ASSERT(condition, ...) \
    do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << " line " << __LINE__ << ": " << printf(__VA_ARGS__) << std::endl; \
            assert(condition); \
        } \
    } while (false)
//#else
//#   define SALSA_ASSERT(...)
//#endif

#ifndef DEBUGPRINT
#define DEBUGPRINT 1
#endif

#if DEBUGPRINT
#define SL std::cout << __LINE__ << std::endl
#define SD(f_, ...) printf((f_), ##__VA_ARGS__)
#define SD_S(args) std::cout << args;
#else
#define SL
#define SD(...)
#define SD_S(...)
#endif
