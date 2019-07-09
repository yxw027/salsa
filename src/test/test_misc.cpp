#include <gtest/gtest.h>
#include "salsa/salsa.h"

using namespace salsa;

TEST (Salsa, IndexGeNoWrap)
{
    Salsa salsa;
    salsa.STATE_BUF_SIZE = 25;
    salsa.xbuf_head_ = 20;
    salsa.xbuf_tail_ = 10;


    EXPECT_TRUE(salsa.stateIdxGe(12, 11));
    EXPECT_TRUE(salsa.stateIdxGe(20, 11));
    EXPECT_TRUE(salsa.stateIdxGe(20, 10));
    EXPECT_TRUE(salsa.stateIdxGe(11, 10));
    EXPECT_TRUE(salsa.stateIdxGe(20, 19));
    EXPECT_FALSE(salsa.stateIdxGe(13, 19));
    EXPECT_FALSE(salsa.stateIdxGe(10, 20));
    EXPECT_FALSE(salsa.stateIdxGe(10, 11));
    EXPECT_FALSE(salsa.stateIdxGe(19, 20));

    salsa.xbuf_head_ = salsa.STATE_BUF_SIZE-1;
    salsa.xbuf_tail_ = 0;

    EXPECT_TRUE(salsa.stateIdxGe(salsa.STATE_BUF_SIZE-1, 0));
    EXPECT_TRUE(salsa.stateIdxGe(1, 0));
    EXPECT_TRUE(salsa.stateIdxGe(salsa.xbuf_head_, salsa.xbuf_tail_));
    EXPECT_TRUE(salsa.stateIdxGe(salsa.STATE_BUF_SIZE-1, salsa.STATE_BUF_SIZE-2));
    EXPECT_FALSE(salsa.stateIdxGe(0, salsa.STATE_BUF_SIZE-1));
    EXPECT_FALSE(salsa.stateIdxGe(0, 1));
    EXPECT_FALSE(salsa.stateIdxGe(salsa.xbuf_tail_, salsa.xbuf_head_));
    EXPECT_FALSE(salsa.stateIdxGe(salsa.STATE_BUF_SIZE-2, salsa.STATE_BUF_SIZE-1));
}

TEST (Salsa, IndexGeWithWrap)
{
    Salsa salsa;
    salsa.STATE_BUF_SIZE = 25;
    salsa.xbuf_head_ = 10;                              `
    salsa.xbuf_tail_ = 20;


    EXPECT_TRUE(salsa.stateIdxGe(8, 22));
    EXPECT_TRUE(salsa.stateIdxGe(0, salsa.STATE_BUF_SIZE-1));
    EXPECT_TRUE(salsa.stateIdxGe(8, 0));
    EXPECT_TRUE(salsa.stateIdxGe(salsa.xbuf_head_, salsa.STATE_BUF_SIZE-1));
    EXPECT_TRUE(salsa.stateIdxGe(salsa.xbuf_head_, salsa.xbuf_tail_));
    EXPECT_TRUE(salsa.stateIdxGe(salsa.xbuf_head_, salsa.xbuf_head_-1));
    EXPECT_TRUE(salsa.stateIdxGe(salsa.xbuf_tail_+1, salsa.xbuf_tail_));
    EXPECT_TRUE(salsa.stateIdxGe(salsa.xbuf_tail_, salsa.xbuf_tail_));
    EXPECT_TRUE(salsa.stateIdxGe(salsa.xbuf_head_, salsa.xbuf_head_));
    EXPECT_TRUE(salsa.stateIdxGe(8,8));

    EXPECT_FALSE(salsa.stateIdxGe(22, 8));
    EXPECT_FALSE(salsa.stateIdxGe(salsa.STATE_BUF_SIZE-1, 0));
    EXPECT_FALSE(salsa.stateIdxGe(0, 8));
    EXPECT_FALSE(salsa.stateIdxGe(salsa.STATE_BUF_SIZE-1, salsa.xbuf_head_));
    EXPECT_FALSE(salsa.stateIdxGe(salsa.xbuf_tail_, salsa.xbuf_head_));
    EXPECT_FALSE(salsa.stateIdxGe(salsa.xbuf_head_-1, salsa.xbuf_head_));
    EXPECT_FALSE(salsa.stateIdxGe(salsa.xbuf_tail_, salsa.xbuf_tail_+1));

    salsa.STATE_BUF_SIZE = 1000;
    salsa.xbuf_head_ = 7;
    salsa.xbuf_tail_ = 986;

    EXPECT_TRUE(salsa.stateIdxGe(5, 994));
}
