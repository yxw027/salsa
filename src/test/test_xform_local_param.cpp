#include <gtest/gtest.h>

#include "factors/xform.h"
#include "salsa/test_common.h"


TEST (XformLocalParam, ADPlusAnalytical)
{
    XformPlus ad;
    XformParam an;

    for (int i = 0; i < 1000; i++)
    {
        xform::Xformd x1 = xform::Xformd::Random();
        Vector6d dx = Vector6d::Random();

        xform::Xformd x2_ad = xform::Xformd::Identity();
        xform::Xformd x2_an = xform::Xformd::Identity();

        ad(x1.data(), dx.data(), x2_ad.data());
        an.Plus(x1.data(), dx.data(), x2_an.data());

        EXPECT_MAT_NEAR(x2_ad.arr(), x2_an.arr(), 1e-14);
    }
}

TEST (XformLocalParam, ADvsAnalyticalJac)
{
    XformParamAD ad(new XformPlus());
    XformParam an;

    for (int i = 0; i < 1000; i++)
    {
        xform::Xformd x1 = xform::Xformd::Random();
        Vector6d dx = Vector6d::Random();

        Eigen::Matrix<double, 7, 6, Eigen::RowMajor> Jad, Jan;
        Jad.setZero();
        Jan.setZero();

        ad.ComputeJacobian(x1.data(), Jad.data());
        an.ComputeJacobian(x1.data(), Jan.data());

        EXPECT_MAT_NEAR(Jan, Jad, 1e-14);
    }
}
