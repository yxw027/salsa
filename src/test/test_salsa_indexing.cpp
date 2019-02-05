#include <gtest/gtest.h>
#include <ceres/ceres.h>

#include "multirotor_sim/simulator.h"
#include "multirotor_sim/controller.h"
#include "geometry/support.h"

#include "salsa/salsa.h"
#include "salsa/test_common.h"

TEST (SalsaIndexing, CheckFirstNode)
{
  Salsa salsa;
  salsa.init("../params/salsa.yaml");

  Simulator sim;
  sim.load("../lib/multirotor_sim/params/sim_params.yaml");

  sim.register_estimator(&salsa);

  while (salsa.current_node_ == -1)
  {
    sim.run();
  }

  for (int i = 0; i < Salsa::N; i++)
  {
    EXPECT_FALSE(salsa.imu_[i].active_);
    if (i == 0)
    {
      EXPECT_TRUE(salsa.initialized_[i]);
      EXPECT_TRUE(salsa.mocap_[i].active_);
      EXPECT_MAT_FINITE(salsa.x_.col(i));
      EXPECT_MAT_FINITE(salsa.v_.col(i));
      EXPECT_MAT_FINITE(salsa.tau_.col(i));
    }
    else
    {
      EXPECT_FALSE(salsa.initialized_[i]);
      EXPECT_FALSE(salsa.mocap_[i].active_);
      EXPECT_MAT_NAN(salsa.x_.col(i));
      EXPECT_MAT_NAN(salsa.v_.col(i));
      EXPECT_MAT_NAN(salsa.tau_.col(i));
    }
  }
  EXPECT_MAT_FINITE(salsa.imu_bias_);

  EXPECT_EQ(salsa.current_node_, 0);
  EXPECT_FINITE(salsa.current_t_);
  EXPECT_MAT_FINITE(salsa.current_x_.arr());
  EXPECT_MAT_FINITE(salsa.current_v_);

  EXPECT_EQ(salsa.x_idx_, 0);
  EXPECT_EQ(salsa.mocap_[0].x_idx_, 0);
  EXPECT_EQ(salsa.imu_[0].from_idx_, 0);
}

TEST (SalsaIndexing, CheckSecondNode)
{
  Salsa salsa;
  salsa.init("../params/salsa.yaml");

  Simulator sim;
  sim.load("../lib/multirotor_sim/params/sim_params.yaml");

  sim.register_estimator(&salsa);

  while (salsa.current_node_ < 1)
  {
    sim.run();
  }

  for (int i = 0; i < Salsa::N; i++)
  {
    if (i == 0)
    {
      EXPECT_TRUE(salsa.initialized_[i]);
      EXPECT_TRUE(salsa.imu_[i].active_);
      EXPECT_TRUE(salsa.mocap_[i].active_);
      EXPECT_MAT_FINITE(salsa.x_.col(i));
      EXPECT_MAT_FINITE(salsa.v_.col(i));
      EXPECT_MAT_FINITE(salsa.tau_.col(i));
    }
    else if (i == 1)
    {
      EXPECT_TRUE(salsa.initialized_[i]);
      EXPECT_FALSE(salsa.imu_[i].active_);
      EXPECT_TRUE(salsa.mocap_[i].active_);
      EXPECT_MAT_FINITE(salsa.x_.col(i));
      EXPECT_MAT_FINITE(salsa.v_.col(i));
      EXPECT_MAT_FINITE(salsa.tau_.col(i));
    }
    else
    {
      EXPECT_FALSE(salsa.initialized_[i]);
      EXPECT_FALSE(salsa.imu_[i].active_);
      EXPECT_FALSE(salsa.mocap_[i].active_);
      EXPECT_MAT_NAN(salsa.x_.col(i));
      EXPECT_MAT_NAN(salsa.v_.col(i));
      EXPECT_MAT_NAN(salsa.tau_.col(i));
    }
  }

  EXPECT_MAT_FINITE(salsa.x_.col(1));
  EXPECT_MAT_FINITE(salsa.v_.col(1));
  EXPECT_MAT_FINITE(salsa.tau_.col(1));
  EXPECT_MAT_FINITE(salsa.imu_bias_);

  EXPECT_EQ(salsa.current_node_, 1);
  EXPECT_FINITE(salsa.current_t_);
  EXPECT_MAT_FINITE(salsa.current_x_.arr());
  EXPECT_MAT_FINITE(salsa.current_v_);

  EXPECT_EQ(salsa.current_node_, 1);
  EXPECT_EQ(salsa.x_idx_, 1);
  EXPECT_EQ(salsa.mocap_[0].x_idx_, 0);
  EXPECT_EQ(salsa.imu_[0].from_idx_, 0);
  EXPECT_EQ(salsa.mocap_[1].x_idx_, 1);
  EXPECT_EQ(salsa.imu_[1].from_idx_, 1);
}

TEST (SalsaIndexing, CheckWindowWrap)
{
  Salsa salsa;
  salsa.init("../params/salsa.yaml");

  Simulator sim;
  sim.load("../lib/multirotor_sim/params/sim_params.yaml");

  sim.register_estimator(&salsa);

  while (salsa.current_node_ < Salsa::N)
  {
    sim.run();
  }

  EXPECT_EQ(salsa.current_node_, Salsa::N);
  EXPECT_EQ(salsa.x_idx_, 0);
  for (int i = 0; i < Salsa::N; i++)
  {
    if (i == 0)
      EXPECT_FALSE(salsa.imu_[i].active_);
    else
      EXPECT_TRUE(salsa.imu_[i].active_);

    EXPECT_TRUE(salsa.mocap_[i].active_);

    EXPECT_EQ(salsa.mocap_[i].x_idx_, i);
    EXPECT_EQ(salsa.imu_[i].from_idx_, i);
    EXPECT_MAT_FINITE(salsa.x_.col(i));
    EXPECT_MAT_FINITE(salsa.v_.col(i));
    EXPECT_MAT_FINITE(salsa.tau_.col(i));
  }
  EXPECT_FINITE(salsa.current_t_);
  EXPECT_MAT_FINITE(salsa.current_x_.arr());
  EXPECT_MAT_FINITE(salsa.current_v_);
}

TEST (SalsaIndexing, CheckWindowWrapPlus)
{
  Salsa salsa;
  salsa.init("../params/salsa.yaml");

  Simulator sim;
  sim.load("../lib/multirotor_sim/params/sim_params.yaml");

  sim.register_estimator(&salsa);

  while (salsa.current_node_ < Salsa::N+3)
  {
    sim.run();
  }

  EXPECT_EQ(salsa.current_node_, Salsa::N+3);
  EXPECT_EQ(salsa.x_idx_, 3);
  for (int i = 0; i < Salsa::N; i++)
  {
    if (i == 3)
      EXPECT_FALSE(salsa.imu_[i].active_);
    else
      EXPECT_TRUE(salsa.imu_[i].active_);

    EXPECT_TRUE(salsa.mocap_[i].active_);

    EXPECT_EQ(salsa.mocap_[i].x_idx_, i);
    EXPECT_EQ(salsa.imu_[i].from_idx_, i);
    EXPECT_MAT_FINITE(salsa.x_.col(i));
    EXPECT_MAT_FINITE(salsa.v_.col(i));
    EXPECT_MAT_FINITE(salsa.tau_.col(i));
  }
  EXPECT_FINITE(salsa.current_t_);
  EXPECT_MAT_FINITE(salsa.current_x_.arr());
  EXPECT_MAT_FINITE(salsa.current_v_);
}

TEST (SalsaIndexing, CurrentStateAlwaysValid)
{
  Salsa salsa;
  salsa.init("../params/salsa.yaml");

  Simulator sim;
  sim.load("../lib/multirotor_sim/params/sim_params.yaml");

  sim.register_estimator(&salsa);

  while (salsa.current_node_ < Salsa::N+3)
  {
    sim.run();
    if (salsa.current_node_ == -1)
    {
      EXPECT_TRUE(std::isnan(salsa.current_t_));
      EXPECT_MAT_NAN(salsa.current_x_.arr());
      EXPECT_MAT_NAN(salsa.current_v_);
    }
    else
    {
      EXPECT_FINITE(salsa.current_t_);
      EXPECT_MAT_FINITE(salsa.current_x_.arr());
      EXPECT_MAT_FINITE(salsa.current_v_);
    }
  }
}
