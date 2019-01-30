#include <gtest/gtest.h>
#include <ceres/ceres.h>

#include "multirotor_sim/simulator.h"
#include "multirotor_sim/controller.h"

#include "salsa/salsa.h"

TEST (SalsaIndexing, CheckFirstNode)
{
  Salsa salsa;

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
      EXPECT_TRUE(salsa.mocap_[i].active_);
    }
    else
    {
      EXPECT_FALSE(salsa.mocap_[i].active_);
    }
  }
  EXPECT_EQ(salsa.current_node_, 0);
  EXPECT_EQ(salsa.x_idx_, 1);
  EXPECT_EQ(salsa.imu_idx_, 0);
  EXPECT_EQ(salsa.mocap_idx_, 1);
  EXPECT_EQ(salsa.mocap_[0].x_idx_, 0);
  EXPECT_EQ(salsa.imu_[0].from_idx_, 0);
}

TEST (SalsaIndexing, CheckSecondNode)
{
  Salsa salsa;

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
      EXPECT_TRUE(salsa.imu_[i].active_);
      EXPECT_TRUE(salsa.mocap_[i].active_);
    }
    else if (i == 1)
    {
      EXPECT_FALSE(salsa.imu_[i].active_);
      EXPECT_TRUE(salsa.mocap_[i].active_);
    }
    else
    {
      EXPECT_FALSE(salsa.imu_[i].active_);
      EXPECT_FALSE(salsa.mocap_[i].active_);
    }
  }

  EXPECT_EQ(salsa.current_node_, 1);
  EXPECT_EQ(salsa.x_idx_, 2);
  EXPECT_EQ(salsa.imu_idx_, 1);
  EXPECT_EQ(salsa.mocap_idx_, 2);
  EXPECT_EQ(salsa.mocap_[0].x_idx_, 0);
  EXPECT_EQ(salsa.imu_[0].from_idx_, 0);
  EXPECT_EQ(salsa.mocap_[1].x_idx_, 1);
  EXPECT_EQ(salsa.imu_[1].from_idx_, 1);
}

TEST (SalsaIndexing, CheckWindowWrap)
{
  Salsa salsa;

  Simulator sim;
  sim.load("../lib/multirotor_sim/params/sim_params.yaml");

  sim.register_estimator(&salsa);

  while (salsa.current_node_ < Salsa::N-1)
  {
    sim.run();
  }

  EXPECT_EQ(salsa.current_node_, Salsa::N-1);
  EXPECT_EQ(salsa.mocap_idx_, 0);
  EXPECT_EQ(salsa.x_idx_, 0);
  EXPECT_EQ(salsa.imu_idx_, Salsa::N-1);
  for (int i = 0; i < Salsa::N; i++)
  {
    if (i == Salsa::N-1)
      EXPECT_FALSE(salsa.imu_[i].active_);
    else
      EXPECT_TRUE(salsa.imu_[i].active_);

    EXPECT_TRUE(salsa.mocap_[i].active_);

    EXPECT_EQ(salsa.mocap_[i].x_idx_, i);
    EXPECT_EQ(salsa.imu_[i].from_idx_, i);
  }
}

TEST (SalsaIndexing, CheckWindowWrapPlus)
{
  Salsa salsa;

  Simulator sim;
  sim.load("../lib/multirotor_sim/params/sim_params.yaml");

  sim.register_estimator(&salsa);

  while (salsa.current_node_ < Salsa::N+3)
  {
    sim.run();
  }

  EXPECT_EQ(salsa.current_node_, Salsa::N+3);
  EXPECT_EQ(salsa.mocap_idx_, 4);
  EXPECT_EQ(salsa.x_idx_, 4);
  EXPECT_EQ(salsa.imu_idx_, 3);
  for (int i = 0; i < Salsa::N; i++)
  {
    if (i == 3)
      EXPECT_FALSE(salsa.imu_[i].active_);
    else
      EXPECT_TRUE(salsa.imu_[i].active_);

    EXPECT_TRUE(salsa.mocap_[i].active_);

    EXPECT_EQ(salsa.mocap_[i].x_idx_, i);
    EXPECT_EQ(salsa.imu_[i].from_idx_, i);
  }
}
