#include <fstream>

#include "Eigen/Core"

#include "salsa/logger.h"
#include "salsa/test_common.h"

using namespace Eigen;
using namespace std;

TEST (Logger, SingleThread)
{
  std::string filename = "/tmp/Logger.dat";
  Logger<double> log(filename);
  const int N = 256;

  MatrixXd data1(N, N), data2(N, N), data3(N, N), data4(N,N), data5(N,N);
  data1.setRandom();
  data2.setRandom();
  data3.setRandom();
  data4.setRandom();
  data5.setRandom();

  log.closeWriteThread();

  log.logVectors(data1);
  EXPECT_EQ(log.buf_free_, LOGGER_BUFFER_SIZE - sizeof(double)*N*N);
  EXPECT_EQ(log.buf_head_, sizeof(double)*N*N);
  EXPECT_EQ(log.buf_tail_, 0);
  log.writeData();
  EXPECT_EQ(log.buf_free_, LOGGER_BUFFER_SIZE);
  EXPECT_EQ(log.buf_head_, LOGGER_BUFFER_SIZE - sizeof(double)*N*N);
  EXPECT_EQ(log.buf_tail_, LOGGER_BUFFER_SIZE - sizeof(double)*N*N);
  log.logVectors(data2);
  EXPECT_EQ(log.buf_free_, LOGGER_BUFFER_SIZE - sizeof(double)*N*N);
  EXPECT_EQ(log.buf_head_, 2*sizeof(double)*N*N - LOGGER_BUFFER_SIZE);
  EXPECT_EQ(log.buf_tail_, LOGGER_BUFFER_SIZE - sizeof(double)*N*N);
  log.writeData();
  EXPECT_EQ(log.buf_free_, LOGGER_BUFFER_SIZE);
  EXPECT_EQ(log.buf_head_, 2*sizeof(double)*N*N - LOGGER_BUFFER_SIZE);
  EXPECT_EQ(log.buf_tail_, 2*sizeof(double)*N*N - LOGGER_BUFFER_SIZE);
  log.logVectors(data3);
  log.writeData();
  log.logVectors(data4);
  log.writeData();
  log.logVectors(data5);
  log.writeData();

  log.close();

  ifstream file(filename, std::ios::in | std::ios::binary);
  MatrixXd data1_in(N,N), data2_in(N,N), data3_in(N,N), data4_in(N,N), data5_in(N,N);

  file.read((char*)data1_in.data(), sizeof(double)*N*N);
  file.read((char*)data2_in.data(), sizeof(double)*N*N);
  file.read((char*)data3_in.data(), sizeof(double)*N*N);
  file.read((char*)data4_in.data(), sizeof(double)*N*N);
  file.read((char*)data5_in.data(), sizeof(double)*N*N);

  EXPECT_MAT_NEAR(data1, data1_in, 1e-14);
  EXPECT_MAT_NEAR(data2, data2_in, 1e-14);
  EXPECT_MAT_NEAR(data3, data3_in, 1e-14);
  EXPECT_MAT_NEAR(data4, data4_in, 1e-14);
  EXPECT_MAT_NEAR(data5, data5_in, 1e-14);
}


TEST (Logger, MultiThread)
{
  std::string filename = "/tmp/Logger.dat";
  Logger<double> log(filename);
  const int N = 256;

  MatrixXd data1(N, N), data2(N, N), data3(N, N), data4(N,N), data5(N,N);
  data1.setRandom();
  data2.setRandom();
  data3.setRandom();
  data4.setRandom();
  data5.setRandom();

  log.logVectors(data1);
  usleep(1000);
  log.logVectors(data2);
  usleep(1000);
  log.logVectors(data3);
  usleep(1000);
  log.logVectors(data4);
  usleep(1000);
  log.logVectors(data5);

  log.close();

  ifstream file(filename, std::ios::in | std::ios::binary);
  MatrixXd data1_in(N,N), data2_in(N,N), data3_in(N,N), data4_in(N,N), data5_in(N,N);

  file.read((char*)data1_in.data(), sizeof(double)*N*N);
  file.read((char*)data2_in.data(), sizeof(double)*N*N);
  file.read((char*)data3_in.data(), sizeof(double)*N*N);
  file.read((char*)data4_in.data(), sizeof(double)*N*N);
  file.read((char*)data5_in.data(), sizeof(double)*N*N);

  EXPECT_MAT_NEAR(data1, data1_in, 1e-14);
  EXPECT_MAT_NEAR(data2, data2_in, 1e-14);
  EXPECT_MAT_NEAR(data3, data3_in, 1e-14);
  EXPECT_MAT_NEAR(data4, data4_in, 1e-14);
  EXPECT_MAT_NEAR(data5, data5_in, 1e-14);
}
