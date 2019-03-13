#include <fstream>

#include "Eigen/Core"

#include "salsa/logger.h"
#include "salsa/test_common.h"

using namespace Eigen;
using namespace std;

TEST (MTLogger, SingleThread)
{
  std::string filename = "/tmp/Logger.dat";
  salsa::MTLogger log(filename);
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

  EXPECT_NEAR((data1 - data1_in).array().abs().sum(), 0.0, 1e-14);
  EXPECT_NEAR((data2 - data2_in).array().abs().sum(), 0.0, 1e-14);
  EXPECT_NEAR((data3 - data3_in).array().abs().sum(), 0.0, 1e-14);
  EXPECT_NEAR((data4 - data4_in).array().abs().sum(), 0.0, 1e-14);
  EXPECT_NEAR((data5 - data5_in).array().abs().sum(), 0.0, 1e-14);
}


TEST (MTLogger, MultiThread)
{
  std::string filename = "/tmp/Logger.dat";
  salsa::MTLogger log(filename);
  const int N = 256;

  MatrixXd data1(N, N), data2(N, N), data3(N, N), data4(N,N), data5(N,N);
  data1.setRandom();
  data2.setRandom();
  data3.setRandom();
  data4.setRandom();
  data5.setRandom();

  log.logVectors(data1);
  usleep(2000);
  log.logVectors(data2);
  usleep(2000);
  log.logVectors(data3);
  usleep(2000);
  log.logVectors(data4);
  usleep(2000);
  log.logVectors(data5);

  log.close();

  ifstream file(filename, std::ios::in | std::ios::binary);
  MatrixXd data1_in(N,N), data2_in(N,N), data3_in(N,N), data4_in(N,N), data5_in(N,N);

  file.read((char*)data1_in.data(), sizeof(double)*N*N);
  file.read((char*)data2_in.data(), sizeof(double)*N*N);
  file.read((char*)data3_in.data(), sizeof(double)*N*N);
  file.read((char*)data4_in.data(), sizeof(double)*N*N);
  file.read((char*)data5_in.data(), sizeof(double)*N*N);

  EXPECT_NEAR((data1 - data1_in).array().abs().sum(), 0.0, 1e-14);
  EXPECT_NEAR((data2 - data2_in).array().abs().sum(), 0.0, 1e-14);
  EXPECT_NEAR((data3 - data3_in).array().abs().sum(), 0.0, 1e-14);
  EXPECT_NEAR((data4 - data4_in).array().abs().sum(), 0.0, 1e-14);
  EXPECT_NEAR((data5 - data5_in).array().abs().sum(), 0.0, 1e-14);
}


TEST (MTLogger, Different_Types)
{
    std::string filename = "/tmp/Logger.dat";
    salsa::MTLogger log(filename);

    int a = 7;
    double b = 28.02;
    float c = 3928.29;
    typedef struct
    {
      float d;
      uint32_t e;
      int16_t f;
    } thing_t;
    thing_t g;
    g.d = 1983.23;
    g.e = 239102930;
    g.f = 28395;

    log.log(a, b, c, g);

    log.close();

    ifstream file(filename, std::ios::in | std::ios::binary);
    int a_in;
    double b_in;
    float c_in;
    thing_t g_in;

    file.read((char*)&a_in, sizeof(a_in));
    file.read((char*)&b_in, sizeof(b_in));
    file.read((char*)&c_in, sizeof(c_in));
    file.read((char*)&g_in, sizeof(g_in));

    EXPECT_EQ(a_in, a);
    EXPECT_EQ(b_in, b);
    EXPECT_EQ(c_in, c);
    EXPECT_EQ(g_in.d, g.d);
    EXPECT_EQ(g_in.e, g.e);
    EXPECT_EQ(g_in.f, g.f);
}
