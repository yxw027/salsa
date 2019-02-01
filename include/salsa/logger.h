#pragma once

#include <cstdint>
#include <deque>
#include <mutex>
#include <fstream>
#include <thread>
#include <unistd.h>
#include <iostream>
#include <Eigen/Dense>

#include "salsa/salsa.h"

#define LOGGER_BUFFER_SIZE 1024*1024

using namespace std;
using namespace Eigen;

class Logger
{
public:
    Logger(std::string filename)
    {
        file_.open(filename);
    }

    ~Logger()
    {
        file_.close();
    }
    template <typename... T>
    void log(T... data)
    {
        int dummy[sizeof...(data)] = { (file_.write((char*)&data, sizeof(T)), 1)... };
    }

    template <typename... T>
    void logVectors(T... data)
    {
        int dummy[sizeof...(data)] = { (file_.write((char*)data.data(), sizeof(typename T::Scalar)*data.rows()*data.cols()), 1)... };
    }

private:
    std::ofstream file_;
};


class MTLogger
{
  enum
  {
    N = LOGGER_BUFFER_SIZE
  };

public:
  MTLogger(std::string filename)
  {
    buffer_ = new char[N];
    file_.open(filename);
    if (!file_.is_open())
      cerr << "unable to open " << filename << endl;

    buf_head_ = buf_tail_ = 0;
    buf_free_ = N;
    shutdown_ = false;
    write_thread = new thread([this](){ this->writeThread(); });
  }

  ~MTLogger()
  {
    close();
  }

  void close()
  {
    closeWriteThread();
    file_.close();
    if (buffer_)
    {
      delete buffer_;
      buffer_ = nullptr;
    }
  }

  void closeWriteThread()
  {
    shutdown_ = true;
    if (write_thread)
    {
      write_thread->join();
      delete write_thread;
      write_thread = nullptr;
    }
  }

  void bufferData(char* addr, size_t size)
  {
    if (size > buf_free_)
    {
      cerr << "Buffer Overflow" << endl;
      exit(2);
    }


    if (size < N - buf_head_)
    {
      // data will fit on end of buffer
      memcpy(&buffer_[buf_head_], addr, size);
      mtx_.lock();
      buf_head_ += size;
      buf_free_ -= size;
      mtx_.unlock();
    }
    else
    {
      // copy the first bit to the end of the buffer
      int n = N - buf_head_;
      memcpy(&buffer_[buf_head_], addr, n);
      mtx_.lock();
      buf_head_ = 0;
      buf_free_ -= n;
      mtx_.unlock();

      // copy the rest to the beginning
      n = size - n;
      memcpy(&buffer_[buf_head_], &addr[n], n);
      mtx_.lock();
      buf_head_ = n;
      buf_free_ -= n;
      mtx_.unlock();
    }
  }

  void writeThread()
  {
    while(1)
    {
      if (!writeData())
      {
        if (shutdown_)
            break;
        usleep(1000);
      }
    }
  }

  int writeData()
  {
    if (buf_head_ > buf_tail_)
    {
      int n = buf_head_ - buf_tail_;
      file_.write(&buffer_[buf_tail_], n);
      mtx_.lock();
      buf_tail_ = buf_head_;
      buf_free_ = N;
      mtx_.unlock();
      return 1;
    }
    else if (buf_tail_ > buf_head_)
    {
      int n = N - buf_tail_;
      file_.write(&buffer_[buf_tail_], n);
      mtx_.lock();
      buf_tail_ = 0;
      buf_free_ += n;
      mtx_.unlock();

      n = buf_head_ - buf_tail_;
      file_.write(buffer_, n);
      mtx_.lock();
      buf_tail_ = buf_head_;
      buf_free_ += n;
      mtx_.unlock();
      return -1;
    }
    else
    {
      return 0;
    }
  }

  template <typename... T>
  void log(T... data)
  {
    int dummy[sizeof...(data)] = { (bufferData((char*)&data, sizeof(T)), 1)... };
  }

  template <typename... T>
  void logVectors(T... data)
  {
    int dummy[sizeof...(data)] = { (bufferData((char*)data.data(), sizeof(typename T::Scalar)*data.rows()*data.cols()), 1)... };
  }

  std::mutex mtx_;
  ofstream file_;
  bool shutdown_;
  char* buffer_;
  int buf_head_, buf_tail_;
  int buf_free_;
  thread* write_thread;
};


