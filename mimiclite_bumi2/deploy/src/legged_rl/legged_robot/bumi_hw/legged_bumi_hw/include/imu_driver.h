
#include <boost/circular_buffer.hpp>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <thread>
//#include "analysis_data.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <string.h>
#include <sys/select.h>
#include <errno.h>
#include <termios.h>
#include <chrono>
#include <pthread.h>

#include "analysis_data.h"
#include "ImuRc.h"
namespace legged{


class ImuDriver{
  
public:
    ImuDriver()=default;
    ~ImuDriver();

    bool init(std::string port);
    const ImuRcData  getimudata();
protected:
    bool initSerial(std::string port);
    void read_thread_func(); 
  
    int port_read(int fd,unsigned char *buf,int len);
   
private:
    
    std::string port_;                      //串口端口
	int baudrate_;                          //波特率
    unsigned char read_buf[512];
    int  serial_; 
  
    pthread_mutex_t wr_mutex;
    std::string data_;

    int buffer_size_;
    unsigned char g_recv_buf[512] = {0};
    unsigned short g_recv_buf_idx = 0;
    protocol_info_t g_output_info = {0};
    ImuRcData imudata_;
        
    //State machine variables for spinOnce
    int bytes_;
 

    //数据处理线程

    std::thread serialread_thread_;
};

}
