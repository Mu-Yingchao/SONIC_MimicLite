#include "imu_driver.h"

#include <map>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/case_conv.hpp>


namespace legged{


ImuDriver::~ImuDriver()
{

}
bool ImuDriver::init(std::string port)
{
    int ret =0;
    port_ = port;
    pthread_mutex_init(&wr_mutex,NULL);
    if(initSerial(port_) == true)
        printf("init imu Serial success\n");

    serialread_thread_ = std::thread(&ImuDriver::read_thread_func, this);

  sched_param sched{ .sched_priority = 94};
  if (pthread_setschedparam(serialread_thread_.native_handle(), SCHED_FIFO, &sched) != 0)
  {
    printf(
        "Failed to set threads priority (one possible reason could be that the user and the group permissions "
        "are not set properly.).\n");
  }
  
  
 
   
    return true;

}
const ImuRcData  ImuDriver::getimudata()
{
   ImuRcData data;
   memset(&data,0,sizeof(data));
   pthread_mutex_lock(&wr_mutex);
   memcpy(&data,&imudata_,sizeof(ImuRcData));
   pthread_mutex_unlock(&wr_mutex);
   return data; 

}
int ImuDriver::port_read(int fd,unsigned char *buf,int len)
{
    int bytes_read = read(fd,buf,len);
    if(bytes_read == -1)
    {
        //perror("read failed");
        return -1;
    }
    return bytes_read;
}

void ImuDriver::read_thread_func() 
{   
    fd_set readfds;
    int ret;
    int bytes_read;
  
    unsigned short cnt = 0;
    int pos = 0;
    memset(read_buf,0,sizeof(read_buf));
    while(1)
    {

        bytes_read = port_read(serial_,read_buf,sizeof(read_buf));
        //printf("read %d \n",bytes_read);
            
        if(bytes_read >0)
        {
            memcpy(g_recv_buf + g_recv_buf_idx, read_buf, bytes_read);             
            g_recv_buf_idx += bytes_read;

        }
        cnt = g_recv_buf_idx;
        pos = 0;
        if(cnt < YIS_OUTPUT_MIN_BYTES)
        {
            usleep(10000);
            //printf("data < yis out min bytes\n");
            continue;
        }
        while(cnt > (unsigned int)0)
        {
            int ret = analysis_data(g_recv_buf + pos, cnt, &g_output_info);
            if(analysis_done == ret)	/*未查找到帧头*/
            {
                pos++;
                cnt--;
            }
            else if(data_len_err == ret)
            {
                break;
            }
            else if(crc_err == ret || analysis_ok == ret)	 /*删除已解析完的完整一帧*/
            {
                output_data_header_t *header = (output_data_header_t *)(g_recv_buf + pos);
                unsigned int frame_len = header->len + YIS_OUTPUT_MIN_BYTES;
                cnt -= frame_len;
                pos += frame_len;
                //memcpy(g_recv_buf, g_recv_buf + pos, cnt);

                if(analysis_ok == ret)
                {
                    pthread_mutex_lock(&wr_mutex);
                    imudata_.acc_x = g_output_info.accel.x;
                    imudata_.acc_y = g_output_info.accel.y;
                    imudata_.acc_z = g_output_info.accel.z;
                    imudata_.gyr_x = g_output_info.angle_rate.x / 57.325;
                    imudata_.gyr_y = g_output_info.angle_rate.y / 57.325;
                    imudata_.gyr_z = g_output_info.angle_rate.z / 57.325;
                    imudata_.q0 = g_output_info.attitude.quaternion_data0; 
                    imudata_.q1 = g_output_info.attitude.quaternion_data1; 
                    imudata_.q2 = g_output_info.attitude.quaternion_data2;
                    imudata_.q3 = g_output_info.attitude.quaternion_data3;
                    // printf("acc x: %f, acc y: %f, acc z: %f\n", 
                    // g_output_info.accel.x , g_output_info.accel.y , g_output_info.accel.z);
                    pthread_mutex_unlock(&wr_mutex);
                }
            }
        }
        memcpy(g_recv_buf, g_recv_buf + pos, cnt);
        g_recv_buf_idx = cnt;
        tcflush(serial_,TCIFLUSH);
        usleep(10000);

    }
   
  
        

   


}


bool ImuDriver::initSerial(std::string port)
{
    int startflag = true;
    struct termios oldtio,newtio;
    //printf(" set serial opt success\n");
    speed_t speed = B921600;
    serial_ = open(port.c_str(), O_RDWR | O_NONBLOCK| O_NOCTTY | O_NDELAY); 
    if (serial_ < 0)	{
        printf("Can't Open Serial Port!\n");
        //exit(0);	
        return false;
    }
	
    printf("open serial port to decode msg!\n");

    //save to oldtio
    tcgetattr(serial_, &oldtio);
    bzero(&newtio, sizeof(newtio));
    newtio.c_cflag = speed | CS8 | CLOCAL | CREAD;
    newtio.c_cflag &= ~CSTOPB;
    newtio.c_cflag &= ~PARENB;
    newtio.c_iflag = IGNPAR;  
    newtio.c_oflag = 0;
    tcflush(serial_,TCIFLUSH);  
    tcsetattr(serial_,TCSAFLUSH,&newtio);  
    tcgetattr(serial_,&oldtio);

    return true;

  
    
   
}







}
