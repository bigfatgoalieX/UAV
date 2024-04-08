#include "stdio.h"
#include "stdint.h"

union my_union{
    float f;
    uint8_t byte[sizeof(float)]; 
};

typedef union my_union FLOAT2BYTES;

size_t encode_float_vals(float* fvals, int len, uint8_t* out) {
  // TODO: encode float values to bytes, return the length of out
  // for()
  // out[0] = 0xff;
  // return 4*len;

  int out_index = 0;
  for (int i = 0; i < len; i++) {
      uint32_t temp;
      *((float*)&temp) = fvals[i];
      out[out_index++] = temp & 0xff;
      out[out_index++] = (temp >> 8) & 0xff;
      out[out_index++] = (temp >> 16) & 0xff;
      out[out_index++] = (temp >> 24) & 0xff;
  }
  return out_index; 
}
void parse_packet(uint8_t* packets, int pk_len, float* obs) {
  // for (int i = 0; i < pk_len; ++i) {
  //   Serial.printf("%u, ", packets[i]);
  // }
  // Serial.println();

  // not using UNION
  // for(int i = 0;i < pk_len/4;i++){
  //   uint_32 temp = (packets[i*4 + 3] << 24) | (packets[i*4 + 2] << 16) | (packets[i*4 + 1] << 8) | packets[i*4];
  //   obs[i] = *((float*)&temp);
  // }
  
  //using UNION
  FLOAT2BYTES my_tool[100]; 
  for(int i = 0;i<pk_len/sizeof(float);i++){
    for(int j = 0;j<sizeof(float);j++){
      my_tool[i].byte[j] = packets[i*sizeof(float)+j];
    } 
  }
  for(int i = 0;i<pk_len/sizeof(float);i++){
    obs[i] = my_tool[i].f;
  }

  // TODO: parse packets into obs
}
int main(){
    // float vals[2] = {1.1, 2.2};
    // printf("vals[0] = %f\n",vals[0]);
    // printf("vals[1] = %f\n",vals[1]);
    // printf("address of vals: %x\n",(&vals));
    // printf("address of vals[0]: %x\n",(&vals[0]));
    // printf("address of vals[1]: %x\n",(&vals[1]));
    // int house = 45;
    // int* flag = &house;
    // printf("address of house: %x\n",&house);
    // printf("flag: %x\n",flag);
    // printf("address of flag: %x\n",(&flag));
    // printf("size of a pointer: %d\n",sizeof(flag));
    // int a = 1;
    // int b = 2;
    // int c = 3;
    // printf("address of a: %p\n",&a);
    // printf("address of b: %p\n",&b);
    // printf("address of c: %p\n",&c);
    uint8_t packets[8] = {0xcd, 0x0c, 0x80, 0x44, 0x66, 0x66, 0x06, 0x40};
    int pk_len = 8;
    float obs[100];
    parse_packet(packets,pk_len,obs);
    for(int i = 0;i<pk_len/sizeof(float);i++){
        printf("OBS_FLOAT:%f\n",obs[i]);
    }
}