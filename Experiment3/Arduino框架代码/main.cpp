#include <Arduino.h>
#include <WiFiUdp.h>
#include <WiFi.h>



#define STASSID "ssid"
#define STAPSK "password"
#define UDP_PORT 1234

#define OBS_DIM 2

WiFiUDP udp_server;
uint8_t udp_read_buffer[32];
uint8_t udp_write_buffer[32];
uint8_t uart_write_buffer[32];
float obs[OBS_DIM];


union my_union{
    float f;
    uint8_t byte[sizeof(float)]; 
};
typedef union my_union FLOAT2BYTES;


extern float *forward(float *input)


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

float process_obs(float* obs, int len) {
  float result = forward(obs)[0];
  // float result = 0;
  // for (int i = 0; i < len; ++i) {
  //   result += obs[i];
  // }
  // TODO: Modify the above code for neural network forward propagation and return the result

  return result;
}

size_t encode_float_vals(float* fvals, int len, uint8_t* out) {
  // TODO: encode float values to bytes, return the length of out
  // for()
  // out[0] = 0xff;
  // return 4*len;

  // not using UNION
  // int out_index = 0;
  // for (int i = 0; i < len; i++) {
  //     uint32_t temp;
  //     *((float*)&temp) = fvals[i];
  //     out[out_index++] = temp & 0xff;
  //     out[out_index++] = (temp >> 8) & 0xff;
  //     out[out_index++] = (temp >> 16) & 0xff;
  //     out[out_index++] = (temp >> 24) & 0xff;
  // }
  // return out_index;

  // using UNION
  FLOAT2BYTES tmp[100];
  int out_index = 0;
  for(int i = 0;i<len;i++){
    tmp[i].f = fvals[i];
    out[out_index++] = tmp[i].byte[0];
    out[out_index++] = tmp[i].byte[1];
    out[out_index++] = tmp[i].byte[2];
    out[out_index++] = tmp[i].byte[3];
  }
  return out_index;
}

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  WiFi.mode(WIFI_STA);
  WiFi.begin(STASSID, STAPSK);

  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(100);
  }
  Serial.printf("IP Address: %s\n", WiFi.localIP().toString().c_str());
  udp_server.begin(UDP_PORT);
}


void loop() {
  // put your main code here, to run repeatedly:
  int len = udp_server.parsePacket();
  if (len != 0) {
    IPAddress remote_ip = udp_server.remoteIP();
    uint16_t remote_port = udp_server.remotePort();
    
    size_t n = udp_server.readBytes(udp_read_buffer, 32);
    // Serial.printf("Read packet from %s:%u [%u] bytes", remote_ip.toString().c_str(), remote_port, n);
    parse_packet(udp_read_buffer, n, obs);
    
    float action = process_obs(obs, OBS_DIM);

    // TODO: send back message to the ground station (UDP -> PC)
    udp_server.beginPacket();
    n = encode_float_vals(&action, 1, udp_write_buffer);
    udp_server.write(udp_write_buffer, n);
    udp_server.endPacket();

    // TODO: send action to the flight controller by UART (Serial -> Crazyflie)
    n = encode_float_vals(&action, 1, uart_write_buffer);
    Serial.write(uart_write_buffer, n);
    // what is uart_write_buffer?
  }
}
