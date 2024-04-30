#include <ESP8266WiFi.h>
#include <ESP8266WiFiMulti.h>
#include <Arduino.h>
#include <WiFiUdp.h>
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "cmath"
#include "stdarg.h"

#define STASSID "HUAWEI"
#define STAPSK "a123456789"

#define UDP_PRINT_IP "192.168.31.91"
#define UDP_PRINT_PORT 59998

#define UDP_BUFFER_SIZE 1024

unsigned int localPort = 8888;
char packetBuffer[UDP_BUFFER_SIZE];  // buffer to hold incoming packet,
WiFiUDP Udp;

#define LED_PIN (2)

float *forward(float *);

const char *ssid = STASSID;
const char *password = STAPSK;

ESP8266WiFiMulti WiFiMulti;

uint8_t rxByte = 0;
uint8_t cksum = 0;

typedef union _cmd_req
{
    struct
    {
        uint8_t req;
        float vx;
        float vy;
        float vz;
    };
    uint8_t buf[13];
} __attribute__((packed)) cmd_req;

cmd_req cmdreq;

typedef union _v_data
{
    struct
    {
        float vx;
        float vy;
        float vz;
        uint32_t land;
        float vx_r;
        float vy_r;
        float vz_r;
        float w_r;
        float theta;
    };
    uint8_t buf[36];
} __attribute__((packed)) v_data;

v_data vsetpoint;

uint8_t checksum(const uint32_t size, uint8_t *packet)
{
    uint8_t cksum = 0;
    for (int i = 0; i < size; i++)
    {
        cksum += packet[i];
    }
    return cksum;
}

int32_t uart_read(const uint32_t size, uint8_t *data)
{
    Serial.readBytes(data, size);
    Serial.readBytes(&rxByte, 1);

    if (checksum(size, data) != rxByte)
    {
        // printf("Uart read: checksum dismatch !\n");
        return 0;
    }
    return 1;
}

void uart_write(const uint32_t size, uint8_t *data)
{
    cksum = checksum(size, data);
    Serial.write(data, size);
    Serial.write(&cksum, 1);
}

void udp_send(const char *ip, uint16_t port, const char *buffer, size_t size) {
    IPAddress ipaddr;
    ipaddr.fromString(ip);
    Udp.beginPacket(ipaddr, port);
    Udp.write(buffer, size);
    Udp.endPacket();
}

// void udp_print(const char *fmt, ...) {
//     static char udp_print_buffer[64];

//     va_list args;
//     va_start(args, fmt);

//     vsprintf(udp_print_buffer, fmt, args);
//     udp_send(UDP_PRINT_IP, UDP_PRINT_PORT, udp_print_buffer, strlen(udp_print_buffer) + 1);

//     va_end(args);
// }

void setup()
{
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);  // LED ON

    Serial.setTimeout(0x0fffffff);
    Serial.begin(115200);
    while (!Serial)
        ;
    //  digitalWrite(LED_PIN, LOW);

    // We start by connecting to a WiFi network
    WiFi.mode(WIFI_STA);
    WiFiMulti.addAP(ssid, password);

    Serial.println();
    Serial.println();
    Serial.print("[ESP8266]: Wait for WiFi... ");

    while (WiFiMulti.run() != WL_CONNECTED)
    {
        Serial.print(".");
        delay(500);
    }

    Serial.println("");
    Serial.println("[ESP8266]: WiFi connected");
    Serial.println("[ESP8266]: IP address: ");
    Serial.println(WiFi.localIP());

    Udp.begin(localPort);
    delay(500);
}

#define OBS_DIM (22)

typedef struct _info_type
{
    float obs[OBS_DIM];
    float p_k;
    float vel;
    float z;
} info_type;

info_type recv_info;

uint32_t step_cnt = 0;
static uint8_t buffer[64];
void loop()
{
    const float dt = 0.2;
    int packetSize = Udp.parsePacket();
    if (packetSize == 0) {
        return;
    }

    // read the packet into packetBufffer
    int n = Udp.read(packetBuffer, UDP_BUFFER_SIZE);
    packetBuffer[n] = 0;
    if (n == sizeof(info_type)) {
        memcpy((void *)(&recv_info), packetBuffer, sizeof(recv_info));
    } else {
        // send a reply, to the IP address and port that sent us the packet we received
        Udp.beginPacket(Udp.remoteIP(), Udp.remotePort());
        Udp.write("ERROR_SIZE");
        Udp.endPacket();
        return;
    }

    digitalWrite(LED_PIN, !digitalRead(LED_PIN));

    float input[OBS_DIM];
    for (int i = 0; i < OBS_DIM; ++i)
    {
        input[i] = recv_info.obs[i];
    }

    float now_z = recv_info.z; // get current height
    float abs_vel = recv_info.vel; // get absolute velocity
    float k_p = recv_info.p_k; // get K_P

    uart_read(sizeof(cmdreq.buf), cmdreq.buf);
    float theta_real = atan2(cmdreq.vy, cmdreq.vx); // read theta from the flight controller
    input[4] = theta_real;
    float *result = forward(input);

    float target_z = 0.45;

    if (cmdreq.buf[0] != 0x66)
    {
        return;
    }

    float d_theta = result[0];
    float vxy = abs_vel;
    float theta = theta_real + d_theta * dt;
    
    
    if (step_cnt < 3) {
        theta = -M_PI / 2;
    }
    step_cnt += 1;

    if (theta > M_PI)
        theta -= 2 * M_PI;
    else if (theta < -M_PI)
        theta += 2 * M_PI;

    float vx = vxy * cos(theta);
    float vy = vxy * sin(theta);
    float vz = (target_z - now_z) * k_p;

    float dist2tar2 = input[2] * input[2] + input[3] * input[3];
    uint32_t yaw_rate = -1;

    uint32_t land = 0;
    if (dist2tar2 < 1.0 * 1.0) {
        land = 1;
    }

    vsetpoint.vx = vx;
    vsetpoint.vy = vy;
    vsetpoint.vz = vz;
    vsetpoint.land = land;
    vsetpoint.vx_r = cmdreq.vx;
    vsetpoint.vy_r = cmdreq.vy;
    vsetpoint.vz_r = cmdreq.vz;
    vsetpoint.w_r = theta_real;
    vsetpoint.theta = theta;

    uart_write(sizeof(vsetpoint.buf), vsetpoint.buf);
}
