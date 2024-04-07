#include <stdint.h>

#define PACKET_SIZE 32
#define ACTION_SIZE 1

uint8_t rx_buffer[PACKET_SIZE];
float act_buffer[ACTION_SIZE];

void uart_read(uint8_t* packet, uint32_t size) {
    // TODO: read uart message
}

void packet_parse(uint8_t* packet, float* action) {
    // TODO: process packet
}

void do_something(float* action) {
    // TODO: just use debug print here
}


void appMain() {
    uart1Init(115200);
    DEBUG_PRINT("APP START!");

    while (1) {
        uart_read(rx_buffer, PACKET_SIZE);

        // TODO: parse the messages received through UART, and extract the actions. 
        packet_parse(rx_buffer, act_buffer);

        // TODO:  call the corresponding control interface based on the action.
        do_something(act_buffer);
    }
}