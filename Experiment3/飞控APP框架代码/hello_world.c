#include <stdint.h>

#define PACKET_SIZE 32
#define ACTION_SIZE 1

uint8_t rx_buffer[PACKET_SIZE];
float act_buffer[ACTION_SIZE];

union my_union{
    float f;
    uint8_t byte[sizeof(float)]; 
};
typedef union my_union FLOAT2BYTES;

void uart_read(uint8_t* packet, uint32_t size) {
    // TODO: read uart message
    for(int i = 0;i<size;i++){
        if(!uart1GetDataWithDefaultTimeout(&packet[i]))
        DEBUG_PRINT("UART_READ ERROR");
    }
}

void packet_parse(uint8_t* packet, float* action) {
    // TODO: process packet
    FLOAT2BYTES tmp;
    for(int i = 0;i<sizeof(float);i++){
        tmp.byte[i] = packet[i];
    }
    action[0] = tmp.f;
}

void do_something(float* action) {
    // TODO: just use debug print here
    DEBUG_PRINT("ACTION = %f",action[0]);
    ledClearAll();
    int state = 1;
    if(action[0]<0.5){
        //left on, right off
        ledSet(LED_GREEN_L, state);
        ledSet(LED_GREEN_R, !state);
    }
    else{
        //left off, right on
        ledSet(LED_GREEN_R, state);
        ledSet(LED_GREEN_L, !state);
    }
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