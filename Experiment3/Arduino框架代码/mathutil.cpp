#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"

void relu(int col, float *input) {
    for (int i = 0; i < col; i++) {
        input[i] = input[i] > 0 ? input[i] : 0;
    }
}

void tanh_(int col, float *input) {
    float x;
    for (int i = 0; i < col; i++) {
        x = input[i];
        if (x > 5) {
            input[i] = 1.0;
        }
        else if (x < -5) {
            input[i] = -1.0;
        }
        else {
            input[i] = tanh(x);
        }
    }
}

void matmul_1d(int row, int col, float *input, float **weights, float *output) {
    for (int i = 0; i < row; i++) {
        output[i] = 0;
        for (int j = 0; j < col; j++) {
            output[i] += input[j] * weights[i][j];
        }
    }
}

void add_bias(int col, float *input, float *bias) {
    for (int i = 0; i < col; i++) {
        input[i] += bias[i];
    }
}

void activate(int col, float *input, const char *act_fun) {
    if (strcmp(act_fun, "relu") == 0) {
        relu(col, input);
    }
    else if(strcmp(act_fun, "line") == 0) {
        return;
    }
    else if(strcmp(act_fun, "tanh") == 0) {
        tanh_(col, input);
    }
    else {
        // printf("Unsupported activation function: %s\n", act_fun);
    }
}

void softmax(float *input, float *output, int size) {
    float max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    float sum_exp = 0.0;
    for (int i = 0; i < size; i++) {
        output[i] = exp(input[i] - max_val);  // avoid overflow
        sum_exp += output[i];
    }

    for (int i = 0; i < size; i++) {
        output[i] /= sum_exp;
    }
}

float mean(float *data, int size) {
    float sum = 0;
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }
    return sum / size;
}

float std_(float *data, int size) {
    float mean_ = mean(data, size);
    float diff_sum = 0;

    for (int i = 0; i < size; i++) {
        float diff = data[i] - mean_;
        diff_sum += diff * diff;
    }

    return sqrt(diff_sum / size);
}