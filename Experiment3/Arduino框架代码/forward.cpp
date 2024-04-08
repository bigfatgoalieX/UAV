static int n = 1;
static int weights_shape[] = {1, 2};
static const char *act_funcs[] = {"line"};
static float __weights_1d[] = {0.500000, 0.500000};
static float __bias_1d[] = {0.000000};
static float *__w0[1] = {&__weights_1d[0]};
static float **weights[1] = {__w0};
static float *bias[1] = {&__bias_1d[0]};
static float inter_result[1];
static float prev_output[1];
static float output[1];

extern void relu(int col, float *input);
extern void tanh_(int col, float *input);
extern void matmul_1d(int row, int col, float *input, float **weights, float *output);
extern void add_bias(int col, float *input, float *bias);
extern void activate(int col, float *input, const char *act_fun);

float *forward(float *input) {
    int row = weights_shape[0], col = weights_shape[1];
    matmul_1d(row, col, input, weights[0], prev_output);
    add_bias(row, prev_output, bias[0]);
    activate(row, prev_output, act_funcs[0]);
    
    for (int i = 1; i < n; i++) {
        int row = weights_shape[2 * i], col = weights_shape[2 * i + 1];

        matmul_1d(row, col, prev_output, weights[i], inter_result);
        add_bias(row, inter_result, bias[i]);
        activate(row, inter_result, act_funcs[i]);

        for (int j = 0; j < row; j++) {
            prev_output[j] = inter_result[j];
        }
    }

    for(int i = 0; i < weights_shape[2 * (n - 1)]; i++) {
        output[i] = prev_output[i];
    }

    return output;
}