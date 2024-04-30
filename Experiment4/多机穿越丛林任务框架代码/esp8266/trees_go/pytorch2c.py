import torch
from torch import nn
import sys

class Net(nn.Module):

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.fc0 = torch.nn.Linear(obs_dim, 64)
        self.fc1 = torch.nn.Linear(64, 64)
        self.fc2 = torch.nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc0(x))
        x = torch.relu(self.fc1(x))
        a = torch.tanh(self.fc2(x))
        return a


net = Net(22, 1)
policy_state_dict = torch.load(sys.argv[1])
net.load_state_dict(policy_state_dict)
state_dict = net.state_dict()


def save_model_as_c(model: torch.nn, activation: list, name: str="forward"):
    state_dict = model.state_dict()
    n_layers = len(state_dict) // 2
    codes = [f'static int n = {n_layers};\n', f'static int weights_shape[] = {{']
    activation_str = str(activation).strip('[]').replace('\'', '"')
    codes.append(f'static const char *act_funcs[] = {{{activation_str}}};\n')
    weights = ''
    bias = ''
    weights_shape = []
    for k, v in state_dict.items():
        tmp = ''
        for s in v.reshape(-1):
            tmp += f'{float(s):f} '
        if k.split('.')[1] == 'weight':
            codes[1] += f'{tuple(v.shape)}'.strip('()') + ', '
            weights += tmp
            weights_shape.append(tuple(v.shape))
        else:
            bias += tmp
    weights = weights.strip().replace(' ', ', ')
    codes.append(f'static float __weights_1d[] = {{{weights}}};\n')
    bias = bias.strip().replace(' ', ', ')
    codes.append(f'static float __bias_1d[] = {{{bias}}};\n')
    codes[1] = codes[1].strip(' ,') + '};\n'

    start_index = 0
    for i in range(n_layers):
        row, col = weights_shape[i]
        init_st = ''
        for j in range(row):
            init_st += f'&__weights_1d[{start_index + j * col}], '
        start_index += row * col
        init_st = '{' + init_st.strip(', ') + '}'
        codes.append(f'static float *__w{i}[{row}] = {init_st};\n')

    init_st = ''
    for i in range(n_layers):
        init_st += f'__w{i}, '
    init_st = '{' + init_st.strip(', ') + '}'

    codes.append(f'static float **weights[{n_layers}] = {init_st};\n')

    init_st = ''
    start_index = 0
    for i in range(n_layers):
        row, col = weights_shape[i]
        init_st += f'&__bias_1d[{start_index}], '
        start_index += row
    init_st = '{' + init_st.strip(', ') + '}'
    codes.append(f'static float *bias[{n_layers}] = {init_st};\n')

    max_output_dim = max([x[0] for x in weights_shape])
    codes.append(f'static float inter_result[{max_output_dim}];\n')
    codes.append(f'static float prev_output[{max_output_dim}];\n')
    codes.append(f'static float output[{weights_shape[-1][0]}];\n')

    codes.append(f"""
void relu(int col, float *input);
void tanh_(int col, float *input);
void matmul_1d(int row, int col, float *input, float **weights, float *output);
void add_bias(int col, float *input, float *bias);
void activate(int col, float *input, const char *act_fun);

float *{name}(float *input) {{
    int row = weights_shape[0], col = weights_shape[1];
    matmul_1d(row, col, input, weights[0], prev_output);
    add_bias(row, prev_output, bias[0]);
    activate(row, prev_output, act_funcs[0]);
    
    for (int i = 1; i < n; i++) {{
        int row = weights_shape[2 * i], col = weights_shape[2 * i + 1];

        matmul_1d(row, col, prev_output, weights[i], inter_result);
        add_bias(row, inter_result, bias[i]);
        activate(row, inter_result, act_funcs[i]);

        for (int j = 0; j < row; j++) {{
            prev_output[j] = inter_result[j];
        }}
    }}

    for(int i = 0; i < weights_shape[2 * (n - 1)]; i++) {{
        output[i] = prev_output[i];
    }}

    return output;
}}""")

    with open(f"{name}.cpp", 'w') as f:
        f.writelines(codes)

    total_para_num = 0
    for ws in weights_shape:
        total_para_num += ws[0] * ws[1] + ws[0]
    print(f'Finish saving, total number of parameters: {total_para_num}')


save_model_as_c(net, ['relu', 'relu', 'tanh'])
