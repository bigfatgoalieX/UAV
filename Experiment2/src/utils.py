
def get_average_gradient(model):

    total_gradients = 0
    total_params = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            total_gradients += param.grad.abs().sum()
            total_params += param.numel()

    average_gradient = total_gradients / total_params
    return average_gradient