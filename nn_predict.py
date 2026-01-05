import numpy as np
import json

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    """
    Supports both 1D and 2D input.
    """
    if x.ndim == 1:
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)
    else:
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def flatten(x):
    return x.reshape(x.shape[0], -1)

def dense(x, W, b):
    return np.dot(x, W) + b

def nn_forward_h5(model_arch, weights, data):
    x = data

    for layer in model_arch:
        layer_type = layer["class_name"]
        cfg = layer["config"]

        if layer_type == "Flatten":
            x = flatten(x)

        elif layer_type == "Dense":
            W = weights[cfg["weights"][0]]
            b = weights[cfg["weights"][1]]
            x = dense(x, W, b)

            act = cfg.get("activation")
            if act == "relu":
                x = relu(x)
            elif act == "softmax":
                x = softmax(x)

    return x

def nn_inference(model_arch, weights, data):
    return nn_forward_h5(model_arch, weights, data)