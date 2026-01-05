import numpy as np
import json

# === Activation functions ===
def relu(x):
    # 實作 ReLU: 小於 0 變 0，大於 0 保持原樣
    return np.maximum(0, x)

def softmax(x):
    # 實作數值穩定的 Softmax
    x = np.asarray(x)
    # 減去最大值以防止指數爆炸 (Numerical Stability)
    if x.ndim == 1:
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)
    else:
        # 針對 batch 處理
        x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
# === Flatten ===
def flatten(x):
    return x.reshape(x.shape[0], -1)

# === Dense layer ===
def dense(x, W, b):
    # 矩陣乘法 + 偏差
    return x @ W + b

# Infer TensorFlow h5 model using numpy
# Support only Dense, Flatten, relu, softmax now
def nn_forward_h5(model_arch, weights, data):
    x = data
    for layer in model_arch:                                                    
        lname = layer['name']
        ltype = layer['type']
        cfg = layer['config']
        wnames = layer['weights']

        if ltype == "Flatten":
            x = flatten(x)
        elif ltype == "Dense":
            # 從 weights 字典中取出 W 和 b
            W = weights[wnames[0]]
            b = weights[wnames[1]]
            x = dense(x, W, b)
            
            # 套用激活函數
            if cfg.get("activation") == "relu":
                x = relu(x)
            elif cfg.get("activation") == "softmax":
                x = softmax(x)

    return x

def nn_inference(model_arch, weights, data):
    return nn_forward_h5(model_arch, weights, data)