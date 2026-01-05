import numpy as np
import json

# === Helper Function (徹底解決型別錯誤的關鍵) ===
def get_param_value(param, index=0):
    """
    不管傳入的是 list, tuple, numpy array 還是 int/float，
    都強制提取出一個乾淨的 int 純量。
    """
    try:
        # 如果是 Numpy Array 或 List/Tuple
        if hasattr(param, '__getitem__') and hasattr(param, '__len__'):
            # 如果是 0-d array (例如 np.array(1))
            if hasattr(param, 'ndim') and param.ndim == 0:
                return int(param)
            # 如果是正常 array/list，取出指定 index
            if len(param) > index:
                return int(param[index])
            else:
                return int(param[0]) # Fallback
        # 如果原本就是數字
        return int(param)
    except Exception:
        # 最後手段
        return int(param)

# === Activation functions ===
def sigmoid(x):
    # 數值穩定版 sigmoid
    return 1.0 / (1.0 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    x = np.asarray(x)
    # 數值穩定性處理：減去最大值防止 exp 爆炸
    if x.ndim == 1:
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)
    else:
        x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# === Layers ===
def flatten(x):
    # Flatten: (N, H, W, C) -> (N, H*W*C)
    return x.reshape(x.shape[0], -1)

def dense(x, W, b):
    # x: (N, features), W: (features, out), b: (out,)
    return x @ W + b

def conv2d(x, W, b, strides=1, padding='SAME'):
    # === 防禦性編碼：確保 stride 是乾淨的 int ===
    stride = get_param_value(strides, 0)
    
    # x: (N, H, W, C)
    # W: (kH, kW, C, F)
    # b: (F,)
    
    N, H, W_in, C = x.shape
    kH, kW, _, F = W.shape
    
    # 計算 Padding
    if padding == 'SAME':
        pad_h = int((kH - 1) // 2)
        pad_w = int((kW - 1) // 2)
        x_padded = np.pad(x, ((0,0), (pad_h, pad_h), (pad_w, pad_w), (0,0)), mode='constant')
    else:
        pad_h, pad_w = 0, 0
        x_padded = x
        
    # 計算輸出尺寸
    H_out = int((H + 2*pad_h - kH) // stride + 1)
    W_out = int((W + 2*pad_w - kW) // stride + 1)
    
    out = np.zeros((N, H_out, W_out, F))
    
    # 執行卷積
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            h_end = h_start + kH
            w_start = j * stride
            w_end = w_start + kW
            
            # 取出 Patch
            patch = x_padded[:, h_start:h_end, w_start:w_end, :]
            
            # Convolution operation
            # patch: (N, kH, kW, C) -> flatten -> (N, kH*kW*C)
            # W: (kH, kW, C, F) -> flatten -> (kH*kW*C, F)
            out[:, i, j, :] = patch.reshape(N, -1) @ W.reshape(-1, F)
            
    return out + b

def max_pooling2d(x, pool_size=(2, 2), strides=(2, 2)):
    N, H, W_in, C = x.shape
    
    # === 防禦性編碼：確保 pH, pW, sH, sW 都是乾淨的 int ===
    # 處理 pool_size
    pH = get_param_value(pool_size, 0)
    pW = get_param_value(pool_size, 1) if hasattr(pool_size, '__len__') and len(pool_size) > 1 else pH
    
    # 處理 strides
    sH = get_param_value(strides, 0)
    sW = get_param_value(strides, 1) if hasattr(strides, '__len__') and len(strides) > 1 else sH
    
    H_out = int((H - pH) // sH + 1)
    W_out = int((W - pW) // sW + 1)
    
    out = np.zeros((N, H_out, W_out, C))
    
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * sH
            h_end = h_start + pH
            w_start = j * sW
            w_end = w_start + pW
            
            patch = x[:, h_start:h_end, w_start:w_end, :]
            out[:, i, j, :] = np.max(patch, axis=(1, 2))
            
    return out

# === Inference Engine ===
def nn_forward_h5(model_arch, weights, data):
    x = data
    # 如果輸入是 (N, 784)，Reshape 回 (N, 28, 28, 1) 給 CNN 用
    if x.ndim == 2 and x.shape[1] == 784:
        x = x.reshape(-1, 28, 28, 1)
        
    for layer in model_arch:                                                
        lname = layer['name']
        ltype = layer['type']
        cfg = layer['config']
        wnames = layer['weights']

        if ltype == "Flatten":
            x = flatten(x)
        
        elif ltype == "Dense":
            W = weights[wnames[0]]
            b = weights[wnames[1]]
            x = dense(x, W, b)
            
        elif ltype == "Conv2D":
            W = weights[wnames[0]]
            b = weights[wnames[1]]
            
            # 從 config 讀取，並直接傳入 conv2d 讓它自己去 parse
            stride_cfg = cfg.get('strides', [1,1])
            padding = cfg.get('padding', 'VALID').upper()
            
            x = conv2d(x, W, b, strides=stride_cfg, padding=padding)
            
        elif ltype == "MaxPooling2D":
            pool_size_cfg = cfg.get('pool_size', [2,2])
            strides_cfg = cfg.get('strides', [2,2])
            
            x = max_pooling2d(x, pool_size=pool_size_cfg, strides=strides_cfg)

        # 統一處理 Activation
        act = cfg.get("activation")
        if act == "relu":
            x = relu(x)
        elif act == "softmax":
            x = softmax(x)
        elif act == "sigmoid":
            x = sigmoid(x)

    return x

def nn_inference(model_arch, weights, data):
    return nn_forward_h5(model_arch, weights, data)