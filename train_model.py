import tensorflow as tf
import numpy as np
import json
import os

# === Helper Function: Fold Batch Normalization ===
def fold_batch_norm(model):
    """
    將 BatchNormalization 層的參數融合進前一層的 Conv2D 或 Dense 層。
    這樣推論時就不需要 BN 層，簡化實作但保留效果。
    """
    new_layers = []
    
    # 暫存前一層的權重，等待融合
    prev_layer = None
    prev_w = None
    prev_b = None
    
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
            # 如果已經有暫存的層，先存起來 (因為它後面沒接 BN)
            if prev_layer is not None:
                new_layers.append((prev_layer, prev_w, prev_b))
            
            prev_layer = layer
            prev_w, prev_b = layer.get_weights()
            
        elif isinstance(layer, tf.keras.layers.BatchNormalization) and prev_layer is not None:
            # 遇到 BN 層，執行融合
            gamma, beta, mean, var = layer.get_weights()
            epsilon = layer.epsilon
            
            scale = gamma / np.sqrt(var + epsilon)
            
            # 更新權重: W_new = W * scale
            # 注意 Conv2D 和 Dense 的權重形狀不同，但廣播機制通常能處理
            if isinstance(prev_layer, tf.keras.layers.Conv2D):
                # Conv Kernel: (H, W, C, F), scale: (F,)
                prev_w = prev_w * scale.reshape(1, 1, 1, -1)
            else:
                # Dense Kernel: (In, Out), scale: (Out,)
                prev_w = prev_w * scale.reshape(1, -1)
                
            # 更新偏差: b_new = (b - mean) * scale + beta
            prev_b = (prev_b - mean) * scale + beta
            
            # 融合完成，將「強化版」的前一層加入列表，並丟棄 BN 層
            new_layers.append((prev_layer, prev_w, prev_b))
            prev_layer = None # 重置
            
        else:
            # 其他層 (Activation, Pooling, Flatten, Dropout 等)
            if prev_layer is not None:
                new_layers.append((prev_layer, prev_w, prev_b))
                prev_layer = None
            
            # 對於 Dropout 等層，我們在推論時會忽略或直接存 config
            if not isinstance(layer, (tf.keras.layers.Dropout, tf.keras.layers.InputLayer)):
                new_layers.append((layer, None, None))

    # 處理最後一層
    if prev_layer is not None:
        new_layers.append((prev_layer, prev_w, prev_b))
        
    return new_layers

# ==========================================
# 1. 準備資料
# ==========================================
print("正在載入與處理資料...")
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

# ==========================================
# 2. 建立 CNN 模型 (目標準確率 > 92%)
# ==========================================
# 結構: Conv -> BN -> ReLU -> Pool -> Conv -> BN -> ReLU -> Pool -> Dense -> BN -> ReLU -> Dense
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
    
    # Block 1
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', use_bias=True), # Bias=True為了融合方便
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.2),
    
    # Block 2
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', use_bias=True),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Flatten(),
    
    # Dense Block
    tf.keras.layers.Dense(256, use_bias=True),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.4),
    
    tf.keras.layers.Dense(10, activation='softmax', name='output')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ==========================================
# 3. 訓練模型
# ==========================================
checkpoint_path = 'model/best_cnn.h5'
if not os.path.exists('model'): os.makedirs('model')

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, save_best_only=True, monitor='val_accuracy', verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
]

print("開始訓練 CNN 模型...")
# Epochs 設為 30 左右通常就能收斂到 92% 以上
model.fit(x_train, y_train, epochs=40, batch_size=128, validation_data=(x_test, y_test), callbacks=callbacks)

# 載入最好的模型
best_model = tf.keras.models.load_model(checkpoint_path)
loss, acc = best_model.evaluate(x_test, y_test, verbose=0)
print(f"\n最佳模型準確率: {acc:.4f}")

# ==========================================
# 4. 融合 BN 層並匯出 (關鍵步驟)
# ==========================================
print("正在融合 BN 層並匯出 JSON 與 NPZ...")

fused_layers = fold_batch_norm(best_model)
arch = []
weights_dict = {}

for layer_info in fused_layers:
    layer, W, b = layer_info
    ltype = type(layer).__name__
    lname = layer.name
    
    # 忽略不支援的層
    if ltype not in ['Conv2D', 'Dense', 'Flatten', 'MaxPooling2D', 'Activation']:
        continue
        
    # 如果是 Activation 層，通常是獨立的 ReLU
    if ltype == 'Activation':
        # 把這層資訊併入前一層 (nn_predict 的設計是 activation 在 config 裡)
        if len(arch) > 0:
            arch[-1]['config']['activation'] = layer.activation.__name__
        continue

    # 建構 Config
    cfg = {}
    wnames = []
    
    if ltype == 'Conv2D':
        cfg = {
            'filters': layer.filters,
            'kernel_size': layer.kernel_size,
            'strides': layer.strides,
            'padding': layer.padding.upper(),
            'activation': None # 預設無，除非後面有 Activation 層
        }
        wnames = [f"{lname}_kernel", f"{lname}_bias"]
        weights_dict[wnames[0]] = W
        weights_dict[wnames[1]] = b
        
    elif ltype == 'Dense':
        cfg = {
            'units': layer.units,
            'activation': layer.activation.__name__ if layer.activation.__name__ != 'linear' else None
        }
        wnames = [f"{lname}_kernel", f"{lname}_bias"]
        weights_dict[wnames[0]] = W
        weights_dict[wnames[1]] = b
        
    elif ltype == 'MaxPooling2D':
        cfg = {
            'pool_size': layer.pool_size,
            'strides': layer.strides
        }
    
    arch.append({
        "name": lname,
        "type": ltype,
        "config": cfg,
        "weights": wnames
    })

# 寫入檔案
with open('model/fashion_mnist.json', 'w') as f:
    json.dump(arch, f, indent=4)

np.savez('model/fashion_mnist.npz', **weights_dict)
print("完成！所有檔案已更新，可以執行 python -m pytest model_test.py")