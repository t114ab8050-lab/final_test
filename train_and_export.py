import tensorflow as tf
import numpy as np
import json
import os

# 1. 載入資料
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 2. 建立更深更大的模型
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='relu', name='dense1'),
    tf.keras.layers.Dense(256, activation='relu', name='dense2'),
    tf.keras.layers.Dense(128, activation='relu', name='dense3'),
    tf.keras.layers.Dense(64, activation='relu', name='dense4'),
    tf.keras.layers.Dense(10, activation='softmax', name='dense5')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3. 訓練模型（epochs 增加到 40）
model.fit(x_train, y_train, epochs=40, batch_size=128, validation_data=(x_test, y_test))

# 4. 儲存模型
if not os.path.exists('model'):
    os.makedirs('model')
model.save('model/fashion_mnist.h5')

# 5. 儲存簡化架構
arch = []
for layer in model.layers:
    ltype = type(layer).__name__
    lname = layer.name
    cfg = {}
    wnames = []
    if ltype == "Dense":
        cfg = {
            "units": layer.units,
            "activation": layer.activation.__name__
        }
        wnames = [f"{lname}_kernel", f"{lname}_bias"]
    elif ltype == "Flatten":
        cfg = {}
    arch.append({
        "name": lname,
        "type": ltype,
        "config": cfg,
        "weights": wnames
    })
with open('model/fashion_mnist.json', 'w') as f:
    json.dump(arch, f)

# 6. 儲存權重（名稱需與 arch 對應）
weights = {}
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Dense):
        weights[f"{layer.name}_kernel"] = layer.get_weights()[0]
        weights[f"{layer.name}_bias"] = layer.get_weights()[1]
np.savez('model/fashion_mnist.npz', **weights)