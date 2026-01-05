import tensorflow as tf
import numpy as np
import json
import os

# ===============================
# 1. Load data
# ===============================
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# ===============================
# 2. Build model (stable high-acc)
# ===============================
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28), name="flatten"),
    tf.keras.layers.Dense(512, activation='relu', name="dense1"),
    tf.keras.layers.Dense(256, activation='relu', name="dense2"),
    tf.keras.layers.Dense(128, activation='relu', name="dense3"),
    tf.keras.layers.Dense(10, activation='softmax', name="dense4"),
])

optimizer = tf.keras.optimizers.SGD(
    learning_rate=0.05,
    momentum=0.9,
    nesterov=True
)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ===============================
# 3. Train
# ===============================
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=3,
    verbose=1
)

model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=128,
    validation_split=0.1,
    callbacks=[lr_callback],
    verbose=2
)

# ===============================
# 4. Evaluate
# ===============================
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n✅ Test accuracy = {test_acc:.4f}")

# ===============================
# 5. Export model arch
# ===============================
arch = []
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Flatten):
        arch.append({
            "name": layer.name,
            "type": "Flatten",
            "config": {},
            "weights": []
        })
    elif isinstance(layer, tf.keras.layers.Dense):
        arch.append({
            "name": layer.name,
            "type": "Dense",
            "config": {
                "activation": layer.activation.__name__
            },
            "weights": [
                f"{layer.name}_kernel",
                f"{layer.name}_bias"
            ]
        })

os.makedirs("model", exist_ok=True)
with open("model/fashion_mnist.json", "w") as f:
    json.dump(arch, f)

# ===============================
# 6. Export weights
# ===============================
weights = {}
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Dense):
        W, b = layer.get_weights()
        weights[f"{layer.name}_kernel"] = W
        weights[f"{layer.name}_bias"] = b

np.savez("model/fashion_mnist.npz", **weights)

print("\n🎉 Export complete!")