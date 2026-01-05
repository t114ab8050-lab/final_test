import os
import json
import numpy as np

def train_and_export(seed=42):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.random.set_seed(seed)
    np.random.seed(seed)

    # ===== Load data =====
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0

    # ===== Model =====
    model = keras.Sequential([
        layers.Input(shape=(28, 28)),
        layers.Flatten(),

        layers.Dense(256),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.3),

        layers.Dense(256),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.3),

        layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        x_train, y_train,
        epochs=40,
        batch_size=128,
        validation_split=0.1,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=6,
                restore_best_weights=True
            )
        ],
        verbose=2,
    )

    test_acc = model.evaluate(x_test, y_test, verbose=0)[1]
    print(f"[TF] Test accuracy = {test_acc:.4f}")

    # ===== Export for NumPy inference =====
    os.makedirs("model", exist_ok=True)

    model_arch = [{"class_name": "Flatten", "config": {}}]
    weights = {}

    # ⚠ 只匯出 Dense（NumPy inference 只支援這個）
    for layer in model.layers:
        if isinstance(layer, layers.Dense):
            W, b = layer.get_weights()
            wname = layer.name + "_W"
            bname = layer.name + "_b"

            weights[wname] = W.astype(np.float32)
            weights[bname] = b.astype(np.float32)

            model_arch.append({
                "class_name": "Dense",
                "config": {
                    "weights": [wname, bname],
                    "activation": layer.activation.__name__
                }
            })

    with open("model/fashion_mnist.json", "w") as f:
        json.dump(model_arch, f, indent=2)

    np.savez("model/fashion_mnist.npz", **weights)

    print("[DONE] Exported model/fashion_mnist.json & fashion_mnist.npz")

if __name__ == "__main__":
    train_and_export()