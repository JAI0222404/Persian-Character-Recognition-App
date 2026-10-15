import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.labels import LETTERS, NUM_CLASSES

IMG_SIZE, BATCH, EPOCHS = 64, 64, 40
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
tf.keras.utils.set_random_seed(42)

d = np.load(ROOT / "dataset.npz")
def prep(X): return (X.astype("float32") / 255.0)[..., None]
X_train, y_train = prep(d["X_train"]), d["y_train"]
X_val, y_val = prep(d["X_val"]), d["y_val"]
X_test, y_test = prep(d["X_test"]), d["y_test"]
print("train/val/test:", len(X_train), len(X_val), len(X_test))

# ---- model ----
def conv_block(x, filters, convs=2):
    for _ in range(convs):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    return layers.Dropout(0.25)(x)

inputs = layers.Input((IMG_SIZE, IMG_SIZE, 1))
x = layers.RandomRotation(0.03, fill_mode="constant")(inputs)     
x = layers.RandomTranslation(0.08, 0.08, fill_mode="constant")(x)
x = layers.RandomZoom(0.12, fill_mode="constant")(x)
x = layers.GaussianNoise(0.05)(x)
x = conv_block(x, 32)
x = conv_block(x, 64)
x = conv_block(x, 128, convs=1)
x = layers.Flatten()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = models.Model(inputs, outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

cbs = [
    callbacks.EarlyStopping(patience=6, restore_best_weights=True),
    callbacks.ModelCheckpoint(MODEL_DIR / "persian_cnn.keras", save_best_only=True),
    callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
]
hist = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                 batch_size=BATCH, epochs=EPOCHS, callbacks=cbs)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, key in zip(ax, ("accuracy", "loss")):
    a.plot(hist.history[key], label="train")
    a.plot(hist.history["val_" + key], label="validation")
    a.set_title(key); a.set_xlabel("epoch"); a.legend()
plt.savefig(MODEL_DIR / "training_curves.png", dpi=120, bbox_inches="tight")

model = tf.keras.models.load_model(MODEL_DIR / "persian_cnn.keras")
pred = model.predict(X_test).argmax(axis=1)
acc = float((pred == y_test).mean())
print(f"\nTEST accuracy (unseen font families): {acc:.3f}")
names = [f"{i}:{name}" for i, (_, name, _) in enumerate(LETTERS)]
report = classification_report(y_test, pred, labels=range(NUM_CLASSES),
                               target_names=names, zero_division=0)
print(report)
(MODEL_DIR / "test_report.txt").write_text(f"test accuracy {acc:.4f}\n\n{report}")

cm = confusion_matrix(y_test, pred, labels=range(NUM_CLASSES))
plt.figure(figsize=(11, 10))
plt.imshow(cm, cmap="Blues"); plt.colorbar()
plt.xlabel("predicted class"); plt.ylabel("true class"); plt.title("Confusion matrix")
plt.savefig(MODEL_DIR / "confusion_matrix.png", dpi=120, bbox_inches="tight")

labels = [{"index": i, "char": c, "name": n, "translit": t}
          for i, (c, n, t) in enumerate(LETTERS)]
(MODEL_DIR / "labels.json").write_text(
    json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
print("Saved model, labels.json, plots, and report to models/")