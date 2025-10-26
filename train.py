# train.py — FINAL ROBUST VERSION
# 运行：python -u train.py
# 产物：assets/best.keras, assets/final.keras, assets/training_curves.png

import os
import pathlib
import urllib.request
import zipfile
import tensorflow as tf
import matplotlib.pyplot as plt

# ===== 超参 =====
IMG_SIZE = 160
BATCH = 16
EPOCHS = 3
AUTOTUNE = tf.data.AUTOTUNE

# ===== 你可以改的路径：数据缓存与解压位置（不要放 C 盘）=====
CACHE_DIR   = r"F:\tfds_cache"                  # 压缩包下载的位置
EXTRACT_DIR = r"F:\tfds_cache\cadf_extracted"   # 解压后的固定目录（我们自己指定）
ASSETS_DIR  = "assets"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

ZIP_URL  = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
ZIP_PATH = os.path.join(CACHE_DIR, "cats_and_dogs_filtered.zip")

def ensure_dataset_ready():
    """手动下载并解压到 EXTRACT_DIR，确保 train/ 和 validation/ 存在"""
    # 1) 下载 zip（已存在就不下）
    if not os.path.exists(ZIP_PATH):
        print(f"⬇️  Downloading dataset zip to: {ZIP_PATH}")
        urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
        print("✅ Downloaded.")

    # 2) 如果 train/ 不存在，则解压（每次解压前先清理同名目录，防止旧残留）
    train_dir = os.path.join(EXTRACT_DIR, "cats_and_dogs_filtered", "train")
    val_dir   = os.path.join(EXTRACT_DIR, "cats_and_dogs_filtered", "validation")

    if not (os.path.isdir(train_dir) and os.path.isdir(val_dir)):
        # 清理旧解压（如果结构不对）
        target_root = os.path.join(EXTRACT_DIR, "cats_and_dogs_filtered")
        if os.path.isdir(target_root):
            import shutil
            shutil.rmtree(target_root, ignore_errors=True)

        print(f"🗜️  Extracting to: {EXTRACT_DIR}")
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(EXTRACT_DIR)
        print("✅ Extracted.")

    # 3) 最终检查
    if not (os.path.isdir(train_dir) and os.path.isdir(val_dir)):
        raise FileNotFoundError(
            "数据未正确解压。\n"
            f"  期望目录：\n    {train_dir}\n    {val_dir}\n"
            f"  实际 EXTRACT_DIR 内容：{list(os.listdir(EXTRACT_DIR))}"
        )

    return train_dir, val_dir

def build_datasets():
    train_dir, val_dir = ensure_dataset_ready()

    print("[paths]")
    print("  ZIP_PATH   :", ZIP_PATH)
    print("  EXTRACT_DIR:", EXTRACT_DIR)
    print("  train_dir  :", train_dir)
    print("  val_dir    :", val_dir)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH,
        label_mode="binary", shuffle=True
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH,
        label_mode="binary", shuffle=False
    )

    def preprocess(img, label):
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        return img, tf.cast(label, tf.float32)

    train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds   = val_ds.map(preprocess,   num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    return train_ds, val_ds

def build_model():
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()
    return model

def plot_history(history, out_path=os.path.join(ASSETS_DIR, "training_curves.png")):
    plt.figure()
    plt.plot(history.history.get("accuracy", []), label="train_acc")
    plt.plot(history.history.get("val_accuracy", []), label="val_acc")
    plt.plot(history.history.get("loss", []), label="train_loss")
    plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.xlabel("Epoch"); plt.title("Training Curves"); plt.legend()
    plt.savefig(out_path, bbox_inches="tight"); plt.close()
    print("🖼️  Saved training curves ->", out_path)

def main():
    ds_train, ds_val = build_datasets()
    model = build_model()

    ckpt_path = os.path.join(ASSETS_DIR, "best.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", mode="max", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=1, restore_best_weights=True)
    ]

    print("🚀 Start training ...")
    history = model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS, callbacks=callbacks)
    print("✅ Training done.")

    final_path = os.path.join(ASSETS_DIR, "final.keras")
    model.save(final_path)
    print("✅ Saved:", ckpt_path, "(best) and", final_path, "(final)")

    plot_history(history)

if __name__ == "__main__":
    main()
