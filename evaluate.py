# evaluate.py — load local validation set + draw confusion matrix & sample preds

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

ASSETS_DIR = "assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

IMG_SIZE = 160
BATCH = 32
AUTOTUNE = tf.data.AUTOTUNE

# 按你的 train.py 手动解压路径来设置（如有修改，请同步改这里）
DATA_ROOT = r"F:\tfds_cache\cadf_extracted\cats_and_dogs_filtered"
VAL_DIR   = os.path.join(DATA_ROOT, "validation")

# 1) 载入模型（优先 best.keras）
for p in [os.path.join(ASSETS_DIR, "best.keras"),
          os.path.join(ASSETS_DIR, "final.keras")]:
    if os.path.exists(p):
        model_path = p
        break
else:
    raise FileNotFoundError("没有找到模型：assets/best.keras 或 assets/final.keras")

print(f"✅ Loaded model: {model_path}")
model = tf.keras.models.load_model(model_path)

if not os.path.isdir(VAL_DIR):
    raise FileNotFoundError(f"验证目录不存在：{VAL_DIR}")

# 2) 构建验证集：先拿 class_names，再预处理+prefetch
val_raw = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    label_mode="binary",
    shuffle=False
)
class_names = val_raw.class_names  # 先取出来
print("Classes:", class_names)

def preprocess(img, label):
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img, tf.cast(label, tf.float32)

val_ds = val_raw.map(preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

# 3) 预测全部样本 -> 混淆矩阵
y_true, y_pred = [], []
for images, labels in val_ds:
    probs = model.predict(images, verbose=0).ravel()
    preds = (probs > 0.5).astype(np.int32)
    y_true.append(labels.numpy().astype(np.int32).ravel())
    y_pred.append(preds)
y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)

cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=2).numpy()

plt.figure(figsize=(4.8, 4.2))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix"); plt.colorbar()
ticks = np.arange(2)
plt.xticks(ticks, class_names, rotation=45); plt.yticks(ticks, class_names)
thr = cm.max() / 2
for i in range(2):
    for j in range(2):
        plt.text(j, i, int(cm[i, j]),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thr else "black")
plt.ylabel("True label"); plt.xlabel("Predicted label")
plt.tight_layout()
cm_path = os.path.join(ASSETS_DIR, "confusion_matrix.png")
plt.savefig(cm_path, bbox_inches="tight"); plt.close()
print("✅ Saved:", cm_path)

# 4) 随机抽一批可视化预测（把 [-1,1] 还原到 [0,255] 方便展示）
sample_batch = next(val_raw.as_numpy_iterator())  # 原始未预处理的
imgs_uint8, labels = sample_batch
# 需要用预处理后的图像做预测
imgs_proc = tf.keras.applications.mobilenet_v2.preprocess_input(imgs_uint8.copy())
probs = model.predict(imgs_proc, verbose=0).ravel()
preds = (probs > 0.5).astype(np.int32)

n = min(12, len(imgs_uint8))
rows, cols = 3, 4
plt.figure(figsize=(12, 8))
for i in range(n):
    plt.subplot(rows, cols, i+1)
    plt.imshow(imgs_uint8[i].astype("uint8"))
    t = class_names[int(labels[i])]
    p = class_names[int(preds[i])]
    conf = probs[i] if preds[i] == 1 else (1 - probs[i])
    plt.title(f"T:{t}  P:{p}\nconf:{conf:.2f}")
    plt.axis("off")
plt.tight_layout()
sp_path = os.path.join(ASSETS_DIR, "sample_preds.png")
plt.savefig(sp_path, bbox_inches="tight"); plt.close()
print("✅ Saved:", sp_path)

