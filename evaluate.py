import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

IMG_SIZE = 224
BATCH = 32
AUTOTUNE = tf.data.AUTOTUNE
os.makedirs("assets", exist_ok=True)

def preprocess(img, label):
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img, tf.cast(label, tf.float32)

def load_val():
    ds_val = tfds.load("cats_vs_dogs", split="train[80%:]", as_supervised=True, shuffle_files=False)
    ds_val = ds_val.map(preprocess, num_parallel_calls=AUTOTUNE).batch(BATCH).prefetch(AUTOTUNE)
    return ds_val

def main():
    model = tf.keras.models.load_model("saved_model")
    ds_val = load_val()

    y_true = []
    y_pred = []
    for x, y in ds_val:
        probs = model.predict(x, verbose=0).ravel()
        preds = (probs > 0.5).astype(np.int32)
        y_pred.append(preds)
        y_true.append(y.numpy().astype(np.int32))

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # Metrics
    acc = (y_true == y_pred).mean()
    print("Val Accuracy:", acc)
    print(classification_report(y_true, y_pred, target_names=["cat","dog"]))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["cat","dog"])
    fig, ax = plt.subplots()
    disp.plot(ax=ax, colorbar=False)
    plt.title("Confusion Matrix")
    plt.savefig("assets/confusion_matrix.png", bbox_inches="tight")
    plt.close()

    # Sample predictions grid
    make_sample_preds_grid()

def make_sample_preds_grid(n=16):
    import tensorflow_datasets as tfds
    raw = tfds.load("cats_vs_dogs", split="train[80%:]", as_supervised=True, shuffle_files=False)
    model = tf.keras.models.load_model("saved_model")

    imgs, lbls, preds = [], [], []
    for img, label in raw.take(n):
        img_resized = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        x = tf.keras.applications.mobilenet_v2.preprocess_input(tf.expand_dims(img_resized, 0))
        p = model.predict(x, verbose=0)[0,0]
        imgs.append(img_resized.numpy().astype("uint8"))
        lbls.append(int(label.numpy()))
        preds.append(int(p > 0.5))

    # Plot grid
    cols = int(np.sqrt(n))
    rows = int(np.ceil(n/cols))
    plt.figure(figsize=(cols*2.5, rows*2.5))
    for i in range(n):
        plt.subplot(rows, cols, i+1)
        plt.imshow(imgs[i])
        title = f"T:{'dog' if lbls[i] else 'cat'} / P:{'dog' if preds[i] else 'cat'}"
        color = "green" if lbls[i]==preds[i] else "red"
        plt.title(title, fontsize=9, color=color)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("assets/sample_preds.png", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
