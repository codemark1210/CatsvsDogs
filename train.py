import os
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

IMG_SIZE = 224
BATCH = 32
EPOCHS = 5
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

os.makedirs("assets", exist_ok=True)

def preprocess(img, label):
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img, tf.cast(label, tf.float32)

def augment(img, label):
    img = tf.image.random_flip_left_right(img, seed=SEED)
    img = tf.image.random_brightness(img, max_delta=0.05)
    return img, label

def build_datasets():
    (ds_train, ds_val), info = tfds.load(
        "cats_vs_dogs",
        split=["train[:80%]", "train[80%:]"],
        with_info=True,
        as_supervised=True,
        shuffle_files=True
    )
    ds_train = ds_train.map(preprocess, num_parallel_calls=AUTOTUNE)\
                       .map(augment, num_parallel_calls=AUTOTUNE)\
                       .shuffle(4096, seed=SEED).batch(BATCH).prefetch(AUTOTUNE)

    ds_val = ds_val.map(preprocess, num_parallel_calls=AUTOTUNE)\
                   .batch(BATCH).prefetch(AUTOTUNE)
    return ds_train, ds_val, info

def build_model():
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False  # stage-1: freeze

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def plot_history(history, out_path="assets/training_curves.png"):
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    plt.figure()
    plt.plot(acc, label="train_acc")
    plt.plot(val_acc, label="val_acc")
    plt.plot(loss, label="train_loss")
    plt.plot(val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.title("Training Curves")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def main():
    ds_train, ds_val, _ = build_datasets()
    model = build_model()

    ckpt_path = "assets/best.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True, monitor="val_accuracy"),
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True)
    ]

    history = model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS, callbacks=callbacks)
    plot_history(history)

    # Save SavedModel for deployment / evaluation
    model.save("saved_model")
    print("Saved SavedModel -> saved_model")
    print("Best checkpoint ->", ckpt_path)

if __name__ == "__main__":
    main()
