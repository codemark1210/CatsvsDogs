import argparse
import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = 224

def load_image(path):
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32")
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, 0), img

def main(image_path):
    model = tf.keras.models.load_model("saved_model")
    x, img = load_image(image_path)
    prob = model.predict(x, verbose=0)[0,0]
    pred = "dog" if prob > 0.5 else "cat"
    print(f"Prediction: {pred} (score={prob:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to an image file")
    args = parser.parse_args()
    main(args.image)
