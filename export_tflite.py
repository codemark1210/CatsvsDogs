import os
import tensorflow as tf

os.makedirs("assets", exist_ok=True)

def export_fp32(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open("assets/model_fp32.tflite", "wb") as f:
        f.write(tflite_model)
    print("Exported assets/model_fp32.tflite")

def export_dynamic(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # dynamic range
    tflite_model = converter.convert()
    with open("assets/model_int8_dynamic.tflite", "wb") as f:
        f.write(tflite_model)
    print("Exported assets/model_int8_dynamic.tflite")

def main():
    model = tf.keras.models.load_model("saved_model")
    export_fp32(model)
    export_dynamic(model)
    print("Done.")

if __name__ == "__main__":
    main()
