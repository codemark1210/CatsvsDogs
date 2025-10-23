# 🐱🐶 Cats vs Dogs Classification with TensorFlow

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Dataset](https://img.shields.io/badge/Dataset-Cats%20vs%20Dogs-blue?logo=google)
![Model](https://img.shields.io/badge/Model-MobileNetV2-success)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Deploy-TFLite-lightgrey?logo=android)

A classic **image classification project** built with **TensorFlow** and **Keras**,  
featuring **transfer learning**, **evaluation with visualization**, and **TensorFlow Lite export** for efficient on-device deployment.

## 📦 Installation
```bash
git clone https://github.com/<your-username>/tf-cats-vs-dogs.git
cd tf-cats-vs-dogs
pip install -r requirements.txt
```

## 🚀 Quick Start
```bash
python train.py
python evaluate.py
python export_tflite.py
python predict.py --image path/to/image.jpg
