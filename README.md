# Cats vs Dogs with TensorFlow (Transfer Learning + TFLite)

A classic, clean, and deployable TensorFlow project:
- **Task:** Binary image classification (cats vs dogs)
- **Approach:** Transfer Learning with `MobileNetV2` + simple head
- **Dataset:** `tfds: cats_vs_dogs`
- **Deploy:** Export to **TensorFlow Lite** (FP32 & dynamic range quantization)
- **Extras:** Confusion matrix, sample predictions, training curves

## Quick Start

```bash
# 1) Create venv (optional)
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

# 2) Install
pip install -r requirements.txt

# 3) Train
python train.py

# 4) Evaluate (confusion matrix & sample preds saved to assets/)
python evaluate.py

# 5) Export TFLite (FP32 + dynamic quantized)
python export_tflite.py

# 6) Single image inference (path to a local image)
python predict.py --image path/to/your_image.jpg
