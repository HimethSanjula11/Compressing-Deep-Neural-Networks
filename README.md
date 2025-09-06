This repository accompanies an MSc dissertation on compressing deep CNNs via quantization
ibm.com
. It targets popular architectures (AlexNet, ResNet-18, MobileNetV3-Small) using both Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT). Quantization converts high-precision (float32) weights to lower bit-width (FP16/INT16/INT8), which reduces model size and often improves inference speed
ibm.com
ai.google.dev
. For example, full 8-bit quantization can yield ~4× smaller models and ~3× faster CPU inference
ai.google.dev
.
Models: AlexNet, ResNet-18, MobileNetV3-Small
Techniques: Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT) at FP16, INT16, INT8
Dataset
We use the Kaggle “Cats vs. Dogs” dataset (≈25k labeled images)
microsoft.com
 for training and evaluation.
Environment
Benchmarks were run on Ubuntu 24.04 with Python 3.12, PyTorch 2.7.0, CUDA 12.0, and an NVIDIA GTX 1650 Ti GPU (no Tensor Cores). For CPU inference we use FBGEMM (Facebook’s high-performance 8-bit matrix library)
docs.pytorch.org
; for GPU inference we use NVIDIA TensorRT (an optimized deep‑learning inference SDK)
docs.nvidia.com
.
Usage
Run the provided scripts. For example:
python ptq.py --model resnet18 --precision int8   # post-training quantization
python qat.py --model alexnet --epochs 10        # quantization-aware training
Benchmarks
We measure inference latency, model file size, and memory usage for each model. Results illustrate the trade-offs between precision, size, and speed introduced by quantization. ⚠️ Note: This is a research prototype (MSc dissertation). Not intended for production use.
