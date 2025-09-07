# Compressing Deep Neural Networks through Quantization

This repository contains the research project for my MSc Artificial Intelligence dissertation (University of Plymouth, PROJ518). The project investigates **quantization** as a method for compressing deep neural networks (DNNs), focusing on **Post-Training Quantization (PTQ)** and **Quantization-Aware Training (QAT)** across different precision formats (**FP32, FP16, INT16, INT8**).

Three representative CNN architectures were selected:
- **AlexNet** (large, classic CNN)
- **ResNet-18** (residual network with moderate depth)
- **MobileNetV3-Small** (lightweight, mobile-optimized CNN)

The models are trained and evaluated on the [Kaggle Cats & Dogs dataset](https://www.kaggle.com/datasets/d4rklucif3r/cat-and-dogs).

---

## 📖 Project Overview
- Establish FP32 baselines for AlexNet, ResNet-18, and MobileNetV3-Small.
- Apply PTQ and QAT to generate FP16, INT16 (simulated), and INT8 quantized versions.
- Benchmark accuracy, inference latency, throughput, and memory footprint.
- Analyze trade-offs between compression efficiency and model fidelity.

---

## 🖥️ Experimental Environment
- **Hardware:** Lenovo IdeaPad Gaming 3  
  AMD Ryzen 7 4800H (8C/16T), 16 GB RAM, NVIDIA GTX 1650 Ti (4 GB, no Tensor Cores)
- **OS:** Ubuntu 24.04.2 LTS (Kernel 6.14)
- **Frameworks:** PyTorch 2.7.0+cu118, TorchVision 0.22.0
- **Backends:**  
  - FBGEMM → INT8 CPU inference  
  - QNNPACK → Lightweight CPU/mobile inference  
  - TensorRT + cuDNN → GPU FP16/INT8 inference  
  - AMP / `.half()` casting → FP16 support
- **Python version:** 3.12.3

---

## 📂 Repository Structure
