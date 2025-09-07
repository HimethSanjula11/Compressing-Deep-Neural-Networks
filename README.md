# Compressing Deep Neural Networks through Quantization

This repository contains the MSc Artificial Intelligence dissertation project (University of Plymouth, PROJ518).  
The research investigates **quantization** as a method for compressing deep neural networks (DNNs), focusing on  
**Post-Training Quantization (PTQ)** and **Quantization-Aware Training (QAT)** across multiple precision formats (**FP32, FP16, INT16, INT8**).

Three CNN architectures were evaluated:
- **AlexNet** (large, early CNN with ~60M parameters)
- **ResNet-18** (residual network with ~11M parameters)
- **MobileNetV3-Small** (lightweight CNN with ~2.5M parameters)

Dataset: [Cats & Dogs (Kaggle)](https://www.kaggle.com/datasets/d4rklucif3r/cat-and-dogs)

---

## 📖 Project Overview
- Establish FP32 baselines for AlexNet, ResNet-18, and MobileNetV3-Small.  
- Apply **PTQ** and **QAT** to obtain FP16, INT16 (simulated), and INT8 quantized versions.  
- Benchmark accuracy, inference latency, throughput, and model footprint.  
- Analyze trade-offs between compression efficiency and predictive fidelity.  

---

## 📊 Dataset
- **Cats vs Dogs dataset** (25,000 images).  
- Preprocessing: resize (224×224), normalization, random crops/flips/rotations.  
- Split: 80% training (8,000 cats + 8,000 dogs), 20% testing (2,000 cats + 2,000 dogs).  

---

## 🖥️ Experimental Environment
- **Hardware:** Lenovo IdeaPad Gaming 3  
  - AMD Ryzen 7 4800H (8C/16T)  
  - 16 GB DDR4 RAM  
  - NVIDIA GTX 1650 Ti (4 GB VRAM, no Tensor Cores)  
- **OS:** Ubuntu 24.04.2 LTS (Kernel 6.14)  
- **Python:** 3.12.3  
- **Frameworks & Backends:**  
  - PyTorch 2.7.0+cu118, TorchVision 0.22.0  
  - FBGEMM → INT8 CPU inference  
  - QNNPACK → lightweight CPU/mobile inference  
  - TensorRT + cuDNN → GPU FP16/INT8 inference  
  - AMP / `.half()` casting → FP16 support  

---

## 📂 Repository Structure
├── models/ # FP32 baseline training scripts

├── quantization/
│ ├── ptq/ # Post-Training Quantization (INT8, FP16, INT16)
│ └── qat/ # Quantization-Aware Training (INT8, FP16, INT16)

├── utils/ # Data loading, benchmarking, profiling

├── results/ # CSV/JSON logs of experiments

├── figures/ # Graphs and plots

└── README.md # Project documentation

## 📦 Dependencies
Install everything at once:
```bash
pip install torch==2.7.0+cu118 torchvision==0.22.0
pip install numpy scipy scikit-learn matplotlib seaborn psutil onnx onnxruntime


🚀 Usage
git clone https://github.com/HimethSanjula11/Compressing-Deep-Neural-Networks.git
cd Compressing-Deep-Neural-Networks

Train FP32 baselines
python models/alexnet_fp32.py
python models/resnet18_fp32.py
python models/mobilenetv3_fp32.py


⚠️ Notes

This is a research prototype for MSc dissertation purposes.

Not optimized for production deployment.

Large model files (*.pth) are ignored using .gitignore.
