This repository accompanies an MSc dissertation on compressing deep CNNs via quantization
ibm.com
. It targets popular architectures (AlexNet, ResNet-18, MobileNetV3-Small) using both Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT). Quantization converts high-precision (float32) weights to lower bit-width (FP16/INT16/INT8), which reduces model size and often improves inference speed
ibm.com
ai.google.dev
. For example, full 8-bit quantization can yield ~4× smaller models and ~3× faster CPU inference
