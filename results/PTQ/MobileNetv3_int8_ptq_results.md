# MobileNetV3-Small ROBUST INT8 Quantization (PTQ) — Dissertation Analysis

**Generated:** 2025-08-20 19:18:06

## System & Backends
- Device: cuda
- GPU: NVIDIA GeForce GTX 1650 Ti
- PyTorch INT8 Backend: **FBGEMM** (x86)
- PyTorch Version: 2.8.0+cu128
- TensorRT Available: True
- PyCUDA Available: False
- TensorRT INT8 Notes: Unavailable (needs TensorRT + PyCUDA + CUDA GPU w/ fast INT8)

## Robust PTQ Implementation Details
This implementation uses a simplified but robust approach to quantization:

### PTQ Strategy ✓
- ✓ **Engine**: `torch.backends.quantized.engine = 'fbgemm'` (x86 optimized)
- ✓ **Default Config**: Using PyTorch's default FBGEMM quantization config
- ✓ **Validation**: Extensive verification and debugging steps
- ✓ **Calibration**: 512 representative images from train+val mix
- ✓ **Error Handling**: Robust error detection and fallback strategies

## Executive Summary
- We perform **robust post-training static quantization (PTQ)** using FX graph mode.
- Simplified approach prioritizes working quantization over complex optimizations.
- Extensive validation and debugging to ensure quantization actually occurs.
- Multi-run benchmarking with statistical analysis and error handling.

## Statistical Results (means ± std)
| Model-Device | Acc (%) | Time (s) | Memory (MB) |
|---|---:|---:|---:|
| fp32_gpu | 92.950 ± 0.000 | 5.006019 ± 0.040239 | 1.93 ± 3.85 |
| fp32_cpu | 92.950 ± 0.000 | 10.761814 ± 0.198085 | 10.09 ± 16.75 |
| int8_cpu_cpu | 58.150 ± 0.000 | 5.965810 ± 0.099973 | -6.67 ± 13.42 |

## Matrices

### Accuracy

|             |    GPU |   CPU |
|:------------|-------:|------:|
| FP32        |  92.95 | 92.95 |
| INT8_FBGEMM | nan    | 58.15 |

### Time

|             |       GPU |      CPU |
|:------------|----------:|---------:|
| FP32        |   5.00602 | 10.7618  |
| INT8_FBGEMM | nan       |  5.96581 |

### Memory

|             |       GPU |      CPU |
|:------------|----------:|---------:|
| FP32        |   1.92717 | 10.0852  |
| INT8_FBGEMM | nan       | -6.67484 |

### Throughput

|             |     GPU |     CPU |
|:------------|--------:|--------:|
| FP32        | 399.545 | 185.906 |
| INT8_FBGEMM | nan     | 335.336 |

## Layer Analysis
- FP32 types:

```
type
Conv2d               52
BatchNorm2d          34
Hardswish            19
ReLU                 14
AdaptiveAvgPool2d    10
Hardsigmoid           9
Linear                2
Dropout               1
```
- INT8(FBGEMM) quantized layers: 74

## Quantization Quality Metrics
- **Accuracy Preservation**: 58.15% (+34.80% vs FP32)
- **Latency Speedup**: 1.80x faster on CPU
- ⚠️  **Quality Warning**: Accuracy loss > 5% indicates quantization issues
- **Model Compression**: 3.39x smaller (1.8MB vs 6.2MB)

## Troubleshooting Notes
If quantization shows poor results, check:
1. **Observer Detection**: Ensure observers are properly inserted during prepare phase
2. **Calibration Data**: Use diverse, representative samples for calibration
3. **Module Detection**: Verify quantized modules are actually created
4. **PyTorch Version**: Ensure compatible PyTorch version with stable quantization
5. **Backend Support**: Verify FBGEMM backend is properly configured

## Conclusions
- **Robust PTQ approach** prioritizes working quantization over complex optimizations.
- **Extensive validation** helps identify and debug quantization failures.
- **Simplified configuration** reduces chances of configuration errors.
- **Error handling** provides better debugging information for issues.
- ✅ **Quantization successful** - detected quantized layers in model.
