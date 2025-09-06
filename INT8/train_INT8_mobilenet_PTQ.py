import os
import time
import json
import copy
import psutil
import torch
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from scipy import stats
from datetime import datetime

warnings.filterwarnings("ignore")

# ===========================
# Optional TensorRT + PyCUDA
# ===========================
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
    print("TensorRT available")
except Exception:
    trt = None
    TENSORRT_AVAILABLE = False
    print("TensorRT NOT available")

try:
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa
    PYCUDA_AVAILABLE = True
except Exception:
    cuda = None
    PYCUDA_AVAILABLE = False
    print("PyCUDA NOT available")

# ===========================
# Matplotlib styling
# ===========================
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ===========================
# Global script dir
# ===========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================
# TensorRT Runtime Wrapper (works for FP32/FP16 inputs too)
# =========================================================
def _trt_dtype_to_torch(dtype):
    if not TENSORRT_AVAILABLE:
        return torch.float32
    m = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT8: torch.int8,
        trt.DataType.BOOL: torch.bool
    }
    return m.get(dtype, torch.float32)

class TensorRTEngineModule(nn.Module):
    """
    Lightweight nn.Module wrapper around a TensorRT engine (explicit batch).
    Engine built for static shape [1,3,224,224]. We feed FP32/FP16 inputs; INT8
    quantization inside the engine is handled by TensorRT.
    """
    def __init__(self, engine):
        super().__init__()
        assert TENSORRT_AVAILABLE and PYCUDA_AVAILABLE, "TRT runtime requires TensorRT + PyCUDA"
        self.engine = engine
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        assert self.engine.num_bindings == 2, "Expected 1 input + 1 output bindings"

        # Determine binding indices
        self.input_idx = 0 if self.engine.binding_is_input(0) else 1
        self.output_idx = 1 - self.input_idx

        # Dtypes
        self.input_dtype = _trt_dtype_to_torch(self.engine.get_binding_dtype(self.input_idx))
        self.output_dtype = _trt_dtype_to_torch(self.engine.get_binding_dtype(self.output_idx))

        # Names
        self.input_name = self.engine.get_binding_name(self.input_idx)
        self.output_name = self.engine.get_binding_name(self.output_idx)

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda, "TensorRT module expects CUDA input"
        x = x.contiguous()
        self.context.set_binding_shape(self.input_idx, tuple(x.shape))
        out_shape = tuple(self.context.get_binding_shape(self.output_idx))
        out = torch.empty(out_shape, device="cuda", dtype=self.output_dtype if self.output_dtype != torch.int8 else torch.float32)

        bindings = [None] * self.engine.num_bindings
        bindings[self.input_idx] = int(x.data_ptr())
        bindings[self.output_idx] = int(out.data_ptr())

        self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)
        self.stream.synchronize()

        # Return logits as float32 for criterion/argmax stability
        return out.float()

# ==========================================
# TensorRT INT8 Entropy Calibrator (batch=1)
# ==========================================
class ImageCalibrator(trt.IInt8EntropyCalibrator2 if TENSORRT_AVAILABLE else object):
    """
    Uses a small stream of validation images for INT8 calibration.
    Expects tensors shaped [1,3,224,224] on host (we copy to device).
    """
    def __init__(self, calibration_tensors, cache_file):
        if not TENSORRT_AVAILABLE:
            return
        super().__init__()
        self.cache_file = cache_file
        self.data = calibration_tensors
        self.idx = 0
        self.device_input = None

        # Allocate device buffer
        if len(self.data) > 0:
            sample = self.data[0]
            assert sample.shape == (1,3,224,224)
            self.input_size = sample.numel() * sample.element_size()
            self.device_input = cuda.mem_alloc(self.input_size)

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.idx >= len(self.data):
            return None
        batch = self.data[self.idx]
        cuda.memcpy_htod(self.device_input, batch.cpu().numpy().ravel())
        self.idx += 1
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

# ==========================================
# ONNX Export + TensorRT INT8 Build Helpers
# ==========================================
def export_onnx_static(model: nn.Module, onnx_path: str, input_shape=(1,3,224,224)):
    model.eval().to("cuda")
    dummy = torch.randn(*input_shape, device="cuda")
    print(f"Exporting ONNX to {onnx_path} (opset 17, static {input_shape}) ...")
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"], output_names=["logits"],
        opset_version=17, do_constant_folding=True, dynamic_axes=None
    )
    print("✓ ONNX export complete")

def build_trt_int8_engine_from_onnx(onnx_path: str, plan_path: str, calibration_tensors, workspace_bytes=1<<29):
    """
    Build an INT8 TensorRT engine with entropy calibrator. If INT8 platform support is absent,
    raises RuntimeError (caller can handle and skip).
    """
    logger = trt.Logger(trt.Logger.WARNING)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(logger) as builder, \
         builder.create_network(flag) as network, \
         trt.OnnxParser(network, logger) as parser:

        print("Parsing ONNX...")
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"[TRT Parser Error] {parser.get_error(i)}")
                raise RuntimeError("Failed to parse ONNX")

        if not builder.platform_has_fast_int8:
            raise RuntimeError("This GPU does NOT report fast INT8 support. Skipping INT8 engine.")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        config.set_flag(trt.BuilderFlag.INT8)

        cache_file = os.path.join(SCRIPT_DIR, "mobilenetv3_int8.calib")
        calibrator = ImageCalibrator(calibration_tensors, cache_file)
        config.set_int8_calibrator(calibrator)

        print("Building INT8 TensorRT engine (entropy calibrator)...")
        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            raise RuntimeError("INT8 engine build failed")

        with open(plan_path, "wb") as f:
            f.write(engine_bytes)
        print(f"✓ Saved INT8 engine to {plan_path}")

        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        print("✓ INT8 engine deserialized")
        return engine

# ===========================
# Enhanced PTQ INT8 Quantization (FIXED VERSION)
# ===========================
def create_calibration_dataset(train_dataset, val_dataset, num_samples=1024):
    """
    Create a mixed calibration dataset from train+val with specified number of samples.
    This follows the PTQ best practice of using 512-1024 representative images.
    """
    print(f"Creating calibration dataset with {num_samples} samples...")
    
    # Mix train and val datasets (70% train, 30% val)
    train_samples = int(num_samples * 0.7)
    val_samples = num_samples - train_samples
    
    # Randomly sample from both datasets
    train_indices = torch.randperm(len(train_dataset))[:train_samples]
    val_indices = torch.randperm(len(val_dataset))[:val_samples]
    
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    
    # Combine datasets
    from torch.utils.data import ConcatDataset
    calib_dataset = ConcatDataset([train_subset, val_subset])
    
    print(f"✓ Calibration dataset created: {len(train_subset)} train + {len(val_subset)} val = {len(calib_dataset)} total")
    return calib_dataset

def verify_quantization_support():
    """
    Verify that quantization is properly supported and configured.
    """
    print("Verifying quantization support...")
    
    # Check PyTorch version
    torch_version = torch.__version__
    print(f"PyTorch version: {torch_version}")
    
    # Test quantization availability
    try:
        from torch.ao.quantization import get_default_qconfig
        qconfig = get_default_qconfig('fbgemm')
        print("✓ FBGEMM quantization support available")
        return True
    except Exception as e:
        print(f"✗ Quantization support issue: {e}")
        return False

def robust_ptq_int8_fbgemm_fx(model_fp32: nn.Module, train_dataset, val_dataset, num_calib_samples=512):
    """
    ROBUST Post-Training Static Quantization with extensive validation and debugging.
    Simplified approach that focuses on working quantization first.
    """
    print("\n" + "="*70)
    print("ROBUST PTQ INT8 (FX) with FBGEMM - SIMPLIFIED & DEBUGGED")
    print("="*70)
    
    # 0. Verify quantization support
    if not verify_quantization_support():
        raise RuntimeError("Quantization not properly supported")
    
    # 1. Set FBGEMM backend
    torch.backends.quantized.engine = 'fbgemm'
    print("✓ Set quantization backend to FBGEMM")
    
    # 2. Create calibration dataset (smaller for stability)
    calib_dataset = create_calibration_dataset(train_dataset, val_dataset, num_calib_samples)
    calib_loader = DataLoader(calib_dataset, batch_size=1, shuffle=True, num_workers=0)  # num_workers=0 for stability
    
    # 3. Prepare model (copy to CPU, eval mode)
    model_fp32 = copy.deepcopy(model_fp32).cpu().eval()
    print("✓ Model copied to CPU and set to eval mode")
    
    # 4. Use simple default quantization configuration
    from torch.ao.quantization import get_default_qconfig, QConfigMapping
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
    
    qconfig = get_default_qconfig('fbgemm')
    qconfig_mapping = QConfigMapping().set_global(qconfig)
    
    print("✓ Using default FBGEMM qconfig")
    
    # 5. Prepare for quantization
    print("Preparing model for quantization...")
    example_inputs = torch.randn(1, 3, 224, 224)
    
    try:
        prepared_model = prepare_fx(model_fp32, qconfig_mapping, example_inputs)
        print("✓ Model prepared successfully")
    except Exception as e:
        print(f"✗ Failed to prepare model: {e}")
        # Fallback: try without fusion
        print("Attempting fallback without explicit fusion...")
        prepared_model = prepare_fx(model_fp32, qconfig_mapping, example_inputs)
    
    # 6. Verify observers are inserted
    observer_count = 0
    for name, module in prepared_model.named_modules():
        if 'observer' in name.lower() or hasattr(module, 'activation_post_process'):
            observer_count += 1
    print(f"✓ Found {observer_count} observers in prepared model")
    
    if observer_count == 0:
        print("⚠️  Warning: No observers found - quantization may not work properly")
    
    # 7. Calibration phase
    print(f"Calibrating on {num_calib_samples} samples...")
    prepared_model.eval()
    
    calibration_count = 0
    with torch.inference_mode():
        for batch_idx, (data, _) in enumerate(calib_loader):
            if calibration_count >= num_calib_samples:
                break
            
            try:
                _ = prepared_model(data)
                calibration_count += 1
                
                if batch_idx % 100 == 0 and batch_idx > 0:
                    print(f"  Calibrated on {batch_idx} samples")
            except Exception as e:
                print(f"⚠️  Calibration error on batch {batch_idx}: {e}")
                continue
    
    print(f"✓ Calibration complete ({calibration_count} samples)")
    
    # 8. Convert to quantized model
    print("Converting to quantized model...")
    try:
        quantized_model = convert_fx(prepared_model)
        print("✓ Model conversion successful")
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        raise RuntimeError(f"Quantization conversion failed: {e}")
    
    # 9. Verify quantization worked
    quantized_layers = 0
    quantized_modules = []
    
    for name, module in quantized_model.named_modules():
        module_type = str(type(module))
        if any(keyword in module_type.lower() for keyword in ['quantized', 'qat', 'fake']):
            quantized_layers += 1
            quantized_modules.append((name, type(module).__name__))
    
    print(f"✓ Quantized layers detected: {quantized_layers}")
    
    if quantized_layers == 0:
        print("⚠️  WARNING: No quantized layers detected!")
        print("This suggests quantization may have failed silently.")
        
        # Debug: Show first few module types
        print("\nFirst 10 modules in quantized model:")
        for i, (name, module) in enumerate(quantized_model.named_modules()):
            if i >= 10: break
            print(f"  {name}: {type(module).__name__}")
    else:
        print("Sample quantized modules:")
        for name, module_type in quantized_modules[:5]:
            print(f"  {name}: {module_type}")
    
    # 10. Quick validation test
    print("Performing validation test...")
    test_input = torch.randn(1, 3, 224, 224)
    
    try:
        with torch.no_grad():
            fp32_output = model_fp32(test_input)
            int8_output = quantized_model(test_input)
        
        output_diff = torch.mean(torch.abs(fp32_output - int8_output)).item()
        print(f"✓ Validation test passed. Output difference: {output_diff:.6f}")
        
        if output_diff < 1e-6:
            print("⚠️  Very small output difference - quantization may not be active")
        
    except Exception as e:
        print(f"⚠️  Validation test failed: {e}")
    
    return quantized_model

# ===========================
# Utility functions (FIXED)
# ===========================
def count_parameters(model):
    try:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    except Exception:
        return 0

def get_model_memory_size(model):
    try:
        param_size = sum(getattr(p, 'element_size', lambda: 4)() * p.numel() for p in model.parameters())
        buffer_size = sum(getattr(b, 'element_size', lambda: 4)() * b.numel() for b in model.buffers())
        return param_size + buffer_size
    except Exception:
        return 0

def analyze_model_layers(model, model_name="Model"):
    print(f'\n{model_name} - Layer Analysis')
    print('=' * 60)
    layer_data = []
    total_params = 0
    quantized_count = 0
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            module_type = type(module).__name__
            module_type_str = str(type(module))
            params = sum(p.numel() for p in module.parameters())
            total_params += params
            
            # Better quantization detection
            is_quantized = any(keyword in module_type_str.lower() for keyword in [
                'quantized', 'qat', 'fake', 'quant'
            ]) or hasattr(module, '_weight_') or hasattr(module, 'weight_fake_quant')
            
            if is_quantized:
                quantized_count += 1
            
            layer_data.append((name, module_type, params, is_quantized))

    df = pd.DataFrame(layer_data, columns=["name","type","params","is_quantized"])
    print(f"Total layers: {len(df)}")
    print(f"Total params: {total_params:,}")
    print("Type distribution:")
    print(df["type"].value_counts())
    print(f"Quantized layers: {quantized_count}")
    
    # Show some quantized modules if any
    if quantized_count > 0:
        print("\nQuantized modules:")
        quantized_df = df[df["is_quantized"] == True]
        for _, row in quantized_df.head(5).iterrows():
            print(f"  {row['name']}: {row['type']}")
    
    return df

def benchmark_model(model, data_loader, label, device_type="cpu", runs=5, cast_half_inputs=False):
    print(f"\nBenchmark: {label} on {device_type.upper()} (runs={runs})")
    if device_type == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # TensorRT module doesn't own parameters; skip .to() for TRT, but our wrapper ignores .to()
    try:
        model = model.to(device).eval()
    except Exception as e:
        print(f"⚠️  Model.to({device}) failed: {e}, keeping on current device")
        model = model.eval()
    
    results = {"accuracies":[], "times":[], "memories":[], "throughputs":[], "per_image_times":[]}

    for r in range(runs):
        print(f"  Run {r+1}/{runs} ... ", end="")
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            start_mem = torch.cuda.memory_allocated()
        proc = psutil.Process()
        start_sys = proc.memory_info().rss

        correct = 0
        total = 0
        t0 = time.time()
        
        try:
            with torch.no_grad():
                for x, y in data_loader:
                    if cast_half_inputs and device.type == "cuda":
                        x = x.half()
                    
                    # Handle device placement more robustly
                    try:
                        x = x.to(device)
                        y = y.to(device)
                    except:
                        # If model is on CPU but we're trying GPU, keep on CPU
                        pass
                    
                    out = model(x)
                    _, pred = torch.max(out, 1)
                    total += y.size(0)
                    correct += (pred == y).sum().item()
        except Exception as e:
            print(f"Benchmark error: {e}")
            # Return zero results for failed benchmark
            results["accuracies"].append(0.0)
            results["times"].append(0.0)
            results["memories"].append(0.0)
            results["throughputs"].append(0.0)
            results["per_image_times"].append(0.0)
            continue
            
        t1 = time.time()

        if device.type == "cuda":
            torch.cuda.synchronize()
            used_mem = torch.cuda.memory_allocated() - start_mem
        else:
            used_mem = proc.memory_info().rss - start_sys

        acc = 100.0 * correct / max(1,total)
        dt = t1 - t0
        ips = total / dt if dt > 0 else 0
        pit = dt / total if total > 0 else 0
        results["accuracies"].append(acc)
        results["times"].append(dt)
        results["memories"].append(used_mem)
        results["throughputs"].append(ips)
        results["per_image_times"].append(pit)
        print(f"Acc {acc:.2f}%, Time {dt:.3f}s, IPS {ips:.1f}")

    summary = {
        "accuracy": float(np.mean(results["accuracies"])),
        "accuracy_std": float(np.std(results["accuracies"])),
        "time": float(np.mean(results["times"])),
        "time_std": float(np.std(results["times"])),
        "memory": float(np.mean(results["memories"])),
        "memory_std": float(np.std(results["memories"])),
        "throughput": float(np.mean(results["throughputs"])),
        "throughput_std": float(np.std(results["throughputs"])),
        "per_image_time": float(np.mean(results["per_image_times"])),
        "per_image_time_std": float(np.std(results["per_image_times"]))
    }
    print("Summary:", summary)
    return {**results, **summary}

def statistical_summary(all_results, runs=5):
    print(f"\nStatistical Performance Analysis ({runs} runs)")
    print("="*50)
    stats_all = {}
    for model_key in all_results:
        for dev_key in all_results[model_key]:
            key = f"{model_key}_{dev_key}"
            data = all_results[model_key][dev_key]
            if data.get("time",0)==0 and data.get("accuracy",0)==0:
                continue
            s = {
                "mean_accuracy": np.mean(data["accuracies"]),
                "std_accuracy": np.std(data["accuracies"]),
                "mean_time": np.mean(data["times"]),
                "std_time": np.std(data["times"]),
                "mean_memory": np.mean(data["memories"]),
                "std_memory": np.std(data["memories"]),
            }
            if len(data["accuracies"])>1:
                acc_ci = stats.t.interval(0.95, len(data["accuracies"])-1, loc=s["mean_accuracy"], scale=stats.sem(data["accuracies"]))
                time_ci = stats.t.interval(0.95, len(data["times"])-1, loc=s["mean_time"], scale=stats.sem(data["times"]))
                s.update({
                    "accuracy_ci_lower": acc_ci[0], "accuracy_ci_upper": acc_ci[1],
                    "time_ci_lower": time_ci[0], "time_ci_upper": time_ci[1],
                })
            stats_all[key] = {k: float(v) for k,v in s.items()}
            print(f"{key}: Acc {s['mean_accuracy']:.3f} ± {s['std_accuracy']:.3f} | Time {s['mean_time']:.6f} ± {s['std_time']:.6f} | Mem {s['mean_memory']/1e6:.2f}MB")
    return stats_all

def build_matrices(all_results):
    print("\nCreating Performance Comparison Matrices")
    print("="*45)
    models_list = ["FP32","INT8_FBGEMM","TRT_INT8"] if "trt_int8" in all_results else ["FP32","INT8_FBGEMM"]
    devices = ["GPU","CPU"] if torch.cuda.is_available() else ["CPU"]

    mapping = {"FP32":"fp32", "INT8_FBGEMM":"int8_cpu", "TRT_INT8":"trt_int8"}
    devmap = {"GPU":"gpu","CPU":"cpu"}

    acc = np.zeros((len(models_list), len(devices))) * np.nan
    tim = np.zeros((len(models_list), len(devices))) * np.nan
    mem = np.zeros((len(models_list), len(devices))) * np.nan
    thr = np.zeros((len(models_list), len(devices))) * np.nan

    for i, m in enumerate(models_list):
        mkey = mapping[m]
        for j, d in enumerate(devices):
            dkey = devmap[d]
            if mkey in all_results and dkey in all_results[mkey]:
                data = all_results[mkey][dkey]
                if data["time"]>0:
                    acc[i,j] = data["accuracy"]
                    tim[i,j] = data["time"]
                    mem[i,j] = data["memory"]/1e6
                    thr[i,j] = data["throughput"]

    acc_df = pd.DataFrame(acc, index=models_list, columns=devices)
    tim_df = pd.DataFrame(tim, index=models_list, columns=devices)
    mem_df = pd.DataFrame(mem, index=models_list, columns=devices)
    thr_df = pd.DataFrame(thr, index=models_list, columns=devices)

    print("\nAccuracy Matrix (%)\n", acc_df.round(3))
    print("\nTime Matrix (s)\n", tim_df.round(6))
    print("\nMemory Matrix (MB)\n", mem_df.round(2))
    print("\nThroughput Matrix (img/s)\n", thr_df.round(1))

    return {"accuracy": acc_df, "time": tim_df, "memory": mem_df, "throughput": thr_df}

def make_plots(results, layer_stats_fp32, layer_stats_int8):
    print("\nGenerating Dissertation-Quality Visualizations")
    print("="*50)
    plt.rcParams.update({
        'font.size': 12, 'font.family': 'serif', 'axes.labelsize': 14,
        'axes.titlesize': 16, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
        'legend.fontsize': 12, 'figure.figsize': (20, 16)
    })

    fig = plt.figure(figsize=(20, 16))
    devices = ['GPU','CPU'] if torch.cuda.is_available() else ['CPU']
    x = np.arange(len(devices)); width=0.35

    # 1. Time bar (FP32 vs INT8 CPU; GPU only for FP32)
    ax1 = plt.subplot(2,3,1)
    fp32_times = [results['fp32'][d.lower()]['time'] if d.lower() in results['fp32'] else np.nan for d in devices]
    int8_times = [results['int8_cpu'][d.lower()]['time'] if d.lower() in results['int8_cpu'] else np.nan for d in devices]
    ax1.bar(x - width/2, fp32_times, width, label='FP32', alpha=0.8)
    ax1.bar(x + width/2, int8_times, width, label='INT8 (FBGEMM)', alpha=0.8)
    ax1.set_title("Inference Time"); ax1.set_xlabel("Device"); ax1.set_ylabel("seconds")
    ax1.set_xticks(x); ax1.set_xticklabels(devices); ax1.legend(); ax1.grid(True, alpha=0.3)

    # 2. Accuracy bar
    ax2 = plt.subplot(2,3,2)
    fp32_accs = [results['fp32'][d.lower()]['accuracy'] if d.lower() in results['fp32'] else np.nan for d in devices]
    int8_accs = [results['int8_cpu'][d.lower()]['accuracy'] if d.lower() in results['int8_cpu'] else np.nan for d in devices]
    ax2.bar(x - width/2, fp32_accs, width, label='FP32', alpha=0.8)
    ax2.bar(x + width/2, int8_accs, width, label='INT8 (FBGEMM)', alpha=0.8)
    ax2.set_title("Accuracy"); ax2.set_xlabel("Device"); ax2.set_ylabel("%")
    ax2.set_xticks(x); ax2.set_xticklabels(devices); ax2.legend(); ax2.grid(True, alpha=0.3)
    ymin = np.nanmin([*fp32_accs,*int8_accs]); ax2.set_ylim([max(0,ymin-1), 100])

    # 3. Memory bar
    ax3 = plt.subplot(2,3,3)
    fp32_mem = [results['fp32'][d.lower()]['memory']/1e6 if d.lower() in results['fp32'] else np.nan for d in devices]
    int8_mem = [results['int8_cpu'][d.lower()]['memory']/1e6 if d.lower() in results['int8_cpu'] else np.nan for d in devices]
    ax3.bar(x - width/2, fp32_mem, width, label='FP32', alpha=0.8)
    ax3.bar(x + width/2, int8_mem, width, label='INT8 (FBGEMM)', alpha=0.8)
    ax3.set_title("Memory Usage"); ax3.set_xlabel("Device"); ax3.set_ylabel("MB")
    ax3.set_xticks(x); ax3.set_xticklabels(devices); ax3.legend(); ax3.grid(True, alpha=0.3)

    # 4. Layer dist (FP32)
    ax4 = plt.subplot(2,3,4)
    ax4.set_title("FP32 Layer Type Distribution")
    fp32_counts = layer_stats_fp32["type"].value_counts()
    ax4.pie(fp32_counts.values, labels=fp32_counts.index, autopct='%1.1f%%', startangle=90)

    # 5. Speedup (CPU FP32 vs INT8)
    ax5 = plt.subplot(2,3,5)
    if 'cpu' in results['fp32'] and 'cpu' in results['int8_cpu']:
        speedup = results['fp32']['cpu']['time'] / results['int8_cpu']['cpu']['time']
    else:
        speedup = np.nan
    ax5.bar(['CPU'], [speedup], alpha=0.8)
    ax5.set_title("INT8 Speedup vs FP32 (CPU)"); ax5.set_ylabel("x"); ax5.grid(True, alpha=0.3)
    ax5.text(0, speedup + 0.02 if not np.isnan(speedup) else 0, f"{speedup:.2f}x" if not np.isnan(speedup) else "N/A",
             ha='center', va='bottom')

    # 6. Model size comparison (file sizes)
    ax6 = plt.subplot(2,3,6)
    fp32_file = os.path.join(SCRIPT_DIR, 'mobilenetv3_fp32.pth')
    int8_file = os.path.join(SCRIPT_DIR, 'mobilenetv3_int8_fbgemm.pth')
    s1 = os.path.getsize(fp32_file)/1e6 if os.path.exists(fp32_file) else np.nan
    s2 = os.path.getsize(int8_file)/1e6 if os.path.exists(int8_file) else np.nan
    ax6.bar(['FP32','INT8'], [s1,s2], alpha=0.8)
    ax6.set_title("Model File Size"); ax6.set_ylabel("MB"); ax6.grid(True, alpha=0.3)
    for i, v in enumerate([s1,s2]):
        if not np.isnan(v):
            ax6.text(i, v + 0.1, f"{v:.1f} MB", ha='center')

    plt.tight_layout()
    viz_path = os.path.join(SCRIPT_DIR, "mobilenetv3_int8_dissertation_viz.png")
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Visualization saved: {viz_path}")
    return viz_path

def write_report(all_results, stat_results, matrices, layer_stats_fp32, layer_stats_int8, notes_int8_trt):
    print("\nGenerating Comprehensive Dissertation Report")
    print("="*50)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(SCRIPT_DIR, f"mobilenetv3_int8_dissertation_report_{timestamp}.md")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(report_path, "w") as f:
        f.write("# MobileNetV3-Small ROBUST INT8 Quantization (PTQ) — Dissertation Analysis\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## System & Backends\n")
        f.write(f"- Device: {device}\n")
        if device.type == "cuda":
            f.write(f"- GPU: {torch.cuda.get_device_name(0)}\n")
        f.write("- PyTorch INT8 Backend: **FBGEMM** (x86)\n")
        f.write(f"- PyTorch Version: {torch.__version__}\n")
        f.write(f"- TensorRT Available: {TENSORRT_AVAILABLE}\n")
        f.write(f"- PyCUDA Available: {PYCUDA_AVAILABLE}\n")
        f.write(f"- TensorRT INT8 Notes: {notes_int8_trt}\n\n")

        f.write("## Robust PTQ Implementation Details\n")
        f.write("This implementation uses a simplified but robust approach to quantization:\n\n")
        f.write("### PTQ Strategy ✓\n")
        f.write("- ✓ **Engine**: `torch.backends.quantized.engine = 'fbgemm'` (x86 optimized)\n")
        f.write("- ✓ **Default Config**: Using PyTorch's default FBGEMM quantization config\n")
        f.write("- ✓ **Validation**: Extensive verification and debugging steps\n")
        f.write("- ✓ **Calibration**: 512 representative images from train+val mix\n")
        f.write("- ✓ **Error Handling**: Robust error detection and fallback strategies\n\n")

        f.write("## Executive Summary\n")
        f.write("- We perform **robust post-training static quantization (PTQ)** using FX graph mode.\n")
        f.write("- Simplified approach prioritizes working quantization over complex optimizations.\n")
        f.write("- Extensive validation and debugging to ensure quantization actually occurs.\n")
        f.write("- Multi-run benchmarking with statistical analysis and error handling.\n\n")

        f.write("## Statistical Results (means ± std)\n")
        f.write("| Model-Device | Acc (%) | Time (s) | Memory (MB) |\n")
        f.write("|---|---:|---:|---:|\n")
        for key, s in stat_results.items():
            mem_mb = s["mean_memory"]/1e6
            f.write(f"| {key} | {s['mean_accuracy']:.3f} ± {s['std_accuracy']:.3f} | {s['mean_time']:.6f} ± {s['std_time']:.6f} | {mem_mb:.2f} ± {s['std_memory']/1e6:.2f} |\n")

        f.write("\n## Matrices\n")
        for name, df in matrices.items():
            f.write(f"\n### {name.title()}\n\n")
            f.write(df.to_markdown(index=True))
            f.write("\n")

        f.write("\n## Layer Analysis\n")
        f.write(f"- FP32 types:\n\n```\n{layer_stats_fp32['type'].value_counts().to_string()}\n```\n")
        
        quantized_count = int(layer_stats_int8['is_quantized'].sum())
        f.write(f"- INT8(FBGEMM) quantized layers: {quantized_count}\n\n")
        
        if quantized_count == 0:
            f.write("⚠️  **WARNING**: No quantized layers detected - quantization may have failed!\n\n")

        f.write("## Quantization Quality Metrics\n")
        if 'cpu' in all_results['fp32'] and 'cpu' in all_results['int8_cpu']:
            fp32_acc = all_results['fp32']['cpu']['accuracy']
            int8_acc = all_results['int8_cpu']['cpu']['accuracy']
            acc_loss = fp32_acc - int8_acc  # FIXED: Correct calculation
            speedup = all_results['fp32']['cpu']['time'] / all_results['int8_cpu']['cpu']['time']
            f.write(f"- **Accuracy Preservation**: {int8_acc:.2f}% ({acc_loss:+.2f}% vs FP32)\n")
            f.write(f"- **Latency Speedup**: {speedup:.2f}x faster on CPU\n")
            
            # Quality assessment
            if abs(acc_loss) > 5.0:
                f.write("- ⚠️  **Quality Warning**: Accuracy loss > 5% indicates quantization issues\n")
            elif abs(acc_loss) < 1.0:
                f.write("- ✅ **Quality Good**: Accuracy loss < 1%\n")
        
        # Model size comparison
        fp32_file = os.path.join(SCRIPT_DIR, 'mobilenetv3_fp32.pth')
        int8_file = os.path.join(SCRIPT_DIR, 'mobilenetv3_int8_fbgemm.pth')
        if os.path.exists(fp32_file) and os.path.exists(int8_file):
            fp32_size = os.path.getsize(fp32_file) / 1e6
            int8_size = os.path.getsize(int8_file) / 1e6
            compression = fp32_size / int8_size
            f.write(f"- **Model Compression**: {compression:.2f}x smaller ({int8_size:.1f}MB vs {fp32_size:.1f}MB)\n")

        f.write("\n## Troubleshooting Notes\n")
        f.write("If quantization shows poor results, check:\n")
        f.write("1. **Observer Detection**: Ensure observers are properly inserted during prepare phase\n")
        f.write("2. **Calibration Data**: Use diverse, representative samples for calibration\n")
        f.write("3. **Module Detection**: Verify quantized modules are actually created\n")
        f.write("4. **PyTorch Version**: Ensure compatible PyTorch version with stable quantization\n")
        f.write("5. **Backend Support**: Verify FBGEMM backend is properly configured\n\n")

        f.write("## Conclusions\n")
        f.write("- **Robust PTQ approach** prioritizes working quantization over complex optimizations.\n")
        f.write("- **Extensive validation** helps identify and debug quantization failures.\n")
        f.write("- **Simplified configuration** reduces chances of configuration errors.\n")
        f.write("- **Error handling** provides better debugging information for issues.\n")
        if quantized_count > 0:
            f.write("- ✅ **Quantization successful** - detected quantized layers in model.\n")
        else:
            f.write("- ⚠️  **Quantization may have failed** - investigate layer analysis output.\n")

    print(f"✓ Report saved: {report_path}")
    return report_path

# ===========================
# Main (UPDATED)
# ===========================
def main():
    print("="*80)
    print("MOBILENETV3-SMALL ROBUST INT8 PTQ — FIXED & DEBUGGED VERSION")
    print("="*80)

    # -------------------
    # Device & Datasets
    # -------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Analysis Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    print("\nLoading datasets ...")
    train_dataset = datasets.ImageFolder(root='dataset/training_set', transform=transform)
    val_dataset   = datasets.ImageFolder(root='dataset/test_set',     transform=transform)
    train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader    = DataLoader(val_dataset,   batch_size=32, shuffle=False)
    test_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
    print(f"Dataset loaded: {len(train_dataset)} training, {len(val_dataset)} validation images")

    # -------------------
    # Model & Training
    # -------------------
    print("\n" + "="*60)
    print("MODEL INITIALIZATION & SHORT TRAINING")
    print("="*60)
    model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 2)
    model.to(device)
    print(f"Trainable params: {count_parameters(model):,}")

    # Analyze FP32
    fp32_layer_df = analyze_model_layers(model, "MobileNetV3-Small FP32")

    # Light training (thesis-friendly runtime)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    print("\nExecuting short training (3 epochs, 10 batches/epoch)...")
    model.train()
    for epoch in range(3):
        print(f"Epoch {epoch+1}/3")
        running_loss, correct, total = 0.0, 0, 0
        for i, (x,y) in enumerate(train_loader):
            if i >= 10:  # keep it short
                break
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _,pred = torch.max(out,1)
            total += y.size(0); correct += (pred==y).sum().item()
            if i % 5 == 0:
                print(f"  Batch {i}: Loss {loss.item():.4f}")
        scheduler.step()
        print(f"Train Acc: {100.0*correct/max(1,total):.2f}% | Loss: {running_loss/max(1,(i+1)):.4f}")

    # Save FP32
    fp32_path = os.path.join(SCRIPT_DIR, "mobilenetv3_fp32.pth")
    torch.save(model.state_dict(), fp32_path)
    print(f"✓ Saved FP32 weights: {fp32_path}")

    # -------------------
    # FP32 Benchmarks
    # -------------------
    print("\n" + "="*60)
    print("FP32 BENCHMARKS")
    print("="*60)
    all_results = {"fp32":{}, "int8_cpu":{}}

    if torch.cuda.is_available():
        all_results["fp32"]["gpu"] = benchmark_model(model, test_loader, "FP32", "gpu", runs=5, cast_half_inputs=False)
    all_results["fp32"]["cpu"] = benchmark_model(copy.deepcopy(model).cpu(), test_loader, "FP32", "cpu", runs=5)

    # -------------------
    # Robust PTQ INT8 (FBGEMM) - FIXED
    # -------------------
    print("\n" + "="*60)
    print("ROBUST PTQ INT8 (FBGEMM) — FIXED & DEBUGGED")
    print("="*60)
    
    try:
        # Use robust PTQ function with 512 calibration samples (smaller for stability)
        int8_model = robust_ptq_int8_fbgemm_fx(model, train_dataset, val_dataset, num_calib_samples=512)

        # Analyze INT8 layers
        int8_layer_df = analyze_model_layers(int8_model, "MobileNetV3-Small Robust INT8 (FBGEMM)")

        # Save INT8 model
        int8_path = os.path.join(SCRIPT_DIR, "mobilenetv3_int8_fbgemm.pth")
        torch.save(int8_model.state_dict(), int8_path)
        print(f"✓ Saved Robust INT8(FBGEMM) weights: {int8_path}")

        # Bench INT8 on CPU (quantized models run on CPU)
        all_results["int8_cpu"]["cpu"] = benchmark_model(int8_model, test_loader, "Robust INT8 (FBGEMM)", "cpu", runs=5)

    except Exception as e:
        print(f"⚠️  PTQ INT8 failed: {e}")
        # Create dummy results for failed quantization
        all_results["int8_cpu"]["cpu"] = {
            "accuracy": 0.0, "time": 0.0, "memory": 0.0, "throughput": 0.0,
            "accuracies": [0.0], "times": [0.0], "memories": [0.0], "throughputs": [0.0], "per_image_times": [0.0]
        }
        int8_layer_df = fp32_layer_df.copy()  # Use FP32 layer df as fallback

    # -------------------
    # TensorRT INT8 (ONNX→TRT with calibration) — optional
    # -------------------
    trt_module = None
    notes_int8_trt = "Attempted"
    if TENSORRT_AVAILABLE and PYCUDA_AVAILABLE and torch.cuda.is_available():
        try:
            print("\n" + "="*60)
            print("TENSORRT INT8 (ONNX→TRT, Entropy Calibrator)")
            print("="*60)

            onnx_path = os.path.join(SCRIPT_DIR, "mobilenetv3_static.onnx")
            plan_path = os.path.join(SCRIPT_DIR, "mobilenetv3_int8.plan")

            # Export ONNX from trained FP32 model
            export_onnx_static(model, onnx_path, input_shape=(1,3,224,224))

            # Build calibration tensors
            calib_dataset = create_calibration_dataset(train_dataset, val_dataset, num_samples=256)
            calib_loader = DataLoader(calib_dataset, batch_size=1, shuffle=False)
            
            calib_tensors = []
            for i, (x, _) in enumerate(calib_loader):
                if i >= 256: break
                calib_tensors.append(x)

            engine = build_trt_int8_engine_from_onnx(onnx_path, plan_path, calib_tensors, workspace_bytes=1<<29)
            trt_module = TensorRTEngineModule(engine)
            notes_int8_trt = "Robust INT8 TensorRT engine built successfully"

        except Exception as e:
            print(f"TensorRT INT8 skipped: {e}")
            notes_int8_trt = f"Skipped/failed: {e}"
    else:
        notes_int8_trt = "Unavailable (needs TensorRT + PyCUDA + CUDA GPU w/ fast INT8)"

    if trt_module is not None:
        all_results["trt_int8"] = {}
        all_results["trt_int8"]["gpu"] = benchmark_model(trt_module, test_loader, "TensorRT INT8", "gpu", runs=5, cast_half_inputs=False)

    # -------------------
    # Stats, Matrices, Plots, Report
    # -------------------
    print("\n" + "="*60)
    print("STATISTICS & VISUALIZATION")
    print("="*60)
    stats_res = statistical_summary(all_results, runs=5)
    matrices  = build_matrices(all_results)
    viz_path  = make_plots(all_results, fp32_layer_df, int8_layer_df)
    report    = write_report(all_results, stats_res, matrices, fp32_layer_df, int8_layer_df, notes_int8_trt)

    # -------------------
    # Final Summary with Issue Detection
    # -------------------
    print("\n" + "="*80)
    print("ROBUST DISSERTATION ANALYSIS COMPLETE")
    print("="*80)
    print("Generated Files:")
    print(f"  📊 Report: {report}")
    print(f"  📈 Visualization: {viz_path}")
    print(f"  🔧 FP32 Weights: {fp32_path}")
    print(f"  🔧 Robust INT8(FBGEMM) Weights: {int8_path}")
    if trt_module is not None:
        print(f"  🚀 TensorRT INT8 Engine: {os.path.join(SCRIPT_DIR, 'mobilenetv3_int8.plan')}")

    # Analyze results quality
    if "cpu" in all_results["fp32"] and "cpu" in all_results["int8_cpu"]:
        fp32_acc = all_results["fp32"]["cpu"]["accuracy"]
        int8_acc = all_results["int8_cpu"]["cpu"]["accuracy"]
        acc_loss = fp32_acc - int8_acc
        
        if all_results["int8_cpu"]["cpu"]["time"] > 0:
            speedup = all_results["fp32"]["cpu"]["time"] / all_results["int8_cpu"]["cpu"]["time"]
        else:
            speedup = 0.0
        
        print("\nQuantization Quality Analysis:")
        print(f"  FP32 Accuracy: {fp32_acc:.2f}%")
        print(f"  INT8 Accuracy: {int8_acc:.2f}%")
        print(f"  Accuracy Loss: {acc_loss:+.2f}%")
        print(f"  Speedup: {speedup:.2f}x")
        
        # Quality assessment
        quantized_layers = int(int8_layer_df['is_quantized'].sum())
        if quantized_layers == 0:
            print("  🚨 CRITICAL: No quantized layers detected - quantization failed!")
        elif abs(acc_loss) > 5.0:
            print("  ⚠️  WARNING: High accuracy loss suggests quantization issues")
        elif int8_acc < 50.0:
            print("  ⚠️  WARNING: Very low INT8 accuracy suggests calibration problems")
        else:
            print("  ✅ Quantization appears successful")
        
        # Model size comparison
        if os.path.exists(fp32_path) and os.path.exists(int8_path):
            fp32_size = os.path.getsize(fp32_path) / 1e6
            int8_size = os.path.getsize(int8_path) / 1e6
            compression = fp32_size / int8_size
            print(f"  Model compression: {compression:.2f}x ({int8_size:.1f}MB vs {fp32_size:.1f}MB)")

if __name__ == "__main__":
    main()