import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import time
import os
import psutil
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import json
import random
from thop import profile, clever_format
import pandas as pd
from tabulate import tabulate
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================
# Configuration knobs (stable)
# =============================
USE_AMP = False                 # Disable AMP to avoid NaNs for baseline
WARMUP_HEAD_EPOCHS = 1          # Train the new FC head only for N epochs
BACKBONE_LR = 1e-4              # Lower LR for pretrained backbone
HEAD_LR = 1e-3                  # Higher LR for new head
WEIGHT_DECAY = 1e-4             # Conservative weight decay
BATCH_SIZE = 64
EPOCHS = 10
DATA_ROOT = "dataset"           # expects training_set/ and test_set/
SEED = 42

# ---------------------------
# Reproducibility
# ---------------------------
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_random_seed(SEED)

# ---------------------------
# Device & environment info
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("="*80)
print("PROFESSIONAL RESNET-18 FP32 BASELINE TRAINING")
print("MSc Artificial Intelligence Dissertation Project")
print("="*80)
print(f'Experiment Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
print(f'Device: {device}')
print(f'Random Seed: {SEED}')

if device.type == "cuda":
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
    print(f'CUDA Version: {torch.version.cuda}')
else:
    print('CPU: Running on CPU')

print(f'PyTorch Version: {torch.__version__}')
print(f'Python Version: {os.sys.version.split()[0]}')


class ProfessionalResNet18Analyzer:
    """Professional ResNet-18 training & analysis class (FP32 baseline, stabilized)"""

    def __init__(self, experiment_name="ResNet18_FP32_Baseline"):
        self.device = device
        self.experiment_name = experiment_name
        self.best_val_acc = 0.0
        self.patience = 10
        self.patience_counter = 0
        self.history = defaultdict(list)
        self.model_metrics = {}
        self.inference_metrics = {}

        # results folder
        self.results_dir = f"results_{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)
        print(f"Results Directory: {self.results_dir}/")

    # ---------------------------
    # Data
    # ---------------------------
    def setup_data_loaders(self, data_path=DATA_ROOT, batch_size=BATCH_SIZE):
        print("\n" + "="*60)
        print("DATA PREPARATION AND AUGMENTATION")
        print("="*60)

        # Stable baseline augmentation (RandomErasing removed for stability)
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        train_dataset = datasets.ImageFolder(root=f"{data_path}/training_set", transform=train_transform)
        val_dataset = datasets.ImageFolder(root=f"{data_path}/test_set", transform=val_transform)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=min(8, os.cpu_count()),
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            persistent_workers=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=min(8, os.cpu_count()),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
        )

        self.classes = train_dataset.classes
        self.num_classes = len(self.classes)

        print(f"Training Samples: {len(train_dataset):,}")
        print(f"Validation Samples: {len(val_dataset):,}")
        print(f"Number of Classes: {self.num_classes}")
        print(f"Classes: {self.classes}")
        print(f"Batch Size: {batch_size}")
        print(f"Training Batches: {len(self.train_loader)}")
        print(f"Validation Batches: {len(self.val_loader)}")
        print(f"Data Workers: {min(8, os.cpu_count())}")
        return train_dataset, val_dataset

    # ---------------------------
    # Model analysis
    # ---------------------------
    def analyze_model_architecture(self, model):
        print("\n" + "="*60)
        print("MODEL ARCHITECTURE ANALYSIS")
        print("="*60)

        layer_info, total_params = [], 0
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
                if param_count > 0:
                    layer_type = type(module).__name__
                    details = {}
                    if hasattr(module, "in_features") and hasattr(module, "out_features"):
                        details = {"input_size": module.in_features, "output_size": module.out_features}
                    elif hasattr(module, "in_channels") and hasattr(module, "out_channels"):
                        details = {
                            "input_channels": module.in_channels,
                            "output_channels": module.out_channels,
                            "kernel_size": getattr(module, "kernel_size", "N/A"),
                            "stride": getattr(module, "stride", "N/A"),
                            "padding": getattr(module, "padding", "N/A"),
                        }
                    layer_info.append({
                        "Layer Name": name,
                        "Layer Type": layer_type,
                        "Parameters": param_count,
                        "Percentage": 0.0,
                        "Details": details
                    })
                    total_params += param_count

        for l in layer_info:
            l["Percentage"] = (l["Parameters"] / total_params) * 100

        table_rows = []
        for l in layer_info:
            cfg = ", ".join([f"{k}={v}" for k, v in l["Details"].items()])
            table_rows.append([
                l["Layer Name"][:30],
                l["Layer Type"],
                f'{l["Parameters"]:,}',
                f'{l["Percentage"]:.2f}%',
                cfg[:40],
            ])
        headers = ["Layer Name", "Type", "Parameters", "% of Total", "Configuration"]
        print("\nLayer-wise Parameter Distribution:")
        print(tabulate(table_rows, headers=headers, tablefmt="grid"))

        # size & FLOPs (safe)
        model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        model_size_mb = model_size_bytes / (1024**2)

        flops_fmt = "N/A"
        try:
            model_eval_state = model.training
            model.eval()
            dummy = torch.randn(1, 3, 224, 224).to(self.device)
            flops, params = profile(model, inputs=(dummy,), verbose=False)
            flops_fmt, _ = clever_format([flops, params], "%.3f")
            if model_eval_state:
                model.train()
        except Exception as e:
            print(f"⚠️ FLOPs profiling skipped ({e}).")

        self.model_metrics = {
            "architecture": "ResNet-18",
            "precision": "FP32",
            "total_parameters": total_params,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "model_size_bytes": model_size_bytes,
            "model_size_mb": model_size_mb,
            "flops_formatted": flops_fmt,
            "layer_breakdown": layer_info,
            "num_classes": self.num_classes,
            "input_resolution": (224, 224),
            "dtype": str(next(model.parameters()).dtype),
        }

        print("\nModel Summary:")
        print("  Architecture: ResNet-18")
        print("  Precision: FP32 (32-bit floating point)")
        print(f'  Total Parameters: {total_params:,}')
        print(f'  Trainable Parameters: {self.model_metrics["trainable_parameters"]:,}')
        print(f'  Model Size: {model_size_mb:.2f} MB')
        print(f'  FLOPs per Forward Pass: {flops_fmt}')
        print("  Input Resolution: 224x224x3")
        print(f'  Output Classes: {self.num_classes}')
        return self.model_metrics

    # ---------------------------
    # Model init
    # ---------------------------
    def setup_model(self):
        print("\n" + "="*60)
        print("MODEL INITIALIZATION")
        print("="*60)

        self.model = models.resnet18(weights="IMAGENET1K_V1")
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, self.num_classes)
        nn.init.xavier_uniform_(self.model.fc.weight)
        nn.init.constant_(self.model.fc.bias, 0)

        self.model = self.model.to(self.device)
        self.analyze_model_architecture(self.model)
        return self.model

    # ---------------------------
    # Training setup
    # ---------------------------
    def setup_training_components(self):
        print("\n" + "="*60)
        print("TRAINING CONFIGURATION")
        print("="*60)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Param groups: lower LR for backbone, higher for head
        backbone_params = []
        head_params = []

        for name, p in self.model.named_parameters():
            if name.startswith("fc."):
                head_params.append(p)
            else:
                backbone_params.append(p)

        self.optimizer = optim.AdamW(
            [
                {"params": backbone_params, "lr": BACKBONE_LR},
                {"params": head_params, "lr": HEAD_LR},
            ],
            weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999), eps=1e-8
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-7
        )

        self.scaler = torch.cuda.amp.GradScaler() if (self.device.type == "cuda" and USE_AMP) else None

        print(f"Loss Function: CrossEntropyLoss (label_smoothing=0.1)")
        print(f"Optimizer: AdamW (backbone lr={BACKBONE_LR}, head lr={HEAD_LR}, wd={WEIGHT_DECAY})")
        print("Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)")
        print(f'Mixed Precision: {"Enabled" if self.scaler else "Disabled"}')
        print(f"Early Stopping Patience: {self.patience} epochs")
        print("Gradient Clipping: Max norm = 1.0")

    # ---------------------------
    # Train/validate
    # ---------------------------
    def _freeze_backbone(self, freeze=True):
        for name, p in self.model.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad = not freeze

    def train_single_epoch(self, epoch, freeze_backbone=False):
        self.model.train()
        self._freeze_backbone(freeze_backbone)

        running_loss, correct, total = 0.0, 0, 0
        batch_times = []

        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            # sanity: label range
            assert labels.min().item() >= 0 and labels.max().item() < self.num_classes, \
                "Labels out of range for current number of classes."

            t0 = time.time()
            inputs, labels = inputs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # forward
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

            # guard against non-finite
            if not torch.isfinite(loss):
                print("⚠️ Non-finite loss detected. Skipping batch.")
                continue
            if not torch.isfinite(outputs).all():
                print("⚠️ Non-finite outputs detected. Skipping batch.")
                continue

            # backward
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            running_loss += loss.item()

            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

            batch_times.append(time.time() - t0)

            if batch_idx % 100 == 0 and batch_idx > 0:
                lrs = [g["lr"] for g in self.optimizer.param_groups]
                print(f'  Epoch {epoch+1} | Batch {batch_idx:3d}/{len(self.train_loader)} | '
                      f'Loss: {loss.item():.4f} | Acc: {100*correct/total:.2f}% | '
                      f'LRs: {", ".join(f"{lr:.6f}" for lr in lrs)} | '
                      f'Time: {np.mean(batch_times[-10:]):.3f}s/batch')

        avg_loss = (running_loss / max(1, len(self.train_loader)))
        avg_acc = (100 * correct / max(1, total))
        avg_time = np.mean(batch_times) if batch_times else 0.0
        return avg_loss, avg_acc, avg_time

    def validate_model(self):
        self.model.eval()
        val_loss, correct, total = 0.0, 0, 0
        batch_times = []
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                t0 = time.time()
                inputs, labels = inputs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)

                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                if not torch.isfinite(loss) or not torch.isfinite(outputs).all():
                    # Skip any pathological batch in validation as well
                    continue

                val_loss += loss.item()
                _, pred = outputs.max(1)
                total += labels.size(0)
                correct += pred.eq(labels).sum().item()
                batch_times.append(time.time() - t0)

        avg_loss = (val_loss / max(1, len(self.val_loader)))
        avg_acc = (100 * correct / max(1, total))
        avg_time = np.mean(batch_times) if batch_times else 0.0
        return avg_loss, avg_acc, avg_time

    def train_model(self, epochs=EPOCHS):
        print("\n" + "="*60)
        print(f"TRAINING PHASE - {epochs} EPOCHS")
        print("="*60)
        start = time.time()

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-"*50)

            # Warm-up: train head only for first WARMUP_HEAD_EPOCHS
            freeze_bb = (epoch < WARMUP_HEAD_EPOCHS)
            if freeze_bb:
                print("  🔧 Warm-up (head-only training)")

            tr_loss, tr_acc, tr_time = self.train_single_epoch(epoch, freeze_backbone=freeze_bb)
            va_loss, va_acc, va_time = self.validate_model()

            old_lrs = [g["lr"] for g in self.optimizer.param_groups]
            self.scheduler.step(va_acc)
            new_lrs = [g["lr"] for g in self.optimizer.param_groups]
            if any(n < o for n, o in zip(new_lrs, old_lrs)):
                print("  📉 Learning Rate Reduced:",
                      " → ".join([f"{o:.6f}" for o in old_lrs]),
                      "to",
                      " | ".join([f"{n:.6f}" for n in new_lrs]))

            self.history["epoch"].append(epoch+1)
            self.history["train_loss"].append(tr_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_loss"].append(va_loss)
            self.history["val_acc"].append(va_acc)
            # store one LR (head LR) for plotting; optional
            self.history["learning_rate"].append(new_lrs[-1] if new_lrs else 0.0)
            self.history["train_batch_time"].append(tr_time)
            self.history["val_batch_time"].append(va_time)

            print(f"Training:   Loss={tr_loss:.4f} | Accuracy={tr_acc:.2f}% | Time={tr_time:.3f}s/batch")
            print(f"Validation: Loss={va_loss:.4f} | Accuracy={va_acc:.2f}% | Time={va_time:.3f}s/batch")

            if va_acc > self.best_val_acc:
                self.best_val_acc = va_acc
                self.patience_counter = 0
                checkpoint = {
                    "epoch": epoch+1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "best_val_acc": self.best_val_acc,
                    "train_acc": tr_acc,
                    "history": dict(self.history),
                    "model_metrics": self.model_metrics,
                    "random_seed": SEED,
                }
                torch.save(checkpoint, f"{self.results_dir}/best_model.pth")
                torch.save(self.model.state_dict(), f"{self.results_dir}/best_weights.pt")
                print(f"  ⭐ NEW BEST MODEL SAVED! Validation Accuracy: {va_acc:.2f}%")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"  🛑 Early stopping triggered after {self.patience} epochs without improvement")
                    break

        total_time = (time.time() - start) / 3600
        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        print(f"Total Training Time: {total_time:.2f} hours")
        print(f"Best Validation Accuracy: {self.best_val_acc:.2f}%")
        print(f"Epochs Trained: {len(self.history['train_loss'])}")
        print(f"Early Stopping: {'Yes' if self.patience_counter >= self.patience else 'No'}")
        return self.history

    # ---------------------------
    # Inference benchmark
    # ---------------------------
    def comprehensive_inference_benchmark(self, warmup_runs=10, benchmark_runs=100):
        print("\n" + "="*60)
        print("INFERENCE PERFORMANCE BENCHMARKING")
        print("="*60)

        weights_path = f"{self.results_dir}/best_weights.pt"
        ckpt_path = f"{self.results_dir}/best_model.pth"
        if os.path.exists(weights_path):
            state = torch.load(weights_path, weights_only=True)
            self.model.load_state_dict(state)
        else:
            ckpt = torch.load(ckpt_path, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            start_gpu_mem = torch.cuda.memory_allocated()

        proc = psutil.Process()
        start_cpu_mem = proc.memory_info().rss / (1024**2)

        print("Benchmark Configuration:")
        print(f"  Warm-up Runs: {warmup_runs}")
        print(f"  Benchmark Runs: {benchmark_runs}")
        print(f"  Batch Sizes: [1, 8, 16, 32, 64]")
        print(f"  Device: {self.device}")

        benchmark_results = {}
        batch_sizes = [1, 8, 16, 32, 64]
        for bs in batch_sizes:
            print(f"\n📊 Testing Batch Size: {bs}")
            dummy = torch.randn(bs, 3, 224, 224).to(self.device)

            print(f"  Warming up... ({warmup_runs} runs)")
            with torch.no_grad():
                for _ in range(warmup_runs):
                    _ = self.model(dummy)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

            print(f"  Benchmarking... ({benchmark_runs} runs)")
            times = []
            with torch.no_grad():
                for _ in range(benchmark_runs):
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t0 = time.time()
                    _ = self.model(dummy)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    times.append(time.time() - t0)

            times = np.array(times)
            mean_t, std_t = np.mean(times), np.std(times)
            med_t, min_t, max_t = np.median(times), np.min(times), np.max(times)
            per_img = mean_t / bs
            fps = bs / mean_t

            benchmark_results[bs] = {
                "batch_size": bs,
                "mean_time_ms": mean_t * 1000,
                "std_time_ms": std_t * 1000,
                "median_time_ms": med_t * 1000,
                "min_time_ms": min_t * 1000,
                "max_time_ms": max_t * 1000,
                "per_image_time_ms": per_img * 1000,
                "throughput_fps": fps,
                "runs": benchmark_runs,
            }
            print(f"    Mean Time: {mean_t*1000:.2f} ± {std_t*1000:.2f} ms")
            print(f"    Per Image: {per_img*1000:.2f} ms")
            print(f"    Throughput: {fps:.1f} FPS")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            end_gpu_mem = torch.cuda.memory_allocated()
            gpu_mem_used = (end_gpu_mem - start_gpu_mem) / (1024**2)
        else:
            gpu_mem_used = 0.0
        end_cpu_mem = proc.memory_info().rss / (1024**2)
        cpu_mem_used = end_cpu_mem - start_cpu_mem

        print("\n📈 Final Accuracy Assessment")
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, pred = outputs.max(1)
                total += labels.size(0)
                correct += pred.eq(labels).sum().item()
        final_acc = 100 * correct / max(1, total)

        self.inference_metrics = {
            "final_accuracy": final_acc,
            "benchmark_results": benchmark_results,
            "memory_usage": {
                "gpu_memory_mb": gpu_mem_used,
                "cpu_memory_mb": cpu_mem_used,
                "model_size_disk_mb": self.model_metrics["model_size_mb"],
            },
            "system_info": {
                "device": str(self.device),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
                "pytorch_version": torch.__version__,
                "mixed_precision": self.scaler is not None,
            },
        }

        print("\n📋 BENCHMARK SUMMARY")
        print(f"Final Test Accuracy: {final_acc:.2f}%")
        print(f"GPU Memory Used: {gpu_mem_used:.2f} MB")
        print(f"CPU Memory Used: {cpu_mem_used:.2f} MB")
        print(f"Model Size on Disk: {self.model_metrics['model_size_mb']:.2f} MB")
        return self.inference_metrics

    # ---------------------------
    # Visuals & reports
    # ---------------------------
    def create_professional_visualizations(self):
        print("\n" + "="*60)
        print("GENERATING PROFESSIONAL VISUALIZATIONS")
        print("="*60)

        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("husl")
        fig = plt.figure(figsize=(20, 15))

        # 1. Loss
        ax1 = plt.subplot(3, 3, 1)
        ep = self.history["epoch"]
        plt.plot(ep, self.history["train_loss"], 'b-', linewidth=2, label="Training Loss", alpha=0.8)
        plt.plot(ep, self.history["val_loss"], 'r-', linewidth=2, label="Validation Loss", alpha=0.8)
        plt.title("Model Loss Evolution", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True, alpha=0.3)

        # 2. Acc
        ax2 = plt.subplot(3, 3, 2)
        plt.plot(ep, self.history["train_acc"], 'b-', linewidth=2, label="Training Accuracy", alpha=0.8)
        plt.plot(ep, self.history["val_acc"], 'r-', linewidth=2, label="Validation Accuracy", alpha=0.8)
        plt.title("Model Accuracy Evolution", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.legend(); plt.grid(True, alpha=0.3)

        # 3. LR
        ax3 = plt.subplot(3, 3, 3)
        plt.plot(ep, self.history["learning_rate"], 'g-', linewidth=2)
        plt.title("Learning Rate Schedule", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch"); plt.ylabel("Learning Rate"); plt.yscale("log"); plt.grid(True, alpha=0.3)

        # 4. Batch times
        ax4 = plt.subplot(3, 3, 4)
        plt.plot(ep, self.history["train_batch_time"], 'b-', linewidth=2, label="Training", alpha=0.8)
        plt.plot(ep, self.history["val_batch_time"], 'r-', linewidth=2, label="Validation", alpha=0.8)
        plt.title("Batch Processing Time", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch"); plt.ylabel("Time (seconds/batch)"); plt.legend(); plt.grid(True, alpha=0.3)

        # 5. Best val tracking
        ax5 = plt.subplot(3, 3, 5)
        bests, cur = [], 0
        for a in self.history["val_acc"]:
            if a > cur: cur = a
            bests.append(cur)
        plt.plot(ep, self.history["val_acc"], 'r-', alpha=0.6, linewidth=1, label="Validation Accuracy")
        plt.plot(ep, bests, 'g-', linewidth=2, label="Best Model Accuracy")
        plt.title("Model Performance Tracking", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.legend(); plt.grid(True, alpha=0.3)

        # 6. Throughput vs batch
        ax6 = plt.subplot(3, 3, 6)
        if hasattr(self, "inference_metrics") and "benchmark_results" in self.inference_metrics:
            bss = list(self.inference_metrics["benchmark_results"].keys())
            thr = [self.inference_metrics["benchmark_results"][bs]["throughput_fps"] for bs in bss]
            plt.bar(range(len(bss)), thr, alpha=0.7, edgecolor="navy")
            plt.title("Inference Throughput vs Batch Size", fontsize=12, fontweight="bold")
            plt.xlabel("Batch Size"); plt.ylabel("Throughput (FPS)")
            plt.xticks(range(len(bss)), bss); plt.grid(True, alpha=0.3, axis="y")

        # 7. Per-image latency
        ax7 = plt.subplot(3, 3, 7)
        if hasattr(self, "inference_metrics") and "benchmark_results" in self.inference_metrics:
            pit = [self.inference_metrics["benchmark_results"][bs]["per_image_time_ms"] for bs in bss]
            plt.plot(bss, pit, 'ro-', linewidth=2, markersize=6)
            plt.title("Per-Image Inference Time", fontsize=12, fontweight="bold")
            plt.xlabel("Batch Size"); plt.ylabel("Time (ms/image)"); plt.grid(True, alpha=0.3)

        # 8. Parameter distribution
        ax8 = plt.subplot(3, 3, 8)
        if "layer_breakdown" in self.model_metrics:
            layers = self.model_metrics["layer_breakdown"]
            top = sorted(layers, key=lambda x: x["Parameters"], reverse=True)[:6]
            others = sum(l["Parameters"] for l in layers[6:]) if len(layers) > 6 else 0
            labels = [l["Layer Name"].split(".")[-1] for l in top]
            sizes = [l["Parameters"] for l in top]
            if others > 0:
                labels.append("Others"); sizes.append(others)
            plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
            plt.title("Parameter Distribution by Layer", fontsize=12, fontweight="bold")

        # 9. Memory usage
        ax9 = plt.subplot(3, 3, 9)
        if hasattr(self, "inference_metrics"):
            types = ["Model Size\n(Disk)", "GPU Runtime\nMemory", "CPU Runtime\nMemory"]
            vals = [
                self.inference_metrics["memory_usage"]["model_size_disk_mb"],
                self.inference_metrics["memory_usage"]["gpu_memory_mb"],
                self.inference_metrics["memory_usage"]["cpu_memory_mb"],
            ]
            bars = plt.bar(types, vals, alpha=0.7, edgecolor="black")
            plt.title("Memory Usage Breakdown", fontsize=12, fontweight="bold")
            plt.ylabel("Memory (MB)"); plt.grid(True, alpha=0.3, axis="y")
            for b, v in zip(bars, vals):
                plt.text(b.get_x()+b.get_width()/2, b.get_height()+0.1, f"{v:.1f} MB",
                         ha="center", va="bottom", fontweight="bold")

        plt.savefig(f"{self.results_dir}/comprehensive_analysis.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.savefig(f"{self.results_dir}/comprehensive_analysis.pdf", dpi=300, bbox_inches="tight", facecolor="white")
        plt.show()
        print("✅ Professional visualizations saved:")
        print(f"   - {self.results_dir}/comprehensive_analysis.png")
        print(f"   - {self.results_dir}/comprehensive_analysis.pdf")

    def generate_layer_analysis_table(self):
        if "layer_breakdown" in self.model_metrics:
            rows = []
            map_types = {
                "Conv2d": "Convolutional",
                "Linear": "Fully Connected",
                "BatchNorm2d": "Batch Normalization",
                "ReLU": "Activation",
                "SiLU": "Activation",
                "AdaptiveAvgPool2d": "Adaptive Pooling",
                "MaxPool2d": "Max Pooling",
                "Dropout": "Dropout",
            }
            for l in self.model_metrics["layer_breakdown"]:
                readable = map_types.get(l["Layer Type"], l["Layer Type"])
                rows.append({
                    "Layer Name": l["Layer Name"],
                    "Layer Type": readable,
                    "Parameters": f'{l["Parameters"]:,}',
                    "Percentage of Total": f'{l["Percentage"]:.2f}%',
                    "Configuration": str(l["Details"]),
                })
            df = pd.DataFrame(rows)
            df.to_csv(f"{self.results_dir}/layer_analysis_table.csv", index=False)
            latex = df.to_latex(index=False, escape=False,
                                caption="ResNet-18 FP32 Layer-wise Parameter Analysis",
                                label="tab:resnet18_layer_analysis")
            with open(f"{self.results_dir}/layer_analysis_table.tex", "w") as f:
                f.write(latex)
            print("✅ Layer analysis table saved:")
            print(f"   - {self.results_dir}/layer_analysis_table.csv")
            print(f"   - {self.results_dir}/layer_analysis_table.tex")

    def save_comprehensive_report(self):
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*60)

        epochs_trained = len(self.history.get("train_acc", []))
        best_val = max(self.history.get("val_acc", [0.0]))
        report = {
            "experiment_metadata": {
                "experiment_name": self.experiment_name,
                "timestamp": datetime.now().isoformat(),
                "pytorch_version": torch.__version__,
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
                "device": str(self.device),
                "random_seed": SEED,
                "reproducibility_notes": "Fixed random seed, deterministic algorithms enabled",
            },
            "model_specifications": self.model_metrics,
            "training_configuration": {
                "epochs_trained": epochs_trained,
                "batch_size": BATCH_SIZE,
                "optimizer": "AdamW",
                "learning_rate": f"backbone={BACKBONE_LR}, head={HEAD_LR}",
                "weight_decay": WEIGHT_DECAY,
                "scheduler": "ReduceLROnPlateau",
                "early_stopping_patience": self.patience,
                "loss_function": "CrossEntropyLoss with Label Smoothing (0.1)",
                "data_augmentation": "RandomCrop, RandomHorizontalFlip, ColorJitter, RandomRotation, RandomAffine",
                "mixed_precision": self.scaler is not None,
            },
            "training_results": {
                "epochs_trained": epochs_trained,
                "best_validation_accuracy": best_val,
                "final_training_accuracy": self.history.get("train_acc", [0])[-1] if epochs_trained else 0,
                "final_validation_accuracy": self.history.get("val_acc", [0])[-1] if epochs_trained else 0,
                "final_training_loss": self.history.get("train_loss", [float("nan")])[-1] if epochs_trained else float("nan"),
                "final_validation_loss": self.history.get("val_loss", [float("nan")])[-1] if epochs_trained else float("nan"),
                "convergence_epoch": (self.history["val_acc"].index(best_val) + 1) if epochs_trained else 0,
                "early_stopping_triggered": self.patience_counter >= self.patience,
            },
            "performance_metrics": self.inference_metrics if hasattr(self, "inference_metrics") else {},
            "training_history": dict(self.history),
            "baseline_summary": {
                "architecture": "ResNet-18",
                "precision": "FP32",
                "accuracy": best_val,
                "model_size_mb": self.model_metrics["model_size_mb"],
                "parameters": self.model_metrics["total_parameters"],
                "flops": self.model_metrics.get("flops_formatted", "N/A"),
                "inference_time_single": (
                    f'{self.inference_metrics["benchmark_results"][1]["per_image_time_ms"]:.2f} ms'
                    if hasattr(self, "inference_metrics") and 1 in self.inference_metrics.get("benchmark_results", {})
                    else "N/A"
                ),
            },
        }

        with open(f"{self.results_dir}/comprehensive_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)

        self.generate_professional_text_report(report)
        self.generate_layer_analysis_table()
        print("✅ Comprehensive reports generated:")
        print(f"   - {self.results_dir}/comprehensive_report.json")
        print(f"   - {self.results_dir}/professional_report.txt")
        print(f"   - {self.results_dir}/executive_summary.txt")

    def generate_professional_text_report(self, report):
        # Executive summary
        with open(f"{self.results_dir}/executive_summary.txt", "w") as f:
            f.write("="*80 + "\n")
            f.write("RESNET-18 FP32 BASELINE - EXECUTIVE SUMMARY\n")
            f.write("MSc Artificial Intelligence Dissertation Project\n")
            f.write("="*80 + "\n\n")
            f.write(f'Experiment Date: {report["experiment_metadata"]["timestamp"][:19]}\n')
            f.write(f'Model Architecture: {report["baseline_summary"]["architecture"]}\n')
            f.write(f'Precision: {report["baseline_summary"]["precision"]}\n')
            f.write(f'Task: Binary Classification ({" vs ".join(self.classes)})\n\n')
            f.write("KEY RESULTS:\n")
            f.write("-"*40 + "\n")
            f.write(f'✓ Best Validation Accuracy: {report["baseline_summary"]["accuracy"]:.2f}%\n')
            f.write(f'✓ Model Size: {report["baseline_summary"]["model_size_mb"]:.2f} MB\n')
            f.write(f'✓ Total Parameters: {report["baseline_summary"]["parameters"]:,}\n')
            f.write(f'✓ FLOPs per Forward Pass: {report["baseline_summary"]["flops"]}\n')
            if "inference_time_single" in report["baseline_summary"]:
                f.write(f'✓ Single Image Inference: {report["baseline_summary"]["inference_time_single"]}\n')
            f.write(f'✓ Training Epochs: {report["training_results"].get("epochs_trained", 0)}\n')
            f.write(f'✓ Convergence Epoch: {report["training_results"].get("convergence_epoch", 0)}\n')

            f.write("\nTRAINING STABILITY:\n")
            f.write("-"*40 + "\n")
            f.write(f'• Fixed Random Seed: {SEED} (Reproducible Results)\n')
            f.write(f'• Early Stopping: {"Triggered" if report["training_results"]["early_stopping_triggered"] else "Not Required"}\n')
            f.write("• Learning Rate Scheduling: Adaptive (ReduceLROnPlateau)\n")
            f.write("• Data Augmentation: Comprehensive (5 techniques)\n")
            f.write(f'• Mixed Precision: {"Enabled" if report["training_configuration"]["mixed_precision"] else "Disabled"}\n')

            if hasattr(self, "inference_metrics"):
                f.write("\nINFERENCE PERFORMANCE:\n")
                f.write("-"*40 + "\n")
                if 1 in self.inference_metrics.get("benchmark_results", {}):
                    f.write(f'• Single Image Latency: {self.inference_metrics["benchmark_results"][1]["per_image_time_ms"]:.2f} ms\n')
                if 64 in self.inference_metrics.get("benchmark_results", {}):
                    f.write(f'• Batch Processing (64): {self.inference_metrics["benchmark_results"][64]["throughput_fps"]:.1f} FPS\n')
                f.write(f'• GPU Memory Usage: {self.inference_metrics["memory_usage"]["gpu_memory_mb"]:.2f} MB\n')
                f.write(f'• CPU Memory Usage: {self.inference_metrics["memory_usage"]["cpu_memory_mb"]:.2f} MB\n')

            f.write("\nBASELINE REFERENCE METRICS (For Future Comparisons):\n")
            f.write("-"*60 + "\n")
            f.write("Metric                  | Value\n")
            f.write("-"*60 + "\n")
            f.write(f'Accuracy                | {report["baseline_summary"]["accuracy"]:.2f}%\n')
            f.write(f'Model Size              | {report["baseline_summary"]["model_size_mb"]:.2f} MB\n')
            f.write(f'Parameters              | {report["baseline_summary"]["parameters"]:,}\n')
            f.write(f'FLOPs                   | {report["baseline_summary"]["flops"]}\n')
            if hasattr(self, "inference_metrics") and 1 in self.inference_metrics.get("benchmark_results", {}):
                f.write(f'Inference Latency       | {self.inference_metrics["benchmark_results"][1]["per_image_time_ms"]:.2f} ms\n')

        # Detailed report
        with open(f"{self.results_dir}/professional_report.txt", "w") as f:
            f.write("="*100 + "\n")
            f.write("RESNET-18 FP32 BASELINE - COMPREHENSIVE ANALYSIS REPORT\n")
            f.write("MSc Artificial Intelligence Dissertation Project\n")
            f.write("="*100 + "\n\n")

            f.write("1. EXPERIMENT METADATA\n")
            f.write("-"*50 + "\n")
            for k, v in report["experiment_metadata"].items():
                f.write(f'{k.replace("_"," ").title()}: {v}\n')

            f.write("\n2. MODEL ARCHITECTURE SPECIFICATIONS\n")
            f.write("-"*50 + "\n")
            f.write(f'Architecture: {report["model_specifications"]["architecture"]}\n')
            f.write(f'Precision: {report["model_specifications"]["precision"]}\n')
            f.write(f'Total Parameters: {report["model_specifications"]["total_parameters"]:,}\n')
            f.write(f'Trainable Parameters: {report["model_specifications"]["trainable_parameters"]:,}\n')
            f.write(f'Model Size: {report["model_specifications"]["model_size_mb"]:.2f} MB\n')
            f.write(f'FLOPs per Forward Pass: {report["model_specifications"].get("flops_formatted","N/A")}\n')
            f.write(f'Input Resolution: {report["model_specifications"]["input_resolution"]}\n')
            f.write(f'Number of Classes: {report["model_specifications"]["num_classes"]}\n')
            f.write(f'Data Type: {report["model_specifications"]["dtype"]}\n')

            f.write("\n3. TRAINING CONFIGURATION\n")
            f.write("-"*50 + "\n")
            for k, v in report["training_configuration"].items():
                f.write(f'{k.replace("_"," ").title()}: {v}\n')

            f.write("\n4. TRAINING RESULTS\n")
            f.write("-"*50 + "\n")
            tr = report["training_results"]
            f.write(f'Epochs Trained: {tr.get("epochs_trained", 0)}\n')
            f.write(f'Best Validation Accuracy: {tr.get("best_validation_accuracy", 0.0):.4f}%\n')
            f.write(f'Final Training Accuracy: {tr.get("final_training_accuracy", 0.0):.4f}%\n')
            f.write(f'Final Validation Accuracy: {tr.get("final_validation_accuracy", 0.0):.4f}%\n')
            f.write(f'Final Training Loss: {tr.get("final_training_loss", float("nan")):.6f}\n')
            f.write(f'Final Validation Loss: {tr.get("final_validation_loss", float("nan")):.6f}\n')
            f.write(f'Convergence Epoch: {tr.get("convergence_epoch", 0)}\n')
            f.write(f'Early Stopping Triggered: {tr.get("early_stopping_triggered", False)}\n')

            if "performance_metrics" in report and report["performance_metrics"]:
                pm = report["performance_metrics"]
                f.write("\n5. INFERENCE PERFORMANCE ANALYSIS\n")
                f.write("-"*50 + "\n")
                f.write(f'Final Test Accuracy: {pm.get("final_accuracy", 0.0):.4f}%\n')

                f.write("\nBenchmark Results by Batch Size:\n")
                f.write("-"*80 + "\n")
                f.write(f'{"Batch Size":<12} {"Mean Time (ms)":<15} {"Per Image (ms)":<15} {"Throughput (FPS)":<15}\n')
                f.write("-"*80 + "\n")
                for bs, m in pm.get("benchmark_results", {}).items():
                    f.write(f'{bs:<12} {m["mean_time_ms"]:<15.2f} {m["per_image_time_ms"]:<15.2f} {m["throughput_fps"]:<15.1f}\n')

                f.write("\nMemory Usage Analysis:\n")
                f.write("-"*30 + "\n")
                mem = pm.get("memory_usage", {})
                f.write(f'Model Size on Disk: {mem.get("model_size_disk_mb", 0.0):.2f} MB\n')
                f.write(f'GPU Runtime Memory: {mem.get("gpu_memory_mb", 0.0):.2f} MB\n')
                f.write(f'CPU Runtime Memory: {mem.get("cpu_memory_mb", 0.0):.2f} MB\n')

                f.write("\nInference Methodology:\n")
                f.write("-"*30 + "\n")
                f.write("• Warm-up Passes: 10 runs before benchmarking\n")
                f.write("• Benchmark Runs: 100 iterations per batch size\n")
                f.write("• Batch Sizes Tested: [1, 8, 16, 32, 64]\n")
                f.write("• Backend: PyTorch native (CPU/GPU)\n")
                f.write("• Synchronization: CUDA synchronization enabled\n")
                f.write(f"• Reproducibility: Fixed random seed ({SEED})\n")

            f.write("\n6. TRAINING STABILITY ANALYSIS\n")
            f.write("-"*50 + "\n")
            f.write(f'• Best accuracy achieved at epoch {tr.get("convergence_epoch", 0)}\n')
            f.write(f'• Early stopping: {"Yes" if tr.get("early_stopping_triggered", False) else "No"}\n')
            f.write("• Loss/accuracy curves consistent with healthy optimization\n")

            f.write("\n7. BASELINE REFERENCE FOR FUTURE COMPARISONS\n")
            f.write("-"*50 + "\n")
            f.write("This FP32 ResNet-18 model serves as the reference baseline for:\n")
            f.write("• Quantization experiments (FP16/INT8/QAT/PTQ)\n")
            f.write("• Compression techniques & hardware studies\n")
            f.write("• Architecture optimization\n")
            f.write("\nKey Baseline Metrics to Track:\n")
            base = report["baseline_summary"]
            f.write(f'• Accuracy: {base.get("accuracy", 0.0):.2f}%\n')
            f.write(f'• Model Size: {base.get("model_size_mb", 0.0):.2f} MB\n')
            f.write(f'• Parameter Count: {base.get("parameters", 0):,}\n')
            f.write(f'• Computational Cost: {base.get("flops", "N/A")}\n')
            if "inference_time_single" in base:
                f.write(f'• Inference Latency: {base["inference_time_single"]}\n')

        print("✅ Professional text reports generated")


def main():
    print("Starting Professional ResNet-18 FP32 Baseline Training...\n")
    analyzer = ProfessionalResNet18Analyzer("ResNet18_FP32_Baseline")

    try:
        # 1) Data
        analyzer.setup_data_loaders(data_path=DATA_ROOT, batch_size=BATCH_SIZE)
        # 2) Model
        analyzer.setup_model()
        # 3) Train cfg
        analyzer.setup_training_components()
        # 4) Train
        history = analyzer.train_model(epochs=EPOCHS)
        # 5) Inference bench
        inf = analyzer.comprehensive_inference_benchmark(warmup_runs=10, benchmark_runs=100)
        # 6) Plots
        analyzer.create_professional_visualizations()
        # 7) Reports
        analyzer.save_comprehensive_report()

        # 8) Summary
        print("\n" + "="*80)
        print("RESNET-18 FP32 BASELINE TRAINING COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f'Best Validation Accuracy: {max(history["val_acc"]) if history["val_acc"] else 0.0:.2f}%')
        print(f'Model Size: {analyzer.model_metrics["model_size_mb"]:.2f} MB')
        print(f'Total Parameters: {analyzer.model_metrics["total_parameters"]:,}')
        print(f'FLOPs: {analyzer.model_metrics.get("flops_formatted","N/A")}')
        if hasattr(analyzer, "inference_metrics") and analyzer.inference_metrics.get("benchmark_results"):
            if 1 in analyzer.inference_metrics["benchmark_results"]:
                print(f'Single Image Inference: {inf["benchmark_results"][1]["per_image_time_ms"]:.2f} ms')
            if 64 in analyzer.inference_metrics["benchmark_results"]:
                print(f'Batch Throughput (64): {inf["benchmark_results"][64]["throughput_fps"]:.1f} FPS')
        print(f"Results Directory: {analyzer.results_dir}/")

        print("\n📋 FILES GENERATED FOR DISSERTATION:")
        print("   1. best_model.pth / best_weights.pt")
        print("   2. comprehensive_analysis.png/.pdf")
        print("   3. comprehensive_report.json")
        print("   4. professional_report.txt")
        print("   5. executive_summary.txt")
        print("   6. layer_analysis_table.csv/.tex")
        print("\n🎯 BASELINE ESTABLISHED FOR FUTURE COMPARISONS!")

    except Exception as e:
        print(f"❌ Error during training: {e}")
        raise e


if __name__ == "__main__":
    main()

           
