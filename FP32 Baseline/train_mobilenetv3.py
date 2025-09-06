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
WEIGHT_DECAY = 5e-4             # Weight decay for MobileNet
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
print("PROFESSIONAL MOBILENETV3-SMALL FP32 BASELINE TRAINING")
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


class ProfessionalMobileNetV3Analyzer:
    """Professional MobileNetV3-Small training & analysis class (FP32 baseline, stabilized)"""

    def __init__(self, experiment_name="MobileNetV3Small_FP32_Baseline"):
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

        # Stable baseline augmentation (matching AlexNet approach)
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
            "architecture": "MobileNetV3-Small",
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
        print("  Architecture: MobileNetV3-Small")
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

        # Load pretrained MobileNetV3-Small
        self.model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        # Replace final classifier layer (classifier[3] in MobileNetV3)
        in_features = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(in_features, self.num_classes)
        nn.init.xavier_uniform_(self.model.classifier[3].weight)
        nn.init.constant_(self.model.classifier[3].bias, 0)

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
            if name.startswith("classifier.3"):
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
            if not name.startswith("classifier.3"):
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
        
        # Check if files exist and load the best model
        if os.path.exists(weights_path):
            print(f"Loading best weights from: {weights_path}")
            state = torch.load(weights_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
        elif os.path.exists(ckpt_path):
            print(f"Loading best checkpoint from: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
        else:
            print(f"⚠️ No saved model found. Using current model state.")
            print(f"Expected paths:\n  - {weights_path}\n  - {ckpt_path}")
        
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
    # Visuals & reports (EXACT MATCH TO ALEXNET)
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
        plt.title("Mean Batch Time per Epoch", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch"); plt.ylabel("Seconds / batch"); plt.legend(); plt.grid(True, alpha=0.3)

        # 5. Top parameter-heavy layers (barh)
        ax5 = plt.subplot(3, 3, 5)
        try:
            lb = self.model_metrics.get("layer_breakdown", [])
            if lb:
                # sort by Parameters desc and take top 10
                top = sorted(lb, key=lambda x: x["Parameters"], reverse=True)[:10]
                names = [f'{x["Layer Name"][-30:]}' for x in top]
                params = [x["Parameters"] for x in top]
                plt.barh(range(len(top)), params)
                plt.yticks(range(len(top)), names)
                plt.title("Top Layers by Parameter Count", fontsize=12, fontweight="bold")
                plt.xlabel("Parameters"); plt.gca().invert_yaxis(); plt.grid(True, axis='x', alpha=0.2)
            else:
                plt.text(0.5, 0.5, "No layer breakdown available", ha='center', va='center', transform=ax5.transAxes)
                plt.axis('off')
        except Exception as e:
            plt.text(0.5, 0.5, f"Layer chart error: {e}", ha='center', va='center', transform=ax5.transAxes)
            plt.axis('off')

        # 6. FLOPs / Model size summary box
        ax6 = plt.subplot(3, 3, 6)
        ax6.axis('off')
        flops = self.model_metrics.get("flops_formatted", "N/A")
        mbytes = self.model_metrics.get("model_size_mb", 0.0)
        tr_params = self.model_metrics.get("trainable_parameters", 0)
        txt = (
            f"Architecture: {self.model_metrics.get('architecture', 'N/A')}\n"
            f"Precision: {self.model_metrics.get('precision', 'N/A')}\n"
            f"Input: {self.model_metrics.get('input_resolution', ('N/A','N/A'))}\n"
            f"Trainable Params: {tr_params:,}\n"
            f"Model Size: {mbytes:.2f} MB\n"
            f"FLOPs/Forward: {flops}"
        )
        ax6.text(0.02, 0.98, txt, va='top', ha='left', fontsize=11, bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="0.8"))

        # 7. Throughput vs batch size (from inference benchmark)
        ax7 = plt.subplot(3, 3, 7)
        if self.inference_metrics and "benchmark_results" in self.inference_metrics:
            br = self.inference_metrics["benchmark_results"]
            bsz = sorted(br.keys())
            fps = [br[b]["throughput_fps"] for b in bsz]
            plt.plot(bsz, fps, marker='o', linewidth=2)
            plt.title("Throughput vs Batch Size", fontsize=12, fontweight="bold")
            plt.xlabel("Batch size"); plt.ylabel("FPS"); plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, "Run benchmark to populate throughput", ha='center', va='center', transform=ax7.transAxes)
            plt.axis('off')

        # 8. Per-image latency vs batch size
        ax8 = plt.subplot(3, 3, 8)
        if self.inference_metrics and "benchmark_results" in self.inference_metrics:
            br = self.inference_metrics["benchmark_results"]
            bsz = sorted(br.keys())
            pit = [br[b]["per_image_time_ms"] for b in bsz]
            plt.plot(bsz, pit, marker='s', linewidth=2)
            plt.title("Per-Image Latency vs Batch Size", fontsize=12, fontweight="bold")
            plt.xlabel("Batch size"); plt.ylabel("ms / image"); plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, "Run benchmark to populate latency", ha='center', va='center', transform=ax8.transAxes)
            plt.axis('off')

        # 9. Confusion matrix (optional)
        ax9 = plt.subplot(3, 3, 9)
        try:
            from sklearn.metrics import confusion_matrix
            self.model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs = inputs.to(self.device)
                    outputs = self.model(inputs)
                    _, preds = outputs.max(1)
                    all_preds.extend(preds.cpu().numpy().tolist())
                    all_labels.extend(labels.numpy().tolist())
            cm = confusion_matrix(all_labels, all_preds, labels=list(range(self.num_classes)))
            sns.heatmap(cm, annot=True, fmt="d", cbar=False, ax=ax9,
                        xticklabels=self.classes, yticklabels=self.classes)
            ax9.set_title("Validation Confusion Matrix", fontsize=12, fontweight="bold")
            ax9.set_xlabel("Predicted"); ax9.set_ylabel("Actual")
        except Exception as e:
            plt.text(0.5, 0.5, f"Confusion matrix unavailable:\n{e}", ha='center', va='center', transform=ax9.transAxes)
            plt.axis('off')

        plt.tight_layout()
        fig_path = os.path.join(self.results_dir, "training_diagnostics.png")
        plt.savefig(fig_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved diagnostics figure → {fig_path}")

        # Persist metrics
        try:
            # history CSV
            hist_df = pd.DataFrame(self.history)
            hist_csv = os.path.join(self.results_dir, "history.csv")
            hist_df.to_csv(hist_csv, index=False)
            # JSON metrics
            with open(os.path.join(self.results_dir, "model_metrics.json"), "w") as f:
                json.dump(self.model_metrics, f, indent=2)
            if self.inference_metrics:
                with open(os.path.join(self.results_dir, "inference_metrics.json"), "w") as f:
                    json.dump(self.inference_metrics, f, indent=2)
            print("Saved history.csv, model_metrics.json, inference_metrics.json")
        except Exception as e:
            print(f"⚠️ Failed to save metrics: {e}")

    # ---------------------------
    # Convenience: export a concise summary table for the report
    # ---------------------------
    def export_report_table(self):
        try:
            # Summarize best metrics
            best_epoch = int(np.argmax(self.history["val_acc"])) + 1 if self.history.get("val_acc") else -1
            summary = {
                "Architecture": self.model_metrics.get("architecture", "MobileNetV3-Small"),
                "Precision": self.model_metrics.get("precision", "FP32"),
                "Best Epoch": best_epoch,
                "Best Val Acc (%)": round(self.best_val_acc, 2),
                "Params": f'{self.model_metrics.get("trainable_parameters", 0):,}',
                "Model Size (MB)": round(self.model_metrics.get("model_size_mb", 0.0), 2),
                "FLOPs/Forward": self.model_metrics.get("flops_formatted", "N/A"),
                "Final Test Acc (%)": round(self.inference_metrics.get("final_accuracy", float("nan")), 2)
                if self.inference_metrics else "N/A",
            }
            df = pd.DataFrame([summary])
            md_path = os.path.join(self.results_dir, "summary_table.md")
            csv_path = os.path.join(self.results_dir, "summary_table.csv")
            df.to_csv(csv_path, index=False)
            # simple markdown export
            with open(md_path, "w") as f:
                f.write(df.to_markdown(index=False))
            print(f"Saved report tables → {csv_path}, {md_path}")
        except Exception as e:
            print(f"⚠️ Could not export report table: {e}")


# ---------------------------
# Main execution
# ---------------------------
if __name__ == "__main__":
    runner = ProfessionalMobileNetV3Analyzer()

    # data
    runner.setup_data_loaders(data_path=DATA_ROOT, batch_size=BATCH_SIZE)

    # model
    runner.setup_model()

    # training config
    runner.setup_training_components()

    # train
    runner.train_model(epochs=EPOCHS)

    # benchmark (optional but recommended for your dissertation)
    runner.comprehensive_inference_benchmark(warmup_runs=10, benchmark_runs=100)

    # visuals + exports
    runner.create_professional_visualizations()
    runner.export_report_table()