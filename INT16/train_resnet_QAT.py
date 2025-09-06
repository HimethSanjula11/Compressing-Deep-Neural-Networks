import os
import sys
import time
import copy
import warnings

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.ao.quantization import (
    QConfig, prepare_qat, convert, fuse_modules,
    MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver
)
from torch.ao.nn.quantized import FloatFunctional  # for skip connections

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import matplotlib.pyplot as plt

# -------------------------
# Clean output / backends
# -------------------------
warnings.filterwarnings("ignore", category=UserWarning)
print(f"PyTorch: {torch.__version__}")

# Optional: improve runtime stability on some CPUs
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.mkldnn.enabled = False  # helps avoid rare INT8 crashes on some setups

# -------------------------
# Device & Output
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
out_dir = "./"
os.makedirs(out_dir, exist_ok=True)

# -------------------------
# Data
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

print("Loading datasets...")
train_dataset = datasets.ImageFolder(root='dataset/training_set', transform=transform)
val_dataset   = datasets.ImageFolder(root='dataset/test_set', transform=transform)
train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=4, pin_memory=True)
val_loader    = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
print(f"Training samples:   {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# -------------------------
# Helpers
# -------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def qat_int8_qconfig_base():
    """
    Base INT8 QAT config:
      - Activations: quint8, per-tensor affine
      - Weights: qint8, per-channel symmetric (we'll override Linear to per-tensor)
    """
    return QConfig(
        activation=MovingAverageMinMaxObserver.with_args(
            dtype=torch.quint8, qscheme=torch.per_tensor_affine
        ),
        weight=MovingAveragePerChannelMinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_channel_symmetric, ch_axis=0
        ),
    )

def force_linear_per_tensor_qconfig(model: nn.Module):
    """
    Stability fix: force *Linear* layers to use per-tensor symmetric weight quantization.
    Convs remain per-channel (accuracy).
    """
    act_obs = MovingAverageMinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
    w_pt    = MovingAverageMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
    lin_qcfg = QConfig(activation=act_obs, weight=w_pt)

    lin_count = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.qconfig = lin_qcfg
            lin_count += 1
    print(f"Applied per-tensor weight qconfig to {lin_count} Linear layer(s).")

def is_int8_quantized_model(model: nn.Module) -> bool:
    """
    Robustly detect a converted INT8 eager-quantized model.
    """
    # explicit flag set right after convert()
    if getattr(model, "is_quantized_model", False):
        return True

    # presence of quantized module types
    try:
        import torch.nn.quantized as nnq
        q_types = (nnq.Conv2d, nnq.Linear, nnq.ReLU)
        for m in model.modules():
            if isinstance(m, q_types):
                return True
    except Exception:
        pass

    # packed params are another strong hint
    for m in model.modules():
        if hasattr(m, "_packed_params"):
            return True

    return False

# -------------------------
# QAT-ready ResNet18
# -------------------------
class QATResNet18(nn.Module):
    """
    Wrap torchvision ResNet18 for eager QAT:
     - QuantStub/DeQuantStub
     - Replace skip-add with FloatFunctional inside custom BasicBlock
     - Provide a fuse_model() that runs in eval() then restores mode
    """
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        try:
            base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        except Exception:
            base = models.resnet18(weights=None if not pretrained else None)

        # Replace the FC head
        base.fc = nn.Linear(base.fc.in_features, num_classes)

        # Extract stem
        self.conv1 = base.conv1
        self.bn1   = base.bn1
        self.relu  = nn.ReLU(inplace=False)  # non-inplace for quant
        self.maxpool = base.maxpool

        # Wrap residual layers with quant-friendly BasicBlock
        self.layer1 = self._wrap_layer(base.layer1)
        self.layer2 = self._wrap_layer(base.layer2)
        self.layer3 = self._wrap_layer(base.layer3)
        self.layer4 = self._wrap_layer(base.layer4)

        self.avgpool = base.avgpool
        self.fc      = base.fc

        # QAT stubs
        from torch.ao.quantization import QuantStub, DeQuantStub
        self.quant   = QuantStub()
        self.dequant = DeQuantStub()

    def _wrap_layer(self, layer):
        wrapped = nn.Sequential()
        for i, block in enumerate(layer):
            wrapped.add_module(str(i), self._wrap_basic_block(block))
        return wrapped

    def _wrap_basic_block(self, block):
        """
        Copy modules from torchvision BasicBlock and route skip add through FloatFunctional.
        """
        class QBasicBlock(nn.Module):
            def __init__(self, b):
                super().__init__()
                self.conv1 = b.conv1
                self.bn1   = b.bn1
                self.relu1 = nn.ReLU(inplace=False)
                self.conv2 = b.conv2
                self.bn2   = b.bn2
                self.downsample = b.downsample  # may be None
                self.relu2 = nn.ReLU(inplace=False)
                self.add   = FloatFunctional()  # quant-friendly add

            def forward(self, x):
                identity = x
                out = self.conv1(x); out = self.bn1(out); out = self.relu1(out)
                out = self.conv2(out); out = self.bn2(out)
                if self.downsample is not None:
                    identity = self.downsample(x)
                out = self.add.add(out, identity)
                out = self.relu2(out)
                return out

        return QBasicBlock(block)

    def fuse_model(self):
        """
        Fuse Conv-BN-ReLU patterns. Fusion requires eval() mode.
        We'll switch to eval(), fuse, then restore previous mode.
        """
        is_training = self.training
        self.eval()

        # Stem
        try:
            fuse_modules(self, [['conv1', 'bn1', 'relu']], inplace=True)
            print("Fused stem: conv1+bn1+relu")
        except Exception as e:
            print(f"[Fuse warn] stem: {e}")

        # Residual blocks
        def fuse_block(block, name):
            try:
                fuse_modules(block, [['conv1', 'bn1', 'relu1']], inplace=True)
            except Exception as e:
                print(f"[Fuse warn] {name}: conv1+bn1+relu1 -> {e}")
            try:
                fuse_modules(block, [['conv2', 'bn2']], inplace=True)
            except Exception as e:
                print(f"[Fuse warn] {name}: conv2+bn2 -> {e}")
            if block.downsample is not None and isinstance(block.downsample, nn.Sequential):
                try:
                    fuse_modules(block.downsample, [['0', '1']], inplace=True)
                except Exception as e:
                    print(f"[Fuse warn] {name}: downsample -> {e}")

        for lname in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(self, lname)
            for i, block in enumerate(layer):
                fuse_block(block, f"{lname}[{i}]")

        if is_training:
            self.train()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x); x = torch.flatten(x, 1); x = self.fc(x)
        x = self.dequant(x)
        return x

# -------------------------
# Train / Validate
# -------------------------
def validate(model, loader, criterion):
    model.eval()
    total, correct, running_loss = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out  = model(x)
            loss = criterion(out, y)
            running_loss += loss.item() * x.size(0)
            pred = out.argmax(1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    return running_loss / len(loader.dataset), 100.0 * correct / total

def train(model, train_loader, val_loader, epochs=10, lr=1e-4, weight_decay=1e-4):
    model = model.to(device)
    crit  = nn.CrossEntropyLoss()
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)

    hist = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc, best_state = 0.0, None

    for ep in range(epochs):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0

        for bi, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out  = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()

            tr_loss += loss.item() * x.size(0)
            pred = out.argmax(1)
            tr_total += y.size(0)
            tr_correct += (pred == y).sum().item()

            if bi % 50 == 0:
                print(f"  Epoch {ep+1}/{epochs} Batch {bi}/{len(train_loader)} Loss {loss.item():.4f}")

        train_loss = tr_loss / len(train_loader.dataset)
        train_acc  = 100.0 * tr_correct / tr_total

        val_loss, val_acc = validate(model, val_loader, crit)
        hist['train_loss'].append(train_loss)
        hist['train_acc'].append(train_acc)
        hist['val_loss'].append(val_loss)
        hist['val_acc'].append(val_acc)

        if val_acc > best_acc:
            best_acc, best_state = val_acc, copy.deepcopy(model.state_dict())

        sched.step(val_loss)
        print(f"Epoch {ep+1}/{epochs}: "
              f"Train Acc {train_acc:.2f}% Loss {train_loss:.4f} | "
              f"Val Acc {val_acc:.2f}% Loss {val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Loaded best model (Val Acc {best_acc:.2f}%)")
    return model, hist, best_acc

# -------------------------
# Inference timing (FIXED)
# -------------------------
def test_inference(model, loader, name="Model"):
    """
    Run inference; INT8 (qnnpack) must run on CPU.
    Float/QAT can run on CUDA if available.
    """
    model.eval()

    q_model = is_int8_quantized_model(model)
    if q_model:
        # INT8 path (QNNPACK): CPU only
        run_device = torch.device("cpu")
        torch.backends.quantized.engine = "qnnpack"
        model = model.to(run_device)
        use_channels_last = True
    else:
        # Float/QAT path
        run_device = device
        model = model.to(run_device)
        use_channels_last = False

    total, correct = 0, 0
    start = time.time()

    with torch.no_grad():
        for x, y in loader:
            if q_model:
                # ensure CPU tensors; qnnpack prefers channels_last
                x = x.to(run_device).contiguous(
                    memory_format=torch.channels_last if use_channels_last else torch.contiguous_format
                )
                y = y.to(run_device)
            else:
                x, y = x.to(run_device), y.to(run_device)

            out = model(x)
            pred = out.argmax(1)
            total += y.size(0)
            correct += (pred == y).sum().item()

    elapsed = time.time() - start
    acc = 100.0 * correct / total
    avg = elapsed / max(1, total)
    print(f"[{name}] Acc {acc:.2f}% | Total {elapsed:.3f}s | {avg*1000:.2f} ms/img on {run_device} (engine={torch.backends.quantized.engine})")
    return acc, elapsed, avg

# -------------------------
# Safe convert to INT8 (qnnpack) — FIX: mark model as quantized
# -------------------------
def safe_convert_qat_int8(model_qat: nn.Module) -> nn.Module:
    """
    Convert a QAT model to INT8 on CPU using qnnpack engine. This avoids the
    per-channel Linear crash seen with fbgemm in eager QAT paths.
    """
    print("Converting QAT model to INT8 on CPU with qnnpack...")
    # Switch engine to qnnpack *before* conversion
    try:
        torch.backends.quantized.engine = 'qnnpack'
        print("  quantized.engine = 'qnnpack'")
    except Exception as e:
        print(f"  Could not set qnnpack engine ({e}), falling back to fbgemm.")
        torch.backends.quantized.engine = 'fbgemm'

    qat_cpu = copy.deepcopy(model_qat).to('cpu').eval()
    int8_model = convert(qat_cpu, inplace=False)

    # Mark the model so our runtime detection is reliable
    setattr(int8_model, "is_quantized_model", True)

    print("✓ Conversion successful.")
    return int8_model

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    print("="*60)
    print("ResNet18 INT8 QAT (stable: per-tensor Linear + qnnpack)")
    print("="*60)

    # 1) Build model
    print("\nStep 1: Building QAT-ready ResNet18...")
    model = QATResNet18(num_classes=2, pretrained=True)
    print(f"Trainable parameters: {count_parameters(model):,}")

    # 2) Fuse (must be eval), then back to train
    print("\nStep 2: Fusing modules (eval-only) ...")
    model.fuse_model()
    print("Fusion complete.")

    # 3) Prepare for QAT (INT8) with stability fix
    print("\nStep 3: Preparing QAT (INT8)...")
    model.qconfig = qat_int8_qconfig_base()     # convs per-channel (default here)
    force_linear_per_tensor_qconfig(model)      # linears per-tensor (stability)
    prepare_qat(model, inplace=True)
    print("QAT graph prepared.")

    # 4) Train (QAT)
    print("\nStep 4: QAT training...")
    model, hist, best_val_acc = train(model, train_loader, val_loader, epochs=10, lr=1e-4)

    # Save FP32+fake-quant weights (QAT model state)
    qat_path = os.path.join(out_dir, "resnet18_qat_int8_state_stable.pth")
    torch.save(model.state_dict(), qat_path)
    print(f"Saved QAT model state: {qat_path}")

    # 5) Convert to INT8 on CPU (qnnpack)
    print("\nStep 5: Converting to INT8...")
    int8_model = safe_convert_qat_int8(model)
    int8_path = os.path.join(out_dir, "resnet18_int8_quantized_stable.pth")
    try:
        torch.save(int8_model.state_dict(), int8_path)
        print(f"Saved INT8 model state: {int8_path}")
    except Exception as e:
        print(f"Warning: could not save INT8 state dict: {e}")

    # 6) Evaluate (QAT can run on CUDA; INT8 must run on CPU)
    print("\nStep 6: Evaluating ...")
    test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    acc_qat,  t_qat,  avg_qat  = test_inference(model,      test_loader, "QAT (FP32 fake-quant)")
    # Be explicit and safe for INT8
    torch.backends.quantized.engine = 'qnnpack'
    int8_model = int8_model.to('cpu')
    acc_int8, t_int8, avg_int8 = test_inference(int8_model, test_loader, "INT8 Quantized (qnnpack)")

    # 7) Plots
    print("\nStep 7: Saving training curves...")
    epochs = range(1, len(hist['val_acc'])+1)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, hist['train_acc'], label='Train Acc')
    plt.plot(epochs, hist['val_acc'],   label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Acc (%)'); plt.title('Accuracy'); plt.grid(alpha=0.3); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(epochs, hist['train_loss'], label='Train Loss')
    plt.plot(epochs, hist['val_loss'],   label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss'); plt.grid(alpha=0.3); plt.legend()
    fig_path = os.path.join(out_dir, "resnet18_qat_training_curves_stable.png")
    plt.tight_layout(); plt.savefig(fig_path, dpi=300, bbox_inches='tight'); plt.close()
    print(f"Saved: {fig_path}")

    # 8) Summary
    print("\n" + "="*60)
    print("INT8 QAT SUMMARY (stable settings)")
    print("="*60)
    print(f"Best Val Acc during QAT: {best_val_acc:.2f}%")
    print(f"QAT Inference:  {avg_qat*1000:.2f} ms/img")
    print(f"INT8 Inference: {avg_int8*1000:.2f} ms/img (CPU, qnnpack)")
    if avg_int8 > 0:
        print(f"Speedup (QAT->INT8): {avg_qat/avg_int8:.2f}x")
    print(f"Acc (QAT):  {acc_qat:.2f}% | Acc (INT8): {acc_int8:.2f}%")
    print(f"QAT state: {qat_path}")
    print(f"INT8 state: {int8_path}")
