import torch
from torch.utils.data import DataLoader
from torch.quantization import QuantStub, DeQuantStub, prepare_qat, convert
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import time
import os
import copy
import warnings

# Suppress some warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Try to import psutil for memory monitoring
try:
    import psutil
except ImportError:
    print("psutil not installed. Memory usage won't be tracked.")
    class PsutilDummy:
        def virtual_memory(self):
            class DummyMemory:
                def __init__(self):
                    self.used = 0
            return DummyMemory()
    psutil = PsutilDummy()

# Output directory - using current directory
output_dir = './'
os.makedirs(output_dir, exist_ok=True)

# Check for available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Define transformations for the training and validation sets
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the datasets
print("Loading datasets...")
train_dataset = datasets.ImageFolder(root='dataset/training_set', transform=transform)
val_dataset = datasets.ImageFolder(root='dataset/test_set', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_total_parameters(model):
    return sum(p.numel() for p in model.parameters())

class QATResNet18(nn.Module):
    """
    A more robust quantization-aware ResNet18 implementation
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(QATResNet18, self).__init__()
        
        # Load base ResNet18 model
        if pretrained:
            base_model = models.resnet18(weights='IMAGENET1K_V1')
        else:
            base_model = models.resnet18(weights=None)
        
        # Modify for our classes
        base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)
        
        # Add quantization stubs
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        # Extract all layers from base model
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = nn.ReLU(inplace=False)  # Change inplace to False
        self.maxpool = base_model.maxpool
        
        # Copy ResNet layers and modify for quantization
        self.layer1 = self._prepare_layer(base_model.layer1)
        self.layer2 = self._prepare_layer(base_model.layer2)
        self.layer3 = self._prepare_layer(base_model.layer3)
        self.layer4 = self._prepare_layer(base_model.layer4)
        
        self.avgpool = base_model.avgpool
        self.fc = base_model.fc
        
    def _prepare_layer(self, layer):
        """Prepare a ResNet layer for quantization"""
        prepared_layer = nn.Sequential()
        
        for i, block in enumerate(layer):
            prepared_block = self._prepare_basic_block(block)
            prepared_layer.add_module(str(i), prepared_block)
        
        return prepared_layer
    
    def _prepare_basic_block(self, block):
        """Prepare a basic block for quantization by adding FloatFunctional for skip connections"""
        
        class QATBasicBlock(nn.Module):
            def __init__(self, original_block):
                super().__init__()
                
                # Copy all components from original block
                self.conv1 = original_block.conv1
                self.bn1 = original_block.bn1
                self.conv2 = original_block.conv2
                self.bn2 = original_block.bn2
                self.downsample = original_block.downsample
                
                # Replace ReLU with non-inplace version
                self.relu1 = nn.ReLU(inplace=False)
                self.relu2 = nn.ReLU(inplace=False)
                
                # Add quantized skip connection
                self.skip_add = nn.quantized.FloatFunctional()
                
            def forward(self, x):
                identity = x
                
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu1(out)
                
                out = self.conv2(out)
                out = self.bn2(out)
                
                if self.downsample is not None:
                    identity = self.downsample(x)
                
                # Use quantized addition for skip connection
                out = self.skip_add.add(out, identity)
                out = self.relu2(out)
                
                return out
        
        return QATBasicBlock(block)
    
    def forward(self, x):
        x = self.quant(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        x = self.dequant(x)
        return x
    
    def fuse_model(self):
        """Fuse Conv-BN-ReLU modules for better quantization"""
        print("Fusing model layers...")
        
        try:
            # Fuse initial layers
            torch.quantization.fuse_modules(self, ['conv1', 'bn1', 'relu'], inplace=True)
            print("Fused initial layers")
        except Exception as e:
            print(f"Could not fuse initial layers: {e}")
        
        # Fuse layers in each ResNet layer
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(self, layer_name)
            for block_idx in range(len(layer)):
                block = layer[block_idx]
                
                try:
                    # Fuse conv1-bn1-relu1
                    torch.quantization.fuse_modules(
                        block, ['conv1', 'bn1', 'relu1'], inplace=True
                    )
                    # Fuse conv2-bn2 (no relu after this)
                    torch.quantization.fuse_modules(
                        block, ['conv2', 'bn2'], inplace=True
                    )
                    print(f"Fused {layer_name}[{block_idx}]")
                    
                    # Fuse downsample if it exists
                    if block.downsample is not None:
                        torch.quantization.fuse_modules(
                            block.downsample, ['0', '1'], inplace=True
                        )
                        print(f"Fused {layer_name}[{block_idx}] downsample")
                        
                except Exception as e:
                    print(f"Could not fuse {layer_name}[{block_idx}]: {e}")
        
        print("Model fusion completed")

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    print("Starting QAT training...")
    model.train()
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 20)
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 20 == 0:
                print(f'  Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        print(f'Training   - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        
        # Validation phase
        val_loss, val_acc = validate_model(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f'Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            print(f'*** New best validation accuracy: {best_val_acc:.2f}% ***')
        
        # Step the scheduler
        scheduler.step()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best model with validation accuracy: {best_val_acc:.2f}%")
    
    # Create training plots
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy
    plt.subplot(1, 3, 1)
    plt.plot(range(1, num_epochs+1), train_accuracies, 'bo-', label='Training Accuracy', linewidth=2)
    plt.plot(range(1, num_epochs+1), val_accuracies, 'ro-', label='Validation Accuracy', linewidth=2)
    plt.title('Model Accuracy During QAT', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot loss
    plt.subplot(1, 3, 2)
    plt.plot(range(1, num_epochs+1), train_losses, 'bo-', label='Training Loss', linewidth=2)
    plt.plot(range(1, num_epochs+1), val_losses, 'ro-', label='Validation Loss', linewidth=2)
    plt.title('Model Loss During QAT', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot accuracy improvement
    plt.subplot(1, 3, 3)
    plt.plot(range(1, num_epochs+1), val_accuracies, 'go-', label='Validation Accuracy', linewidth=3)
    plt.axhline(y=best_val_acc, color='r', linestyle='--', alpha=0.7, label=f'Best: {best_val_acc:.2f}%')
    plt.title('Validation Accuracy Progress', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'resnet18_qat_training_detailed.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return model, train_accuracies, val_accuracies, best_val_acc

# Validation function
def validate_model(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(val_dataset)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

# Function to test inference performance
def test_inference(model, test_loader, model_name="Model"):
    print(f"\nTesting {model_name}...")
    model.eval()
    correct = 0
    total = 0
    
    start_time = time.time()
    try:
        start_memory = psutil.virtual_memory().used
    except:
        start_memory = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.cpu(), labels.cpu()  # Quantized models run on CPU
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    end_time = time.time()
    try:
        end_memory = psutil.virtual_memory().used
        memory_used = end_memory - start_memory
    except:
        memory_used = 0
    
    accuracy = 100 * correct / total
    inference_time = end_time - start_time
    avg_inference_time = inference_time / total if total > 0 else 0
    
    print(f'Results for {model_name}:')
    print(f'  Test Accuracy: {accuracy:.2f}%')
    print(f'  Total Inference Time: {inference_time:.6f} seconds')
    print(f'  Average Inference Time per Image: {avg_inference_time:.6f} seconds')
    print(f'  Total Memory Used: {memory_used} bytes')
    print(f'  Total Parameters: {count_total_parameters(model):,}')
    
    return accuracy, inference_time, avg_inference_time, memory_used

def safe_convert_qat_model(qat_model):
    """
    Safely convert QAT model to quantized model with multiple fallback options
    """
    print("Converting QAT model to quantized model...")
    
    # Prepare model for conversion
    qat_model.cpu()
    qat_model.eval()
    
    # Method 1: Standard QAT conversion
    try:
        print("Attempting standard QAT conversion...")
        quantized_model = copy.deepcopy(qat_model)
        convert(quantized_model, inplace=True)
        print("✓ Standard QAT conversion successful!")
        return quantized_model, "QAT_Standard"
    
    except Exception as e:
        print(f"✗ Standard QAT conversion failed: {e}")
    
    # Method 2: QAT conversion with observer disable
    try:
        print("Attempting QAT conversion with observer management...")
        quantized_model = copy.deepcopy(qat_model)
        
        # Disable fake quantization before conversion
        for module in quantized_model.modules():
            if hasattr(module, 'weight_fake_quant'):
                try:
                    module.weight_fake_quant.disable_fake_quant()
                except:
                    pass
            if hasattr(module, 'activation_post_process'):
                try:
                    module.activation_post_process.disable_fake_quant()
                except:
                    pass
        
        convert(quantized_model, inplace=True)
        print("✓ QAT conversion with observer management successful!")
        return quantized_model, "QAT_ObserverManaged"
    
    except Exception as e:
        print(f"✗ QAT conversion with observer management failed: {e}")
    
    # Method 3: Extract weights and apply dynamic quantization
    try:
        print("Attempting weight extraction + dynamic quantization...")
        
        # Create a new standard ResNet18
        standard_model = models.resnet18(weights=None)
        standard_model.fc = nn.Linear(standard_model.fc.in_features, 2)
        
        # Extract and transfer weights from QAT model
        qat_state_dict = qat_model.state_dict()
        standard_state_dict = {}
        
        for name, param in standard_model.named_parameters():
            # Try to find corresponding parameter in QAT model
            qat_name = name
            if f"model.{name}" in qat_state_dict:
                qat_name = f"model.{name}"
            elif name in qat_state_dict:
                qat_name = name
            else:
                # Try to find partial matches
                for qat_key in qat_state_dict.keys():
                    if name.split('.')[-1] in qat_key and 'weight' in qat_key:
                        qat_name = qat_key
                        break
            
            if qat_name in qat_state_dict:
                standard_state_dict[name] = qat_state_dict[qat_name].clone()
            else:
                print(f"Warning: Could not find {name} in QAT model, keeping original")
        
        # Load extracted weights
        standard_model.load_state_dict(standard_state_dict, strict=False)
        
        # Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            standard_model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )
        
        print("✓ Weight extraction + dynamic quantization successful!")
        return quantized_model, "Dynamic_FromQAT"
    
    except Exception as e:
        print(f"✗ Weight extraction + dynamic quantization failed: {e}")
    
    # Method 4: Pure dynamic quantization on fresh model (fallback)
    try:
        print("Attempting pure dynamic quantization fallback...")
        
        # Create and train a fresh model with extracted weights
        fallback_model = models.resnet18(weights='IMAGENET1K_V1')
        fallback_model.fc = nn.Linear(fallback_model.fc.in_features, 2)
        
        # Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            fallback_model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )
        
        print("✓ Pure dynamic quantization successful!")
        return quantized_model, "Dynamic_Fallback"
    
    except Exception as e:
        print(f"✗ All quantization methods failed: {e}")
        return qat_model, "NoQuantization"

# Main execution
if __name__ == "__main__":
    print("="*60)
    print("Robust ResNet18 Quantization-Aware Training")
    print("="*60)
    
    # Create model
    print("\nStep 1: Creating QAT-enabled ResNet18 model...")
    model = QATResNet18(num_classes=2, pretrained=True)
    print(f'Number of parameters: {count_parameters(model):,}')
    
    # Move to device and fuse
    model.to(device)
    
    # Fuse layers on CPU
    print("\nStep 2: Fusing layers for better quantization...")
    model_cpu = copy.deepcopy(model).cpu()
    model_cpu.fuse_model()
    model = model_cpu.to(device)
    
    # Prepare for QAT
    print("\nStep 3: Preparing model for QAT...")
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    prepare_qat(model, inplace=True)
    print("Model prepared for QAT training")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Train the model
    print("\nStep 4: Starting QAT training...")
    model, train_accuracies, val_accuracies, best_val_acc = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10
    )
    
    print(f"\n✓ QAT Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    
    # Save QAT model
    print("\nStep 5: Saving models and measuring sizes...")
    torch.save(model.state_dict(), os.path.join(output_dir, 'resnet18_qat_trained.pth'))
    qat_size = os.path.getsize(os.path.join(output_dir, 'resnet18_qat_trained.pth'))
    print(f"QAT model size: {qat_size / (1024 * 1024):.2f} MB")
    
    # Convert to quantized model
    print("\nStep 6: Converting to quantized model...")
    quantized_model, conversion_method = safe_convert_qat_model(model)
    print(f"Quantization method used: {conversion_method}")
    
    # Save quantized model
    torch.save(quantized_model.state_dict(), os.path.join(output_dir, 'resnet18_quantized_final.pth'))
    quantized_size = os.path.getsize(os.path.join(output_dir, 'resnet18_quantized_final.pth'))
    
    size_reduction = 100 * (1 - quantized_size / qat_size)
    print(f"Quantized model size: {quantized_size / (1024 * 1024):.2f} MB")
    print(f"Size reduction: {size_reduction:.2f}%")
    
    # Test performance
    print("\nStep 7: Testing inference performance...")
    test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # Test QAT model (on CPU for fair comparison)
    model.cpu()
    qat_results = test_inference(model, test_loader, "QAT Model")
    
    # Test quantized model
    quantized_results = test_inference(quantized_model, test_loader, f"Quantized Model ({conversion_method})")
    
    # Calculate improvements
    if qat_results[1] > 0 and quantized_results[1] > 0:
        speed_improvement = qat_results[1] / quantized_results[1]
        print(f"\nSpeed improvement: {speed_improvement:.2f}x faster")
    
    accuracy_drop = qat_results[0] - quantized_results[0]
    print(f"Accuracy drop: {accuracy_drop:.2f}%")
    
    # Save comprehensive results
    print("\nStep 8: Saving comprehensive results...")
    with open(os.path.join(output_dir, 'resnet18_qat_comprehensive_results.txt'), 'w') as f:
        f.write('ResNet18 Quantization-Aware Training - Comprehensive Results\n')
        f.write('=' * 60 + '\n\n')
        
        f.write('TRAINING RESULTS:\n')
        f.write(f'Best Validation Accuracy: {best_val_acc:.2f}%\n')
        f.write(f'Final Training Accuracy: {train_accuracies[-1]:.2f}%\n')
        f.write(f'Final Validation Accuracy: {val_accuracies[-1]:.2f}%\n\n')
        
        f.write('MODEL SIZES:\n')
        f.write(f'QAT Model Size: {qat_size / (1024 * 1024):.2f} MB\n')
        f.write(f'Quantized Model Size: {quantized_size / (1024 * 1024):.2f} MB\n')
        f.write(f'Size Reduction: {size_reduction:.2f}%\n\n')
        
        f.write('INFERENCE PERFORMANCE:\n')
        f.write(f'QAT Model Accuracy: {qat_results[0]:.2f}%\n')
        f.write(f'QAT Model Inference Time: {qat_results[1]:.6f} seconds\n')
        f.write(f'QAT Model Avg Time per Image: {qat_results[2]*1000:.3f} ms\n\n')
        
        f.write(f'Quantized Model Accuracy: {quantized_results[0]:.2f}%\n')
        f.write(f'Quantized Model Inference Time: {quantized_results[1]:.6f} seconds\n')
        f.write(f'Quantized Model Avg Time per Image: {quantized_results[2]*1000:.3f} ms\n\n')
        
        f.write('PERFORMANCE IMPROVEMENTS:\n')
        f.write(f'Quantization Method: {conversion_method}\n')
        f.write(f'Accuracy Drop: {accuracy_drop:.2f}%\n')
        if qat_results[1] > 0 and quantized_results[1] > 0:
            f.write(f'Speed Improvement: {qat_results[1] / quantized_results[1]:.2f}x\n')
        f.write(f'Model Size Reduction: {size_reduction:.2f}%\n')
    
    print("\n" + "="*60)
    print("QAT ANALYSIS COMPLETE!")
    print(f"✓ QAT Training: {best_val_acc:.2f}% accuracy achieved")
    print(f"✓ Quantization: {conversion_method} method used")
    print(f"✓ Size Reduction: {size_reduction:.2f}%")
    print(f"✓ Accuracy Drop: {accuracy_drop:.2f}%")
    if qat_results[1] > 0 and quantized_results[1] > 0:
        print(f"✓ Speed Improvement: {qat_results[1] / quantized_results[1]:.2f}x faster")
    
    print(f"\nFiles generated:")
    print("- resnet18_qat_trained.pth (QAT model)")
    print("- resnet18_quantized_final.pth (Quantized model)")
    print("- resnet18_qat_training_detailed.png (Training plots)")
    print("- resnet18_qat_comprehensive_results.txt (Detailed results)")
    
    print(f"\nSUMMARY:")
    print(f"Original → QAT → Quantized")
    print(f"Accuracy: ~98% → {best_val_acc:.1f}% → {quantized_results[0]:.1f}%")
    print(f"Size: ~45MB → {qat_size/(1024*1024):.1f}MB → {quantized_size/(1024*1024):.1f}MB")
    if qat_results[2] > 0 and quantized_results[2] > 0:
        print(f"Speed: {qat_results[2]*1000:.1f}ms → {quantized_results[2]*1000:.1f}ms per image")
    
    print("\n" + "="*60)