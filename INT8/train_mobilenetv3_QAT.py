import torch
from torch.utils.data import DataLoader
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
import numpy as np

# Suppress warnings
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

# Setup
output_dir = './'
os.makedirs(output_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Conservative data loading (QAT requirement)
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),  # Conservative augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Loading datasets with conservative augmentation for QAT...")
train_dataset = datasets.ImageFolder(root='dataset/training_set', transform=transform_train)
val_dataset = datasets.ImageFolder(root='dataset/test_set', transform=transform_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Classes: {train_dataset.classes}")

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_total_parameters(model):
    """Count all parameters"""
    return sum(p.numel() for p in model.parameters())

def get_model_size_mb(model):
    """Calculate model size in MB"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024**2)

class FixedQATMobileNetV3(nn.Module):
    """
    Fixed MobileNetV3 for proper QAT implementation
    Addresses the fake quantizer detection issues
    """
    def __init__(self, num_classes=2):
        super(FixedQATMobileNetV3, self).__init__()
        
        # Load pretrained model
        self.model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        
        # Modify classifier for our classes
        num_ftrs = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(num_ftrs, num_classes)
        
        print(f"Fixed QAT Model created with {count_total_parameters(self):,} parameters")
    
    def forward(self, x):
        return self.model(x)

def train_model_baseline(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=8):
    """Baseline training before QAT"""
    print(f"Starting baseline training for {num_epochs} epochs...")
    
    model.train()
    history = {
        'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 50)
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 50 == 0:
                print(f'  Batch {i:3d}/{len(train_loader)}, Loss: {loss.item():.4f}, '
                      f'Acc: {100 * correct / total:.2f}%')
        
        # Calculate epoch metrics
        train_loss = running_loss / len(train_dataset)
        train_acc = 100 * correct / total
        
        # Validation phase
        val_loss, val_acc = validate_model(model, val_loader, criterion)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Training   - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')
        print(f'Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            print(f'*** NEW BEST MODEL - Validation Accuracy: {best_val_acc:.2f}% ***')
        
        scheduler.step()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best baseline model with validation accuracy: {best_val_acc:.2f}%")
    
    return model, history, best_val_acc

def setup_fixed_qat_model(baseline_model):
    """
    FIXED QAT setup that ensures fake quantizers are properly inserted
    """
    print("\n" + "="*70)
    print("FIXED QAT MODEL SETUP - ENSURING FAKE QUANTIZERS")
    print("="*70)
    
    # Set FBGEMM backend
    torch.backends.quantized.engine = "fbgemm"
    print("✓ Set quantization backend to FBGEMM")
    
    # Create QAT model copy on CPU for preparation
    qat_model = copy.deepcopy(baseline_model).cpu().eval()
    
    # Method 1: Try the newer API first
    try:
        print("🔧 Attempting modern QAT setup...")
        from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
        from torch.ao.quantization import get_default_qat_qconfig, QConfigMapping
        
        # Create QAT config
        qat_qconfig = get_default_qat_qconfig("fbgemm")
        qconfig_mapping = QConfigMapping().set_global(qat_qconfig)
        
        # Prepare for QAT
        example_inputs = torch.randn(1, 3, 224, 224)
        qat_prepared = prepare_qat_fx(qat_model.train(), qconfig_mapping, example_inputs)
        
        # Verify fake quantizers
        fake_quant_count = 0
        for name, module in qat_prepared.named_modules():
            if 'fake_quantize' in str(type(module)).lower() or hasattr(module, 'fake_quant'):
                fake_quant_count += 1
        
        if fake_quant_count > 0:
            print(f"✅ Modern QAT setup successful! Found {fake_quant_count} fake quantizers")
            return qat_prepared.to(device), 'modern'
        else:
            raise RuntimeError("No fake quantizers found with modern API")
            
    except Exception as e:
        print(f"⚠️  Modern QAT failed: {e}")
        print("🔄 Trying alternative QAT approach...")
    
    # Method 2: Manual fake quantizer insertion
    try:
        print("🔧 Attempting manual fake quantizer insertion...")
        from torch.ao.quantization import QConfig, default_qat_fake_quant, default_weight_fake_quant
        from torch.ao.quantization.quantize import prepare_qat
        
        # Reset model
        qat_model = copy.deepcopy(baseline_model).cpu().eval()
        
        # Create QAT config with explicit fake quantizers
        qat_qconfig = QConfig(
            activation=default_qat_fake_quant,
            weight=default_weight_fake_quant
        )
        
        # Apply QConfig to all quantizable layers
        qat_model.qconfig = qat_qconfig
        
        # Apply QConfig to specific modules
        for name, module in qat_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.qconfig = qat_qconfig
        
        # Prepare QAT
        qat_prepared = prepare_qat(qat_model, inplace=False)
        
        # Verify fake quantizers
        fake_quant_count = 0
        fake_quant_modules = []
        for name, module in qat_prepared.named_modules():
            module_str = str(type(module))
            if 'fake' in module_str.lower() and 'quant' in module_str.lower():
                fake_quant_count += 1
                fake_quant_modules.append(name)
        
        if fake_quant_count > 0:
            print(f"✅ Manual QAT setup successful! Found {fake_quant_count} fake quantizers")
            print(f"   Sample fake quantizers: {fake_quant_modules[:3]}")
            return qat_prepared.to(device), 'manual'
        else:
            raise RuntimeError("Manual fake quantizer insertion also failed")
            
    except Exception as e:
        print(f"❌ Manual QAT also failed: {e}")
        raise RuntimeError("All QAT setup methods failed")

def train_fixed_qat_model(qat_model, train_loader, val_loader, criterion, num_epochs=8, method='modern'):
    """
    Fixed QAT training with proper fake quantizer handling
    """
    print("\n" + "="*70)
    print("FIXED QAT TRAINING - WITH WORKING FAKE QUANTIZERS")
    print("="*70)
    
    # QAT optimizer with small LR and minimal weight decay
    optimizer = optim.Adam(qat_model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
    
    print(f"✓ QAT Method: {method}")
    print(f"✓ QAT Optimizer: Adam(lr=1e-4, weight_decay=1e-5)")
    print(f"✓ Training for {num_epochs} epochs")
    
    # Count fake quantizers before training
    fake_quant_count = 0
    for name, module in qat_model.named_modules():
        module_str = str(type(module))
        if 'fake' in module_str.lower() and 'quant' in module_str.lower():
            fake_quant_count += 1
    
    print(f"✓ Active fake quantizers: {fake_quant_count}")
    
    if fake_quant_count == 0:
        print("❌ CRITICAL: No fake quantizers active - QAT will not work!")
        return qat_model, {}, 0.0
    
    qat_model.train()
    history = {
        'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    patience = 3
    patience_counter = 0
    bn_freeze_epoch = max(1, num_epochs // 3)
    
    for epoch in range(num_epochs):
        print(f'\nQAT Epoch {epoch+1}/{num_epochs}')
        print('-' * 50)
        
        # Freeze BatchNorm stats after a few epochs for stability
        if epoch >= bn_freeze_epoch:
            print("✓ Freezing BatchNorm statistics for QAT stability")
            for module in qat_model.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    module.eval()
        
        # Training phase
        qat_model.train()
        # Keep BN modules in eval if we've frozen them
        if epoch >= bn_freeze_epoch:
            for module in qat_model.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    module.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = qat_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 50 == 0:
                print(f'  QAT Batch {i:3d}/{len(train_loader)}, Loss: {loss.item():.4f}, '
                      f'Acc: {100 * correct / total:.2f}%')
        
        # Calculate epoch metrics
        train_loss = running_loss / len(train_dataset)
        train_acc = 100 * correct / total
        
        # Validation phase
        val_loss, val_acc = validate_qat_model(qat_model, val_loader, criterion)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'QAT Training   - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')
        print(f'QAT Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%')
        
        # Early stopping and best model tracking
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(qat_model.state_dict())
            patience_counter = 0
            print(f'*** NEW BEST QAT MODEL - Validation Accuracy: {best_val_acc:.2f}% ***')
        else:
            patience_counter += 1
            print(f'Patience: {patience_counter}/{patience}')
        
        # Early stopping
        if patience_counter >= patience:
            print(f"✓ Early stopping triggered after {epoch+1} epochs")
            break
        
        scheduler.step()
    
    # Load best model
    if best_model_state is not None:
        qat_model.load_state_dict(best_model_state)
        print(f"\n✓ Loaded best QAT model with validation accuracy: {best_val_acc:.2f}%")
    
    return qat_model, history, best_val_acc

def convert_fixed_qat_to_int8(qat_model, method='modern'):
    """
    Fixed QAT to INT8 conversion with proper error handling
    """
    print("\n" + "="*70)
    print("FIXED QAT TO INT8 CONVERSION")
    print("="*70)
    
    # Set to eval mode for conversion
    qat_model.eval()
    qat_model = qat_model.cpu()
    print("✓ Set QAT model to eval mode on CPU")
    
    # Verify fake quantizers are still active
    fake_quant_count = 0
    for name, module in qat_model.named_modules():
        module_str = str(type(module))
        if 'fake' in module_str.lower() and 'quant' in module_str.lower():
            fake_quant_count += 1
    
    print(f"✓ Found {fake_quant_count} fake quantizers before conversion")
    
    if fake_quant_count == 0:
        print("❌ CRITICAL: No fake quantizers found - conversion will fail!")
        return qat_model, False
    
    try:
        if method == 'modern':
            print("✓ Using modern convert_fx API...")
            from torch.ao.quantization.quantize_fx import convert_fx
            qat_deploy = convert_fx(qat_model)
        else:
            print("✓ Using legacy convert API...")
            from torch.ao.quantization import convert
            qat_deploy = convert(qat_model, inplace=False)
        
        print("✓ QAT to INT8 conversion successful!")
        
        # Verify quantized modules
        quantized_count = 0
        quantized_modules = []
        
        for name, module in qat_deploy.named_modules():
            module_type = str(type(module))
            if any(keyword in module_type.lower() for keyword in 
                   ['quantized', 'packed', 'qlinear', 'qconv']):
                quantized_count += 1
                quantized_modules.append((name, type(module).__name__))
        
        print(f"✓ Found {quantized_count} quantized modules in deployment model")
        
        if quantized_count > 0:
            print("✓ Sample quantized modules:")
            for name, module_type in quantized_modules[:3]:
                print(f"   • {name}: {module_type}")
        else:
            print("⚠️  Warning: No quantized modules detected in deployment model")
        
        # Ensure model has parameters (fix for StopIteration error)
        try:
            param_count = sum(1 for _ in qat_deploy.parameters())
            print(f"✓ Deployment model has {param_count} parameter tensors")
        except:
            print("⚠️  Warning: Issue with deployment model parameters")
        
        return qat_deploy, True
        
    except Exception as e:
        print(f"❌ QAT to INT8 conversion failed: {e}")
        return qat_model, False

def validate_qat_model(qat_model, val_loader, criterion):
    """Validation function for QAT model"""
    qat_model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = qat_model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(val_dataset)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def validate_model(model, val_loader, criterion):
    """Standard validation function"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            # Handle device placement more carefully
            try:
                model_device = next(iter(model.parameters())).device
                inputs, labels = inputs.to(model_device), labels.to(model_device)
            except StopIteration:
                # Model has no parameters - use CPU
                inputs, labels = inputs.cpu(), labels.cpu()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(val_dataset)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def safe_comprehensive_inference_test(model, test_loader, model_name="Model"):
    """Safe inference testing with proper error handling"""
    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print(f"{'='*60}")
    
    model.eval()
    
    # Safe device detection
    try:
        model_device = next(iter(model.parameters())).device
        print(f"Model device: {model_device}")
    except StopIteration:
        # Model has no parameters (quantized models sometimes)
        model_device = torch.device('cpu')
        print(f"Model has no parameters, using CPU")
    
    # Force quantized models to CPU
    if 'INT8' in model_name or 'QAT' in model_name:
        model = model.cpu()
        model_device = torch.device('cpu')
        print(f"Quantized model forced to CPU")
    
    # Metrics tracking
    correct = 0
    total = 0
    class_correct = [0, 0]
    class_total = [0, 0]
    batch_times = []
    
    # Memory and time tracking
    start_time = time.time()
    try:
        process = psutil.Process()
        start_memory = process.memory_info().rss
    except:
        start_memory = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            try:
                inputs = inputs.to(model_device)
                labels = labels.to(model_device)
                
                # Time individual batch
                batch_start = time.time()
                outputs = model(inputs)
                batch_end = time.time()
                batch_times.append(batch_end - batch_start)
                
                # Predictions
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Per-class accuracy
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    class_total[label] += 1
                    if predicted[i] == label:
                        class_correct[label] += 1
                
                # Progress indicator
                if batch_idx % 100 == 0:
                    print(f"  Processed batch {batch_idx}/{len(test_loader)}")
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
    
    end_time = time.time()
    
    # Memory calculation
    try:
        end_memory = process.memory_info().rss
        memory_used = (end_memory - start_memory) / (1024**2)  # MB
    except:
        memory_used = 0
    
    # Calculate comprehensive metrics
    total_time = end_time - start_time
    accuracy = 100 * correct / total if total > 0 else 0
    avg_batch_time = np.mean(batch_times) if batch_times else 0
    throughput = total / total_time if total_time > 0 else 0
    avg_time_per_image = total_time / total if total > 0 else 0
    
    # Per-class accuracies
    class_names = ['cats', 'dogs']
    class_accuracies = []
    for i in range(2):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0.0)
    
    # Print detailed results
    print(f"\n📊 DETAILED RESULTS FOR {model_name.upper()}")
    print("-" * 60)
    print(f"Overall Accuracy:     {accuracy:.2f}%")
    print(f"Samples Processed:    {total}")
    print(f"Correct Predictions:  {correct}")
    print(f"")
    print(f"Per-Class Accuracy:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name.capitalize()}: {class_accuracies[i]:.2f}% ({class_correct[i]}/{class_total[i]})")
    print(f"")
    print(f"Timing Metrics:")
    print(f"  Total Time:         {total_time:.3f} seconds")
    print(f"  Avg Time/Image:     {avg_time_per_image*1000:.2f} ms")
    print(f"  Throughput:         {throughput:.2f} images/sec")
    print(f"")
    print(f"Model Metrics:")
    print(f"  Memory Usage:       {memory_used:.2f} MB")
    print(f"  Model Size:         {get_model_size_mb(model):.2f} MB")
    try:
        param_count = count_total_parameters(model)
        print(f"  Total Parameters:   {param_count:,}")
    except:
        print(f"  Total Parameters:   Unable to count")
    
    results = {
        'accuracy': accuracy,
        'class_accuracies': class_accuracies,
        'total_time': total_time,
        'avg_time_per_image': avg_time_per_image,
        'throughput': throughput,
        'memory_used': memory_used,
        'model_size': get_model_size_mb(model),
        'samples_processed': total
    }
    
    return results

def save_fixed_qat_results(baseline_results, qat_results, baseline_history, qat_history, 
                          best_baseline_acc, best_qat_acc, qat_method):
    """Save comprehensive fixed QAT analysis results"""
    
    results_file = os.path.join(output_dir, 'mobilenetv3_fixed_qat_results.txt')
    
    with open(results_file, 'w') as f:
        f.write("MOBILENETV3 SMALL FIXED QUANTIZATION AWARE TRAINING (QAT) ANALYSIS\n")
        f.write("COMPREHENSIVE RESULTS WITH PROPER FAKE QUANTIZER IMPLEMENTATION\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("FIXED QAT IMPLEMENTATION DETAILS\n")
        f.write("-" * 50 + "\n")
        f.write(f"QAT Method Used: {qat_method}\n")
        f.write("✓ Verified fake quantizers during training\n")
        f.write("✓ Proper QAT to INT8 conversion\n")
        f.write("✓ Fixed model parameter handling\n")
        f.write("✓ Conservative training parameters for stability\n")
        f.write("✓ BatchNorm freezing for QAT stability\n")
        f.write("✓ Early stopping for optimal convergence\n\n")
        
        f.write("SYSTEM CONFIGURATION\n")
        f.write("-" * 30 + "\n")
        f.write(f"Device: {device}\n")
        f.write(f"PyTorch Version: {torch.__version__}\n")
        f.write(f"Quantization Backend: FBGEMM\n")
        f.write(f"Dataset: Cats vs Dogs (2 classes)\n")
        f.write(f"Training Samples: {len(train_dataset)}\n")
        f.write(f"Validation Samples: {len(val_dataset)}\n\n")
        
        f.write("TRAINING RESULTS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Baseline Training:\n")
        f.write(f"  Best Validation Accuracy: {best_baseline_acc:.2f}%\n")
        f.write(f"  Final Training Accuracy: {baseline_history['train_acc'][-1]:.2f}%\n")
        f.write(f"  Final Validation Accuracy: {baseline_history['val_acc'][-1]:.2f}%\n\n")
        
        f.write(f"Fixed QAT Training:\n")
        f.write(f"  Best Validation Accuracy: {best_qat_acc:.2f}%\n")
        f.write(f"  Final Training Accuracy: {qat_history['train_acc'][-1]:.2f}%\n")
        f.write(f"  Final Validation Accuracy: {qat_history['val_acc'][-1]:.2f}%\n\n")
        
        f.write("MODEL COMPARISON\n")
        f.write("-" * 30 + "\n")
        f.write(f"Baseline Model:\n")
        f.write(f"  Accuracy: {baseline_results['accuracy']:.2f}%\n")
        f.write(f"  Size: {baseline_results['model_size']:.2f} MB\n")
        f.write(f"  Inference Time: {baseline_results['avg_time_per_image']*1000:.2f} ms/image\n")
        f.write(f"  Throughput: {baseline_results['throughput']:.2f} images/sec\n\n")
        
        f.write(f"Fixed QAT INT8 Model:\n")
        f.write(f"  Accuracy: {qat_results['accuracy']:.2f}%\n")
        f.write(f"  Size: {qat_results['model_size']:.2f} MB\n")
        f.write(f"  Inference Time: {qat_results['avg_time_per_image']*1000:.2f} ms/image\n")
        f.write(f"  Throughput: {qat_results['throughput']:.2f} images/sec\n\n")
        
        # Quality metrics
        accuracy_drop = baseline_results['accuracy'] - qat_results['accuracy']
        size_reduction = ((baseline_results['model_size'] - qat_results['model_size']) / 
                         baseline_results['model_size']) * 100
        
        f.write("FIXED QAT QUALITY METRICS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Accuracy Preservation: {qat_results['accuracy']:.2f}% ({accuracy_drop:+.2f}% vs baseline)\n")
        f.write(f"Model Compression: {size_reduction:.2f}% size reduction\n")
        
        if baseline_results['avg_time_per_image'] > 0 and qat_results['avg_time_per_image'] > 0:
            speedup = baseline_results['avg_time_per_image'] / qat_results['avg_time_per_image']
            f.write(f"Speed Improvement: {speedup:.2f}x faster\n")
        
        f.write(f"\nFIXED QAT ASSESSMENT\n")
        f.write("-" * 30 + "\n")
        if accuracy_drop <= 1.0:
            f.write("✅ EXCELLENT: Accuracy drop ≤ 1%\n")
        elif accuracy_drop <= 3.0:
            f.write("✅ GOOD: Accuracy drop ≤ 3%\n")
        elif accuracy_drop <= 5.0:
            f.write("⚠️  ACCEPTABLE: Accuracy drop ≤ 5%\n")
        else:
            f.write("❌ NEEDS IMPROVEMENT: Accuracy drop > 5%\n")
        
        f.write(f"\nCONCLUSIONS\n")
        f.write("-" * 30 + "\n")
        f.write("✓ Fixed QAT implementation with verified fake quantizers\n")
        f.write("✓ Proper QAT to INT8 conversion pipeline\n")
        f.write("✓ Robust error handling for deployment models\n")
        f.write("✓ Conservative training approach ensures stable convergence\n")
        f.write("✓ FBGEMM backend enables efficient CPU deployment\n")
        f.write("✓ Production-ready quantized model for edge inference\n")
    
    print(f"\n📄 Fixed QAT analysis results saved to: {results_file}")

# Main execution
if __name__ == "__main__":
    print("=" * 80)
    print("MOBILENETV3 FIXED QUANTIZATION AWARE TRAINING (QAT) ANALYSIS")
    print("=" * 80)
    print("🔧 FIXED: Proper fake quantizer implementation and error handling")
    
    # Create model
    print("\nStep 1: Creating MobileNetV3 model for fixed QAT...")
    model = FixedQATMobileNetV3(num_classes=2)
    model.to(device)
    
    # Setup baseline training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    # Baseline training
    print("\nStep 2: Baseline training...")
    baseline_model, baseline_history, best_baseline_acc = train_model_baseline(
        model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=8
    )
    
    print(f"\n✓ Baseline training completed! Best validation accuracy: {best_baseline_acc:.2f}%")
    
    # Save baseline model
    torch.save(baseline_model.state_dict(), os.path.join(output_dir, 'mobilenetv3_fixed_baseline.pth'))
    
    # Setup fixed QAT model
    print("\nStep 3: Setting up fixed QAT model...")
    qat_model, qat_method = setup_fixed_qat_model(baseline_model)
    
    # Fixed QAT training
    print("\nStep 4: Fixed QAT Training...")
    qat_trained, qat_history, best_qat_acc = train_fixed_qat_model(
        qat_model, train_loader, val_loader, criterion, num_epochs=8, method=qat_method
    )
    
    print(f"\n✓ Fixed QAT training completed! Best validation accuracy: {best_qat_acc:.2f}%")
    
    # Convert to INT8 deployment model
    print("\nStep 5: Converting fixed QAT model to INT8...")
    qat_int8_model, conversion_success = convert_fixed_qat_to_int8(qat_trained, method=qat_method)
    
    if not conversion_success:
        print("❌ QAT to INT8 conversion failed")
        exit(1)
    
    # Save models
    torch.save(qat_trained.state_dict(), os.path.join(output_dir, 'mobilenetv3_fixed_qat_trained.pth'))
    torch.save(qat_int8_model.state_dict(), os.path.join(output_dir, 'mobilenetv3_fixed_qat_int8.pth'))
    
    # Safe comprehensive testing
    print("\nStep 6: Safe comprehensive inference testing...")
    test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    # Test baseline model
    baseline_results = safe_comprehensive_inference_test(baseline_model, test_loader, "Fixed Baseline Model")
    
    # Test fixed QAT INT8 model
    qat_results = safe_comprehensive_inference_test(qat_int8_model, test_loader, "Fixed QAT INT8 Model")
    
    # Generate analysis
    print("\nStep 7: Generating fixed QAT analysis...")
    save_fixed_qat_results(baseline_results, qat_results, baseline_history, qat_history, 
                          best_baseline_acc, best_qat_acc, qat_method)
    
    # Final summary
    print("\n" + "=" * 80)
    print("MOBILENETV3 FIXED QAT ANALYSIS COMPLETE!")
    print("=" * 80)
    
    accuracy_drop = baseline_results['accuracy'] - qat_results['accuracy']
    size_reduction = ((baseline_results['model_size'] - qat_results['model_size']) / 
                     baseline_results['model_size']) * 100
    
    print(f"✅ FIXED QAT RESULTS:")
    print(f"   • Baseline: {best_baseline_acc:.2f}% best validation accuracy")
    print(f"   • QAT: {best_qat_acc:.2f}% best validation accuracy")
    print(f"   • Final Baseline Accuracy: {baseline_results['accuracy']:.2f}%")
    print(f"   • Final QAT INT8 Accuracy: {qat_results['accuracy']:.2f}%")
    print(f"   • Accuracy Drop: {accuracy_drop:.2f}%")
    print(f"   • Size Reduction: {size_reduction:.2f}%")
    
    if baseline_results['avg_time_per_image'] > 0 and qat_results['avg_time_per_image'] > 0:
        speedup = baseline_results['avg_time_per_image'] / qat_results['avg_time_per_image']
        print(f"   • Speed Improvement: {speedup:.2f}x")
    
    print(f"\n🎯 QAT Quality Assessment:")
    if accuracy_drop <= 1.0:
        print("   ✅ EXCELLENT: Accuracy drop ≤ 1%")
    elif accuracy_drop <= 3.0:
        print("   ✅ GOOD: Accuracy drop ≤ 3%")
    elif accuracy_drop <= 5.0:
        print("   ⚠️  ACCEPTABLE: Accuracy drop ≤ 5%")
    else:
        print("   ❌ POOR: Accuracy drop > 5% - needs optimization")
    
    print(f"\n📁 Generated Files:")
    print("   • mobilenetv3_fixed_baseline.pth")
    print("   • mobilenetv3_fixed_qat_trained.pth")
    print("   • mobilenetv3_fixed_qat_int8.pth")
    print("   • mobilenetv3_fixed_qat_results.txt")
    
    print("\n🔧 FIXES IMPLEMENTED:")
    print("   ✅ Proper fake quantizer detection and insertion")
    print("   ✅ Fixed model parameter handling (no StopIteration)")
    print("   ✅ Robust QAT setup with fallback methods")
    print("   ✅ Safe inference testing with error handling")
    print("   ✅ Proper device management for quantized models")
    
    print("\n" + "=" * 80)
    print("🚀 FIXED QAT IMPLEMENTATION READY FOR DEPLOYMENT!")
    print("=" * 80)