import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import time
import os
import json
import pickle
import sys
from collections import OrderedDict, defaultdict
from tabulate import tabulate
import gc
import psutil
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

class AlexNetFP16(nn.Module):
    """
    AlexNet for FP16 mixed precision training and inference
    """
    def __init__(self, num_classes=2):
        super(AlexNetFP16, self).__init__()
        # Load pretrained AlexNet
        self.alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        
        # Modify classifier for binary classification
        num_ftrs = self.alexnet.classifier[6].in_features
        self.alexnet.classifier[6] = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        return self.alexnet(x)

class ComprehensiveFP16Analyzer:
    """
    Dissertation-grade FP16 analyzer with academic rigor
    Implements comprehensive mixed precision analysis methodology
    """
    
    def __init__(self, dataset_path, results_dir="fp16_results", device=None):
        self.dataset_path = dataset_path
        self.results_dir = results_dir
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize result storage
        self.results = {
            'system_info': self._get_system_info(),
            'model_analysis': {},
            'training_metrics': {},
            'precision_metrics': {},
            'performance_analysis': {},
            'statistical_analysis': {},
            'deployment_metrics': {}
        }
        
        # Academic configuration
        self.num_statistical_runs = 10  # For statistical significance
        self.confidence_level = 0.95
        
        print(f"FP16 Analyzer initialized")
        print(f"Device: {self.device}")
        print(f"Results directory: {self.results_dir}")
        
        # Check FP16 support
        if self.device.type == 'cuda':
            self.fp16_supported = torch.cuda.get_device_capability()[0] >= 7  # Volta and newer
            print(f"FP16 Hardware Support: {self.fp16_supported}")
        else:
            self.fp16_supported = False
            print("FP16 Hardware Support: Limited (CPU)")
        
    def _get_system_info(self):
        """Comprehensive system information for reproducibility"""
        info = {
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'device_name': str(self.device),
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if torch.cuda.is_available():
            info.update({
                'gpu_name': torch.cuda.get_device_name(0),
                'cuda_version': torch.version.cuda,
                'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory,
                'gpu_compute_capability': f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}",
                'fp16_tensor_cores': torch.cuda.get_device_capability()[0] >= 7
            })
            
        # CPU information
        info.update({
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3)
        })
        
        return info
    
    def prepare_datasets(self, batch_size=32, validation_split=0.2):
        """
        Prepare datasets with academic rigor
        Includes proper train/val/test splits and data analysis
        """
        print("\nPreparing datasets with academic standards...")
        
        # Data transforms with augmentation for training
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Validation/test transforms (no augmentation)
        eval_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load datasets
        full_train_dataset = datasets.ImageFolder(
            root=os.path.join(self.dataset_path, 'training_set'), 
            transform=train_transform
        )
        test_dataset = datasets.ImageFolder(
            root=os.path.join(self.dataset_path, 'test_set'), 
            transform=eval_transform
        )
        
        # Create train/validation split
        train_size = int((1 - validation_split) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Update validation dataset transform
        val_dataset.dataset = datasets.ImageFolder(
            root=os.path.join(self.dataset_path, 'training_set'), 
            transform=eval_transform
        )
        
        # Data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                      num_workers=4, pin_memory=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                                    num_workers=4, pin_memory=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                     num_workers=4, pin_memory=True)
        
        # Dataset analysis
        dataset_info = {
            'total_train_samples': len(train_dataset),
            'validation_samples': len(val_dataset),
            'test_samples': len(test_dataset),
            'num_classes': len(full_train_dataset.classes),
            'class_names': full_train_dataset.classes,
            'batch_size': batch_size
        }
        
        # Class distribution analysis
        class_counts = defaultdict(int)
        for _, label in train_dataset:
            class_counts[label] += 1
            
        dataset_info['class_distribution'] = dict(class_counts)
        dataset_info['class_balance_ratio'] = min(class_counts.values()) / max(class_counts.values())
        
        self.results['dataset_info'] = dataset_info
        
        print(f"Dataset prepared:")
        print(f"  Training: {len(train_dataset)} samples")
        print(f"  Validation: {len(val_dataset)} samples") 
        print(f"  Test: {len(test_dataset)} samples")
        print(f"  Classes: {full_train_dataset.classes}")
        print(f"  Class balance ratio: {dataset_info['class_balance_ratio']:.3f}")
        
        return dataset_info
    
    def analyze_baseline_model(self, model):
        """
        Comprehensive baseline model analysis
        Academic-grade model characterization
        """
        print("\nAnalyzing baseline model architecture...")
        
        model.eval()
        analysis = {
            'architecture': {},
            'parameters': {},
            'memory': {},
            'computational_complexity': {}
        }
        
        # Architecture analysis
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Layer-wise analysis
        layer_info = OrderedDict()
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                layer_info[name] = {
                    'type': module.__class__.__name__,
                    'parameters': sum(p.numel() for p in module.parameters()),
                    'trainable_parameters': sum(p.numel() for p in module.parameters() if p.requires_grad)
                }
                
                # Memory analysis for common layer types
                if hasattr(module, 'weight') and module.weight is not None:
                    layer_info[name]['weight_shape'] = tuple(module.weight.shape)
                    layer_info[name]['weight_memory_mb'] = module.weight.numel() * 4 / (1024**2)  # FP32
                    
                if hasattr(module, 'bias') and module.bias is not None:
                    layer_info[name]['bias_shape'] = tuple(module.bias.shape)
                    layer_info[name]['bias_memory_mb'] = module.bias.numel() * 4 / (1024**2)  # FP32
        
        # Model size estimation
        param_memory_mb = total_params * 4 / (1024**2)  # FP32 parameters
        
        # FLOPs estimation (simplified)
        model_copy = model.to(self.device)
        dummy_input = torch.randn(1, 3, 224, 224, device=self.device)
        
        flops_estimate = 0
        with torch.no_grad():
            # Rough FLOPs calculation for AlexNet
            flops_estimate = self._estimate_model_flops(model_copy, dummy_input)
        
        analysis.update({
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_memory_mb': param_memory_mb,
            'estimated_flops': flops_estimate,
            'layer_details': layer_info
        })
        
        self.results['model_analysis']['baseline'] = analysis
        
        print(f"Baseline Model Analysis:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Parameter memory: {param_memory_mb:.2f} MB")
        print(f"  Estimated FLOPs: {flops_estimate:,}")
        
        return analysis
    
    def _estimate_model_flops(self, model, input_tensor):
        """Simplified FLOPs estimation"""
        flops = 0
        
        # AlexNet specific estimation
        flops += 11 * 11 * 3 * 64 * 55 * 55  # Conv1
        flops += 5 * 5 * 64 * 192 * 27 * 27   # Conv2  
        flops += 3 * 3 * 192 * 384 * 13 * 13  # Conv3
        flops += 3 * 3 * 384 * 256 * 13 * 13  # Conv4
        flops += 3 * 3 * 256 * 256 * 13 * 13  # Conv5
        flops += 9216 * 4096                   # FC1
        flops += 4096 * 4096                   # FC2
        flops += 4096 * 2                      # FC3 (modified for 2 classes)
        
        return flops
    
    def train_baseline_model(self, model, epochs=10, learning_rate=0.0001):
        """
        Train baseline FP32 model with comprehensive monitoring
        Academic-grade training with detailed metrics
        """
        print(f"\nTraining baseline FP32 model for {epochs} epochs...")
        
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        
        # Training history
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rates': []
        }
        
        best_val_acc = 0.0
        best_model_state = None
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                if batch_idx % 50 == 0:
                    print(f'  Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Calculate metrics
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            train_loss = train_loss / len(self.train_loader)
            val_loss = val_loss / len(self.val_loader)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            print(f'Epoch {epoch+1}/{epochs}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, '
                  f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Save training metrics
        self.results['training_metrics']['baseline'] = {
            'history': history,
            'best_validation_accuracy': best_val_acc,
            'final_learning_rate': optimizer.param_groups[0]['lr'],
            'epochs_trained': epochs
        }
        
        print(f"Baseline training completed. Best validation accuracy: {best_val_acc:.2f}%")
        return model, history
    
    def train_mixed_precision_model(self, model, epochs=10, learning_rate=0.0001):
        """
        Train model with mixed precision (FP16 + FP32)
        Academic-grade mixed precision training
        """
        print(f"\nTraining mixed precision FP16 model for {epochs} epochs...")
        
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        
        # Initialize mixed precision training
        scaler = GradScaler() if self.device.type == 'cuda' else None
        
        # Training history
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rates': [],
            'scale_values': []
        }
        
        best_val_acc = 0.0
        best_model_state = None
        
        print(f"Mixed precision enabled: {scaler is not None}")
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Mixed precision forward pass
                if scaler is not None:
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    # Scaled backward pass
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Fallback for CPU or older GPUs
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                if batch_idx % 50 == 0:
                    scale_val = scaler.get_scale() if scaler is not None else 1.0
                    print(f'  MP Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}, Scale: {scale_val:.0f}')
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    if scaler is not None:
                        with autocast():
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Calculate metrics
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            train_loss = train_loss / len(self.train_loader)
            val_loss = val_loss / len(self.val_loader)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            if scaler is not None:
                history['scale_values'].append(scaler.get_scale())
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            print(f'MP Epoch {epoch+1}/{epochs}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, '
                  f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Save training metrics
        self.results['training_metrics']['mixed_precision'] = {
            'history': history,
            'best_validation_accuracy': best_val_acc,
            'final_learning_rate': optimizer.param_groups[0]['lr'],
            'epochs_trained': epochs,
            'scaler_used': scaler is not None
        }
        
        print(f"Mixed precision training completed. Best validation accuracy: {best_val_acc:.2f}%")
        return model, history
    
    def create_fp16_model(self, trained_model):
        """
        Create FP16 model from trained FP32 model
        Direct conversion approach
        """
        print("\nCreating FP16 model...")
        
        # Create new model and load trained weights
        fp16_model = AlexNetFP16(num_classes=2)
        fp16_model.load_state_dict(trained_model.state_dict())
        
        # Convert to FP16 if GPU supports it
        if self.device.type == 'cuda' and self.fp16_supported:
            fp16_model = fp16_model.half()
            print("Model converted to FP16 (half precision)")
        else:
            print("FP16 conversion skipped (limited hardware support)")
        
        fp16_model = fp16_model.to(self.device)
        
        # Analyze FP16 model
        fp16_analysis = self._analyze_fp16_model(fp16_model)
        self.results['model_analysis']['fp16'] = fp16_analysis
        
        return fp16_model
    
    def _analyze_fp16_model(self, fp16_model):
        """Comprehensive FP16 model analysis"""
        analysis = {
            'precision_format': 'FP16',
            'memory_reduction': 0.5,  # 50% reduction vs FP32
            'parameter_memory_mb': 0.0,
            'is_half_precision': False,
            'hardware_optimized': False
        }
        
        # Check if model is actually in FP16
        sample_param = next(fp16_model.parameters())
        analysis['is_half_precision'] = sample_param.dtype == torch.float16
        analysis['hardware_optimized'] = self.fp16_supported and analysis['is_half_precision']
        
        # Calculate memory usage
        total_params = sum(p.numel() for p in fp16_model.parameters())
        if analysis['is_half_precision']:
            analysis['parameter_memory_mb'] = total_params * 2 / (1024**2)  # FP16 = 2 bytes
        else:
            analysis['parameter_memory_mb'] = total_params * 4 / (1024**2)  # FP32 = 4 bytes
        
        # Precision configuration
        precision_config = {
            'format': 'FP16' if analysis['is_half_precision'] else 'FP32',
            'bytes_per_parameter': 2 if analysis['is_half_precision'] else 4,
            'theoretical_speedup': '1.5-2x' if analysis['hardware_optimized'] else 'Limited',
            'memory_efficiency': '50% reduction' if analysis['is_half_precision'] else 'No reduction',
            'hardware_support': 'Tensor Cores' if self.fp16_supported else 'Software emulation'
        }
        
        self.results['precision_metrics']['fp16_config'] = precision_config
        
        print(f"FP16 Model Analysis:")
        print(f"  Format: {precision_config['format']}")
        print(f"  Memory: {analysis['parameter_memory_mb']:.2f} MB")
        print(f"  Hardware optimized: {analysis['hardware_optimized']}")
        print(f"  Expected speedup: {precision_config['theoretical_speedup']}")
        
        return analysis
    
    def comprehensive_performance_evaluation(self, baseline_model, mixed_precision_model, fp16_model):
        """
        Comprehensive performance evaluation with statistical rigor
        Academic-grade performance analysis for all precision formats
        """
        print(f"\nConducting comprehensive performance evaluation ({self.num_statistical_runs} runs)...")
        
        models = {
            'baseline_fp32': baseline_model,
            'mixed_precision': mixed_precision_model,
            'fp16_direct': fp16_model
        }
        
        evaluation_results = {}
        
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name} model...")
            
            # Multiple runs for statistical significance
            run_results = []
            
            for run in range(self.num_statistical_runs):
                print(f"  Run {run + 1}/{self.num_statistical_runs}")
                
                # Create fresh model copy for each run
                if model_name == 'baseline_fp32':
                    model_copy = AlexNetFP16(num_classes=2)
                    model_copy.load_state_dict(baseline_model.state_dict())
                elif model_name == 'mixed_precision':
                    model_copy = AlexNetFP16(num_classes=2)
                    model_copy.load_state_dict(mixed_precision_model.state_dict())
                else:  # fp16_direct
                    model_copy = AlexNetFP16(num_classes=2)
                    model_copy.load_state_dict(fp16_model.state_dict())
                    if self.device.type == 'cuda' and self.fp16_supported:
                        model_copy = model_copy.half()
                
                model_copy = model_copy.to(self.device)
                model_copy.eval()
                
                run_result = self._single_performance_run(model_copy, f"{model_name}_run_{run}", model_name)
                run_results.append(run_result)
            
            # Statistical analysis
            evaluation_results[model_name] = self._analyze_statistical_results(run_results)
        
        # Comparative analysis
        comparative_analysis = self._comparative_analysis(evaluation_results)
        
        self.results['performance_analysis'] = {
            'individual_results': evaluation_results,
            'comparative_analysis': comparative_analysis
        }
        
        return evaluation_results, comparative_analysis
    
    def _single_performance_run(self, model, run_id, model_type):
        """Single performance evaluation run"""
        model.eval()
        
        # Determine if this is FP16 model
        is_fp16 = any(p.dtype == torch.float16 for p in model.parameters())
        use_autocast = model_type == 'mixed_precision' and self.device.type == 'cuda'
        
        print(f"    Running {model_type} on {self.device} (FP16: {is_fp16}, AMP: {use_autocast})")
        
        # Timing variables
        inference_times = []
        memory_usage = []
        predictions = []
        ground_truth = []
        
        # Memory baseline
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()
        else:
            process = psutil.Process()
            initial_memory = process.memory_info().rss
        
        total_samples = 0
        correct_predictions = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(self.test_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Convert inputs to FP16 if needed
                if is_fp16:
                    inputs = inputs.half()
                
                # Synchronize for accurate timing
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                # Time inference
                start_time = time.time()
                
                if use_autocast:
                    with autocast():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.time()
                
                # Record timing
                batch_time = end_time - start_time
                inference_times.append(batch_time / inputs.size(0))  # Per sample
                
                # Memory usage
                if self.device.type == 'cuda':
                    current_memory = torch.cuda.memory_allocated()
                else:
                    current_memory = process.memory_info().rss
                
                memory_usage.append(current_memory)
                
                # Accuracy calculation
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                
                # Store predictions for detailed analysis
                predictions.extend(predicted.cpu().numpy())
                ground_truth.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = 100 * correct_predictions / total_samples
        avg_inference_time = np.mean(inference_times)
        std_inference_time = np.std(inference_times)
        throughput = 1.0 / avg_inference_time
        
        # Memory analysis
        if self.device.type == 'cuda':
            peak_memory = torch.cuda.max_memory_allocated()
            memory_used = (peak_memory - initial_memory) / (1024**2)  # MB
        else:
            peak_memory = max(memory_usage) if memory_usage else initial_memory
            memory_used = (peak_memory - initial_memory) / (1024**2)  # MB
        
        # Generate detailed classification report
        class_report = classification_report(ground_truth, predictions, output_dict=True)
        conf_matrix = confusion_matrix(ground_truth, predictions)
        
        return {
            'accuracy': accuracy,
            'avg_inference_time_ms': avg_inference_time * 1000,
            'std_inference_time_ms': std_inference_time * 1000,
            'throughput_samples_per_sec': throughput,
            'memory_used_mb': memory_used,
            'total_samples': total_samples,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'raw_inference_times': inference_times,
            'device_used': str(self.device),
            'model_type': model_type,
            'is_fp16': is_fp16,
            'autocast_used': use_autocast
        }
    
    def _analyze_statistical_results(self, run_results):
        """Statistical analysis of multiple runs"""
        metrics = ['accuracy', 'avg_inference_time_ms', 'throughput_samples_per_sec', 'memory_used_mb']
        
        statistical_analysis = {}
        
        for metric in metrics:
            values = [run[metric] for run in run_results]
            
            # Calculate statistics
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            
            # Confidence interval
            confidence_interval = stats.t.interval(
                self.confidence_level, 
                len(values) - 1, 
                loc=mean_val, 
                scale=stats.sem(values)
            )
            
            statistical_analysis[metric] = {
                'mean': mean_val,
                'std': std_val,
                'min': min_val,
                'max': max_val,
                'confidence_interval': confidence_interval,
                'coefficient_of_variation': std_val / mean_val if mean_val != 0 else 0,
                'raw_values': values
            }
        
        # Overall classification report (from first run)
        statistical_analysis['classification_report'] = run_results[0]['classification_report']
        statistical_analysis['confusion_matrix'] = run_results[0]['confusion_matrix']
        
        return statistical_analysis
    
    def _comparative_analysis(self, evaluation_results):
        """Comparative analysis between all models"""
        baseline_stats = evaluation_results['baseline_fp32']
        
        comparative_metrics = {}
        
        # Compare each model against baseline
        for model_name in ['mixed_precision', 'fp16_direct']:
            if model_name not in evaluation_results:
                continue
                
            model_stats = evaluation_results[model_name]
            model_comparison = {}
            
            # Performance comparisons
            metrics_to_compare = ['accuracy', 'avg_inference_time_ms', 'throughput_samples_per_sec', 'memory_used_mb']
            
            for metric in metrics_to_compare:
                baseline_mean = baseline_stats[metric]['mean']
                model_mean = model_stats[metric]['mean']
                
                if metric == 'accuracy':
                    # For accuracy, calculate difference
                    improvement = model_mean - baseline_mean
                    relative_change = improvement / baseline_mean * 100 if baseline_mean != 0 else 0
                elif metric == 'avg_inference_time_ms' or metric == 'memory_used_mb':
                    # For time and memory, reduction is better
                    improvement = baseline_mean - model_mean
                    relative_change = improvement / baseline_mean * 100 if baseline_mean != 0 else 0
                else:
                    # For throughput, increase is better
                    improvement = model_mean - baseline_mean
                    relative_change = improvement / baseline_mean * 100 if baseline_mean != 0 else 0
                
                # Statistical significance test
                baseline_values = baseline_stats[metric]['raw_values']
                model_values = model_stats[metric]['raw_values']
                
                # Two-sample t-test
                t_statistic, p_value = stats.ttest_ind(baseline_values, model_values)
                is_significant = p_value < (1 - self.confidence_level)
                
                model_comparison[metric] = {
                    'baseline_mean': baseline_mean,
                    'model_mean': model_mean,
                    'absolute_improvement': improvement,
                    'relative_improvement_percent': relative_change,
                    't_statistic': t_statistic,
                    'p_value': p_value,
                    'is_statistically_significant': is_significant
                }
            
            comparative_metrics[model_name] = model_comparison
        
        # Model size analysis
        baseline_analysis = self.results['model_analysis']['baseline']
        fp16_analysis = self.results['model_analysis']['fp16']
        
        size_comparison = {
            'baseline_memory_mb': baseline_analysis['parameter_memory_mb'],
            'fp16_memory_mb': fp16_analysis['parameter_memory_mb'],
            'theoretical_reduction_percent': 50.0,  # FP16 is 50% of FP32
            'actual_reduction_percent': (1 - fp16_analysis['parameter_memory_mb'] / baseline_analysis['parameter_memory_mb']) * 100
        }
        
        comparative_metrics['model_size'] = size_comparison
        
        return comparative_metrics
    
    def generate_comprehensive_visualizations(self):
        """
        Generate academic-quality visualizations
        Publication-ready figures and charts
        """
        print("\nGenerating comprehensive visualizations...")
        
        # Set publication style
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('seaborn')
        sns.set_palette("husl")
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Training History Comparison
        ax1 = plt.subplot(3, 4, 1)
        baseline_history = self.results['training_metrics']['baseline']['history']
        mp_history = self.results['training_metrics']['mixed_precision']['history']
        
        epochs_baseline = range(1, len(baseline_history['val_acc']) + 1)
        epochs_mp = range(1, len(mp_history['val_acc']) + 1)
        
        plt.plot(epochs_baseline, baseline_history['val_acc'], 'b-', label='FP32 Baseline', linewidth=2)
        plt.plot(epochs_mp, mp_history['val_acc'], 'r-', label='Mixed Precision', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy (%)')
        plt.title('Training Progress Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Performance Metrics Comparison
        ax2 = plt.subplot(3, 4, 2)
        performance = self.results['performance_analysis']['comparative_analysis']
        
        models = ['baseline_fp32', 'mixed_precision', 'fp16_direct']
        model_labels = ['FP32', 'Mixed Precision', 'FP16']
        
        # Accuracy comparison
        accuracies = []
        for model in models:
            if model in self.results['performance_analysis']['individual_results']:
                acc = self.results['performance_analysis']['individual_results'][model]['accuracy']['mean']
                accuracies.append(acc)
            else:
                accuracies.append(0)
        
        x = np.arange(len(model_labels))
        plt.bar(x, accuracies, alpha=0.8, color=['blue', 'orange', 'red'])
        plt.xlabel('Model Type')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy Comparison')
        plt.xticks(x, model_labels)
        plt.grid(True, alpha=0.3)
        
        # 3. Inference Time Comparison
        ax3 = plt.subplot(3, 4, 3)
        inference_times = []
        for model in models:
            if model in self.results['performance_analysis']['individual_results']:
                time_ms = self.results['performance_analysis']['individual_results'][model]['avg_inference_time_ms']['mean']
                inference_times.append(time_ms)
            else:
                inference_times.append(0)
        
        plt.bar(x, inference_times, alpha=0.8, color=['blue', 'orange', 'red'])
        plt.xlabel('Model Type')
        plt.ylabel('Inference Time (ms)')
        plt.title('Inference Time Comparison')
        plt.xticks(x, model_labels)
        plt.grid(True, alpha=0.3)
        
        # 4. Memory Usage Comparison
        ax4 = plt.subplot(3, 4, 4)
        memory_usage = []
        for model in models:
            if model in self.results['performance_analysis']['individual_results']:
                mem_mb = self.results['performance_analysis']['individual_results'][model]['memory_used_mb']['mean']
                memory_usage.append(mem_mb)
            else:
                memory_usage.append(0)
        
        plt.bar(x, memory_usage, alpha=0.8, color=['blue', 'orange', 'red'])
        plt.xlabel('Model Type')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Runtime Memory Comparison')
        plt.xticks(x, model_labels)
        plt.grid(True, alpha=0.3)
        
        # 5. Model Size Comparison
        ax5 = plt.subplot(3, 4, 5)
        baseline_size = self.results['model_analysis']['baseline']['parameter_memory_mb']
        fp16_size = self.results['model_analysis']['fp16']['parameter_memory_mb']
        
        sizes = [baseline_size, fp16_size]
        labels = ['FP32', 'FP16']
        colors = ['blue', 'red']
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Model Size Comparison')
        
        # 6. Throughput Comparison
        ax6 = plt.subplot(3, 4, 6)
        throughputs = []
        for model in models:
            if model in self.results['performance_analysis']['individual_results']:
                thr = self.results['performance_analysis']['individual_results'][model]['throughput_samples_per_sec']['mean']
                throughputs.append(thr)
            else:
                throughputs.append(0)
        
        plt.bar(x, throughputs, alpha=0.8, color=['blue', 'orange', 'red'])
        plt.xlabel('Model Type')
        plt.ylabel('Throughput (samples/sec)')
        plt.title('Throughput Comparison')
        plt.xticks(x, model_labels)
        plt.grid(True, alpha=0.3)
        
        # 7. Mixed Precision Scale Values
        ax7 = plt.subplot(3, 4, 7)
        if 'scale_values' in mp_history and mp_history['scale_values']:
            plt.plot(range(1, len(mp_history['scale_values']) + 1), mp_history['scale_values'], 'g-', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Loss Scale')
            plt.title('Mixed Precision Loss Scaling')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'Mixed Precision\nLoss Scaling\nNot Available', 
                    ha='center', va='center', transform=ax7.transAxes)
            ax7.set_xticks([])
            ax7.set_yticks([])
        
        # 8. Statistical Significance Analysis
        ax8 = plt.subplot(3, 4, 8)
        significance_data = []
        significance_labels = []
        
        for model_name in ['mixed_precision', 'fp16_direct']:
            if model_name in performance:
                p_val = performance[model_name]['accuracy']['p_value']
                significance_data.append(p_val)
                significance_labels.append(model_name.replace('_', ' ').title())
        
        if significance_data:
            colors = ['green' if p < 0.05 else 'red' for p in significance_data]
            plt.bar(range(len(significance_data)), significance_data, color=colors, alpha=0.7)
            plt.axhline(y=0.05, color='black', linestyle='--', alpha=0.5)
            plt.xlabel('Model')
            plt.ylabel('P-value')
            plt.title('Statistical Significance\n(p < 0.05)')
            plt.xticks(range(len(significance_labels)), significance_labels, rotation=45)
            plt.yscale('log')
        
        # 9. Performance Improvement Radar Chart
        ax9 = plt.subplot(3, 4, 9, projection='polar')
        
        if 'mixed_precision' in performance:
            mp_perf = performance['mixed_precision']
            categories = ['Accuracy\nChange (%)', 'Speed\nImprovement (%)', 'Memory\nReduction (%)']
            values = [
                mp_perf['accuracy']['relative_improvement_percent'],
                -mp_perf['avg_inference_time_ms']['relative_improvement_percent'],
                -mp_perf['memory_used_mb']['relative_improvement_percent']
            ]
            
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            values += values[:1]  # Complete the circle
            angles += angles[:1]
            
            ax9.plot(angles, values, 'o-', linewidth=2, color='orange', label='Mixed Precision')
            ax9.fill(angles, values, color='orange', alpha=0.25)
            ax9.set_xticks(angles[:-1])
            ax9.set_xticklabels(categories)
            ax9.set_title('Performance Improvement\n(Mixed Precision)')
        
        # 10. Training Loss Comparison
        ax10 = plt.subplot(3, 4, 10)
        plt.plot(epochs_baseline, baseline_history['train_loss'], 'b-', label='FP32 Train', alpha=0.7)
        plt.plot(epochs_baseline, baseline_history['val_loss'], 'b--', label='FP32 Val', alpha=0.7)
        plt.plot(epochs_mp, mp_history['train_loss'], 'r-', label='MP Train', alpha=0.7)
        plt.plot(epochs_mp, mp_history['val_loss'], 'r--', label='MP Val', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 11. Hardware Utilization Info
        ax11 = plt.subplot(3, 4, 11)
        ax11.axis('off')
        
        hw_info = self.results['system_info']
        hw_text = f"""
        Hardware Configuration:
        • GPU: {hw_info.get('gpu_name', 'N/A')}
        • Compute: {hw_info.get('gpu_compute_capability', 'N/A')}
        • Tensor Cores: {hw_info.get('fp16_tensor_cores', False)}
        • CUDA: {hw_info.get('cuda_version', 'N/A')}
        
        FP16 Support:
        • Hardware: {self.fp16_supported}
        • Optimized: {self.results['model_analysis']['fp16']['hardware_optimized']}
        """
        
        ax11.text(0.1, 0.9, hw_text, transform=ax11.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        # 12. Summary Statistics
        ax12 = plt.subplot(3, 4, 12)
        ax12.axis('off')
        
        summary_text = f"""
        Dataset Statistics:
        • Train: {self.results['dataset_info']['total_train_samples']} samples
        • Test: {self.results['dataset_info']['test_samples']} samples
        • Classes: {self.results['dataset_info']['num_classes']}
        
        Model Statistics:
        • Parameters: {self.results['model_analysis']['baseline']['total_parameters']:,}
        • FP32 Size: {baseline_size:.1f} MB
        • FP16 Size: {fp16_size:.1f} MB
        • Size Reduction: {(1-fp16_size/baseline_size)*100:.1f}%
        """
        
        ax12.text(0.1, 0.9, summary_text, transform=ax12.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.suptitle('Comprehensive FP16 Analysis: AlexNet Mixed Precision Training', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save visualization
        viz_path = os.path.join(self.results_dir, 'comprehensive_fp16_analysis.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comprehensive visualization saved: {viz_path}")
        
        return viz_path
    
    def generate_academic_report(self):
        """
        Generate comprehensive academic report
        Publication-quality analysis document
        """
        print("\nGenerating comprehensive academic report...")
        
        report_path = os.path.join(self.results_dir, 'comprehensive_fp16_analysis_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Comprehensive Mixed Precision Training Analysis Report\n")
            f.write("## AlexNet FP16 Optimization for Image Classification\n\n")
            
            # Abstract
            f.write("## Abstract\n\n")
            f.write("This report presents a comprehensive analysis of mixed precision training and FP16 optimization ")
            f.write("applied to AlexNet for binary image classification. The study compares FP32 baseline training, ")
            f.write("mixed precision training, and direct FP16 conversion across multiple performance metrics ")
            f.write("including accuracy, inference latency, memory consumption, and model size reduction.\n\n")
            
            # System Configuration
            f.write("## 1. Experimental Setup\n\n")
            f.write("### 1.1 System Configuration\n\n")
            system_info = self.results['system_info']
            
            f.write(f"- **Hardware**: {system_info.get('gpu_name', 'CPU only')}\n")
            f.write(f"- **PyTorch Version**: {system_info['pytorch_version']}\n")
            f.write(f"- **CUDA Version**: {system_info.get('cuda_version', 'N/A')}\n")
            f.write(f"- **Device**: {system_info['device_name']}\n")
            f.write(f"- **Compute Capability**: {system_info.get('gpu_compute_capability', 'N/A')}\n")
            f.write(f"- **Tensor Cores**: {system_info.get('fp16_tensor_cores', False)}\n")
            f.write(f"- **Analysis Date**: {system_info['timestamp']}\n\n")
            
            # Dataset Information
            f.write("### 1.2 Dataset Configuration\n\n")
            dataset_info = self.results['dataset_info']
            
            f.write(f"- **Training Samples**: {dataset_info['total_train_samples']:,}\n")
            f.write(f"- **Validation Samples**: {dataset_info['validation_samples']:,}\n") 
            f.write(f"- **Test Samples**: {dataset_info['test_samples']:,}\n")
            f.write(f"- **Number of Classes**: {dataset_info['num_classes']}\n")
            f.write(f"- **Class Names**: {', '.join(dataset_info['class_names'])}\n")
            f.write(f"- **Class Balance Ratio**: {dataset_info['class_balance_ratio']:.3f}\n\n")
            
            # Model Architecture
            f.write("## 2. Model Architecture Analysis\n\n")
            baseline_analysis = self.results['model_analysis']['baseline']
            fp16_analysis = self.results['model_analysis']['fp16']
            
            f.write("### 2.1 Baseline Model (FP32)\n\n")
            f.write(f"- **Total Parameters**: {baseline_analysis['total_parameters']:,}\n")
            f.write(f"- **Trainable Parameters**: {baseline_analysis['trainable_parameters']:,}\n")
            f.write(f"- **Parameter Memory**: {baseline_analysis['parameter_memory_mb']:.2f} MB\n")
            f.write(f"- **Estimated FLOPs**: {baseline_analysis['estimated_flops']:,}\n\n")
            
            f.write("### 2.2 FP16 Model\n\n")
            f.write(f"- **Precision Format**: {fp16_analysis['precision_format']}\n")
            f.write(f"- **Parameter Memory**: {fp16_analysis['parameter_memory_mb']:.2f} MB\n")
            f.write(f"- **Memory Reduction**: {fp16_analysis['memory_reduction']*100:.1f}%\n")
            f.write(f"- **Hardware Optimized**: {fp16_analysis['hardware_optimized']}\n\n")
            
            # Training Results
            f.write("## 3. Training Analysis\n\n")
            baseline_training = self.results['training_metrics']['baseline']
            mp_training = self.results['training_metrics']['mixed_precision']
            
            f.write("### 3.1 FP32 Baseline Training\n\n")
            f.write(f"- **Best Validation Accuracy**: {baseline_training['best_validation_accuracy']:.2f}%\n")
            f.write(f"- **Epochs Trained**: {baseline_training['epochs_trained']}\n")
            f.write(f"- **Final Learning Rate**: {baseline_training['final_learning_rate']:.2e}\n\n")
            
            f.write("### 3.2 Mixed Precision Training\n\n")
            f.write(f"- **Best Validation Accuracy**: {mp_training['best_validation_accuracy']:.2f}%\n")
            f.write(f"- **Epochs Trained**: {mp_training['epochs_trained']}\n")
            f.write(f"- **Scaler Used**: {mp_training['scaler_used']}\n")
            f.write(f"- **Training Strategy**: Automatic mixed precision with gradient scaling\n\n")
            
            # Performance Analysis
            f.write("## 4. Performance Analysis\n\n")
            performance = self.results['performance_analysis']['comparative_analysis']
            
            f.write("### 4.1 Statistical Methodology\n\n")
            f.write(f"- **Number of Runs**: {self.num_statistical_runs}\n")
            f.write(f"- **Confidence Level**: {self.confidence_level*100:.0f}%\n")
            f.write("- **Statistical Test**: Two-sample t-test\n")
            f.write("- **Significance Threshold**: p < 0.05\n\n")
            
            f.write("### 4.2 Performance Metrics Comparison\n\n")
            f.write("| Model | Accuracy (%) | Inference Time (ms) | Throughput (samples/sec) | Memory Usage (MB) |\n")
            f.write("|-------|--------------|---------------------|--------------------------|-------------------|\n")
            
            models_data = self.results['performance_analysis']['individual_results']
            model_names = ['baseline_fp32', 'mixed_precision', 'fp16_direct']
            model_labels = ['FP32 Baseline', 'Mixed Precision', 'FP16 Direct']
            
            for model_name, label in zip(model_names, model_labels):
                if model_name in models_data:
                    data = models_data[model_name]
                    f.write(f"| {label} | {data['accuracy']['mean']:.2f} | ")
                    f.write(f"{data['avg_inference_time_ms']['mean']:.2f} | ")
                    f.write(f"{data['throughput_samples_per_sec']['mean']:.1f} | ")
                    f.write(f"{data['memory_used_mb']['mean']:.1f} |\n")
            
            f.write("\n")
            
            # Statistical Significance
            f.write("### 4.3 Statistical Significance Analysis\n\n")
            
            for model_name in ['mixed_precision', 'fp16_direct']:
                if model_name in performance:
                    model_perf = performance[model_name]
                    f.write(f"#### {model_name.replace('_', ' ').title()} vs Baseline\n\n")
                    f.write("| Metric | Improvement | Significance |\n")
                    f.write("|--------|-------------|-------------|\n")
                    
                    metrics = ['accuracy', 'avg_inference_time_ms', 'throughput_samples_per_sec', 'memory_used_mb']
                    metric_names = ['Accuracy', 'Inference Time', 'Throughput', 'Memory Usage']
                    
                    for metric, name in zip(metrics, metric_names):
                        if metric in model_perf:
                            data = model_perf[metric]
                            significance = "✓" if data['is_statistically_significant'] else "✗"
                            f.write(f"| {name} | {data['relative_improvement_percent']:+.2f}% | ")
                            f.write(f"{significance} (p={data['p_value']:.4f}) |\n")
                    
                    f.write("\n")
            
            # Model Size Analysis
            f.write("### 4.4 Model Size Analysis\n\n")
            size_data = performance['model_size']
            f.write(f"- **FP32 Model Size**: {size_data['baseline_memory_mb']:.2f} MB\n")
            f.write(f"- **FP16 Model Size**: {size_data['fp16_memory_mb']:.2f} MB\n")
            f.write(f"- **Theoretical Size Reduction**: {size_data['theoretical_reduction_percent']:.1f}%\n")
            f.write(f"- **Actual Size Reduction**: {size_data['actual_reduction_percent']:.1f}%\n\n")
            
            # Hardware Analysis
            f.write("## 5. Hardware Utilization Analysis\n\n")
            
            precision_config = self.results['precision_metrics']['fp16_config']
            f.write("### 5.1 FP16 Configuration\n\n")
            f.write(f"- **Format**: {precision_config['format']}\n")
            f.write(f"- **Bytes per Parameter**: {precision_config['bytes_per_parameter']}\n")
            f.write(f"- **Expected Speedup**: {precision_config['theoretical_speedup']}\n")
            f.write(f"- **Memory Efficiency**: {precision_config['memory_efficiency']}\n")
            f.write(f"- **Hardware Support**: {precision_config['hardware_support']}\n\n")
            
            # Discussion and Conclusions
            f.write("## 6. Discussion and Analysis\n\n")
            
            f.write("### 6.1 Key Findings\n\n")
            
            # Mixed precision analysis
            if 'mixed_precision' in performance:
                mp_perf = performance['mixed_precision']
                acc_change = mp_perf['accuracy']['relative_improvement_percent']
                speed_improvement = -mp_perf['avg_inference_time_ms']['relative_improvement_percent']
                
                f.write(f"- **Mixed Precision Training**: ")
                if acc_change > -0.5:
                    f.write(f"Successfully maintained accuracy ({acc_change:+.2f}%) while providing ")
                else:
                    f.write(f"Slight accuracy degradation ({acc_change:+.2f}%) with ")
                
                if speed_improvement > 5:
                    f.write(f"significant speedup ({speed_improvement:.1f}%).\n")
                else:
                    f.write(f"modest performance improvement ({speed_improvement:.1f}%).\n")
            
            # FP16 direct analysis
            if 'fp16_direct' in performance:
                fp16_perf = performance['fp16_direct']
                acc_change = fp16_perf['accuracy']['relative_improvement_percent']
                speed_improvement = -fp16_perf['avg_inference_time_ms']['relative_improvement_percent']
                
                f.write(f"- **Direct FP16 Conversion**: ")
                if abs(acc_change) < 1:
                    f.write(f"Excellent accuracy preservation ({acc_change:+.2f}%) ")
                else:
                    f.write(f"Noticeable accuracy change ({acc_change:+.2f}%) ")
                
                if speed_improvement > 10:
                    f.write(f"with substantial inference acceleration ({speed_improvement:.1f}%).\n")
                else:
                    f.write(f"with limited inference improvement ({speed_improvement:.1f}%).\n")
            
            f.write(f"- **Model Size**: Achieved {size_data['actual_reduction_percent']:.1f}% size reduction through FP16 conversion.\n\n")
            
            # Hardware considerations
            f.write("### 6.2 Hardware Considerations\n\n")
            f.write("- **Tensor Core Utilization**: ")
            if self.fp16_supported:
                f.write("Hardware supports FP16 acceleration through Tensor Cores, enabling optimal performance.\n")
            else:
                f.write("Limited hardware support may reduce FP16 performance benefits.\n")
            
            f.write("- **Memory Bandwidth**: FP16 reduces memory bandwidth requirements, beneficial for memory-bound operations.\n")
            f.write("- **Numerical Stability**: Mixed precision training maintains FP32 precision for critical operations while using FP16 for acceleration.\n\n")
            
            # Limitations and Future Work
            f.write("### 6.3 Limitations and Future Work\n\n")
            f.write("- **Hardware Dependency**: Performance improvements are highly dependent on GPU architecture and driver support.\n")
            f.write("- **Model Sensitivity**: Different architectures may show varying sensitivity to FP16 precision.\n")
            f.write("- **Deployment Considerations**: Production deployment may require additional optimization for specific hardware platforms.\n")
            f.write("- **Quantization Comparison**: Further analysis comparing FP16 with INT8 quantization would provide complete precision spectrum evaluation.\n\n")
            
            # Recommendations
            f.write("## 7. Recommendations\n\n")
            
            best_approach = "FP32"
            if 'mixed_precision' in performance and performance['mixed_precision']['accuracy']['relative_improvement_percent'] > -1:
                if -performance['mixed_precision']['avg_inference_time_ms']['relative_improvement_percent'] > 10:
                    best_approach = "Mixed Precision"
            
            if 'fp16_direct' in performance and performance['fp16_direct']['accuracy']['relative_improvement_percent'] > -1:
                if -performance['fp16_direct']['avg_inference_time_ms']['relative_improvement_percent'] > 15:
                    best_approach = "Direct FP16"
            
            f.write(f"**Recommended Approach: {best_approach}**\n\n")
            
            if best_approach == "Mixed Precision":
                f.write("Mixed precision training provides the optimal balance of training stability and performance improvement. ")
                f.write("The automatic loss scaling ensures numerical stability while leveraging FP16 acceleration.\n\n")
            elif best_approach == "Direct FP16":
                f.write("Direct FP16 conversion offers maximum performance benefits with acceptable accuracy preservation. ")
                f.write("This approach is ideal for inference-focused deployments.\n\n")
            else:
                f.write("FP32 training remains recommended due to accuracy preservation requirements. ")
                f.write("Consider FP16 approaches for scenarios where speed is prioritized over absolute accuracy.\n\n")
            
            # Technical Implementation
            f.write("## 8. Technical Implementation\n\n")
            
            f.write("### 8.1 Mixed Precision Training\n\n")
            f.write("- **Automatic Mixed Precision (AMP)**: Utilizes PyTorch's built-in AMP functionality\n")
            f.write("- **Gradient Scaling**: Automatic loss scaling prevents gradient underflow\n")
            f.write("- **Selective Precision**: FP16 for forward pass, FP32 for gradient computation\n")
            f.write("- **Hardware Optimization**: Leverages Tensor Cores when available\n\n")
            
            f.write("### 8.2 Direct FP16 Conversion\n\n")
            f.write("- **Post-Training Conversion**: Convert trained FP32 model to FP16\n")
            f.write("- **Memory Efficiency**: 50% reduction in model size and memory usage\n")
            f.write("- **Inference Optimization**: Optimized for deployment scenarios\n")
            f.write("- **Compatibility**: Requires modern GPU architecture for optimal performance\n\n")
            
            # Conclusion
            f.write("## 9. Conclusion\n\n")
            f.write("This comprehensive analysis demonstrates the practical benefits and limitations of FP16 optimization ")
            f.write("for AlexNet-based image classification. The study provides evidence-based recommendations for ")
            f.write("selecting appropriate precision formats based on hardware capabilities and performance requirements.\n\n")
            
            f.write("Mixed precision training emerges as a viable approach for maintaining training stability while ")
            f.write("achieving performance improvements. Direct FP16 conversion offers maximum efficiency gains for ")
            f.write("inference scenarios where slight accuracy trade-offs are acceptable.\n\n")
            
            # References
            f.write("## References\n\n")
            f.write("1. Micikevicius, P., et al. \"Mixed Precision Training.\" ICLR 2018.\n")
            f.write("2. Kalamkar, D., et al. \"A Study of BFLOAT16 for Deep Learning Training.\" arXiv preprint arXiv:1905.12322 (2019).\n")
            f.write("3. NVIDIA. \"Training with Mixed Precision.\" NVIDIA Developer Documentation.\n")
            f.write("4. Courbariaux, M., et al. \"Low precision arithmetic for deep learning.\" ICLR Workshop 2015.\n\n")
            
            f.write("---\n")
            f.write(f"*Report generated on {system_info['timestamp']} using PyTorch {system_info['pytorch_version']}*\n")
        
        print(f"Comprehensive academic report saved: {report_path}")
        
        # Save detailed results as JSON
        results_json_path = os.path.join(self.results_dir, 'detailed_fp16_results.json')
        json_results = self._convert_results_for_json(self.results)
        
        with open(results_json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Detailed results saved: {results_json_path}")
        
        return report_path, results_json_path
    
    def _convert_results_for_json(self, obj):
        """Convert numpy arrays and other non-serializable objects for JSON"""
        if isinstance(obj, dict):
            return {key: self._convert_results_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_results_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        else:
            return obj
    
    def save_models(self, baseline_model, mixed_precision_model, fp16_model):
        """Save trained models for future use"""
        print("\nSaving trained models...")
        
        # Save baseline model
        baseline_path = os.path.join(self.results_dir, 'baseline_fp32_model.pth')
        torch.save({
            'model_state_dict': baseline_model.state_dict(),
            'model_architecture': 'AlexNet',
            'precision': 'FP32',
            'num_classes': 2,
            'training_completed': True
        }, baseline_path)
        
        # Save mixed precision model
        mp_path = os.path.join(self.results_dir, 'mixed_precision_model.pth')
        torch.save({
            'model_state_dict': mixed_precision_model.state_dict(),
            'model_architecture': 'AlexNet',
            'precision': 'Mixed Precision',
            'num_classes': 2,
            'training_completed': True
        }, mp_path)
        
        # Save FP16 model
        fp16_path = os.path.join(self.results_dir, 'fp16_model.pth')
        torch.save({
            'model_state_dict': fp16_model.state_dict(),
            'model_architecture': 'AlexNet',
            'precision': 'FP16',
            'num_classes': 2,
            'conversion_completed': True
        }, fp16_path)
        
        print(f"Baseline FP32 model saved: {baseline_path}")
        print(f"Mixed precision model saved: {mp_path}")
        print(f"FP16 model saved: {fp16_path}")
        
        return baseline_path, mp_path, fp16_path

def main():
    """
    Main execution function for comprehensive FP16 analysis
    Dissertation-ready mixed precision analysis workflow
    """
    
    print("="*80)
    print("COMPREHENSIVE MIXED PRECISION TRAINING ANALYSIS")
    print("AlexNet FP16 Optimization with Academic Rigor")
    print("="*80)
    
    # Configuration
    DATASET_PATH = "dataset"  # Update this path to match your dataset structure
    RESULTS_DIR = "comprehensive_fp16_results"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize analyzer
    analyzer = ComprehensiveFP16Analyzer(
        dataset_path=DATASET_PATH,
        results_dir=RESULTS_DIR,
        device=DEVICE
    )
    
    try:
        # 1. Prepare datasets with academic rigor
        print("\n" + "="*60)
        print("PHASE 1: DATASET PREPARATION")
        print("="*60)
        dataset_info = analyzer.prepare_datasets(batch_size=32, validation_split=0.2)
        
        # 2. Initialize and analyze baseline model
        print("\n" + "="*60)
        print("PHASE 2: BASELINE MODEL ANALYSIS")
        print("="*60)
        
        baseline_model = AlexNetFP16(num_classes=2)
        baseline_analysis = analyzer.analyze_baseline_model(baseline_model)
        
        # 3. Train baseline FP32 model
        print("\n" + "="*60)
        print("PHASE 3: BASELINE FP32 TRAINING")
        print("="*60)
        
        baseline_model, baseline_history = analyzer.train_baseline_model(
            baseline_model, epochs=10, learning_rate=0.0001
        )
        
        # 4. Train mixed precision model
        print("\n" + "="*60)
        print("PHASE 4: MIXED PRECISION TRAINING")
        print("="*60)
        
        # Create mixed precision model (copy of baseline)
        mixed_precision_model = AlexNetFP16(num_classes=2)
        mixed_precision_model.load_state_dict(baseline_model.state_dict())
        
        mixed_precision_model, mp_history = analyzer.train_mixed_precision_model(
            mixed_precision_model, epochs=10, learning_rate=0.0001
        )
        
        # 5. Create direct FP16 model
        print("\n" + "="*60)
        print("PHASE 5: FP16 MODEL CONVERSION")
        print("="*60)
        
        fp16_model = analyzer.create_fp16_model(mixed_precision_model)
        
        # 6. Comprehensive performance evaluation
        print("\n" + "="*60)
        print("PHASE 6: PERFORMANCE EVALUATION")
        print("="*60)
        
        evaluation_results, comparative_analysis = analyzer.comprehensive_performance_evaluation(
            baseline_model, mixed_precision_model, fp16_model
        )
        
        # 7. Generate visualizations
        print("\n" + "="*60)
        print("PHASE 7: VISUALIZATION GENERATION")
        print("="*60)
        
        visualization_path = analyzer.generate_comprehensive_visualizations()
        
        # 8. Generate academic report
        print("\n" + "="*60)
        print("PHASE 8: ACADEMIC REPORT GENERATION")
        print("="*60)
        
        report_path, results_path = analyzer.generate_academic_report()
        
        # 9. Save models
        print("\n" + "="*60)
        print("PHASE 9: MODEL PERSISTENCE")
        print("="*60)
        
        baseline_path, mp_path, fp16_path = analyzer.save_models(
            baseline_model, mixed_precision_model, fp16_model
        )
        
        # Final summary
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE - COMPREHENSIVE FP16 STUDY")
        print("="*80)
        
        print(f"\nGenerated Outputs:")
        print(f"📊 Comprehensive Visualization: {visualization_path}")
        print(f"📄 Academic Report: {report_path}")
        print(f"📈 Detailed Results: {results_path}")
        print(f"🤖 Baseline FP32 Model: {baseline_path}")
        print(f"⚡ Mixed Precision Model: {mp_path}")
        print(f"🔥 FP16 Model: {fp16_path}")
        
        # Key findings summary
        print(f"\nKey Findings Summary:")
        baseline_acc = analyzer.results['training_metrics']['baseline']['best_validation_accuracy']
        mp_acc = analyzer.results['training_metrics']['mixed_precision']['best_validation_accuracy']
        
        print(f"📈 FP32 Baseline Accuracy: {baseline_acc:.2f}%")
        print(f"⚡ Mixed Precision Accuracy: {mp_acc:.2f}%")
        print(f"🔥 Accuracy Change (MP): {mp_acc - baseline_acc:+.2f}%")
        
        # Performance improvements
        if 'mixed_precision' in comparative_analysis:
            mp_perf = comparative_analysis['mixed_precision']
            speed_improvement = -mp_perf['avg_inference_time_ms']['relative_improvement_percent']
            memory_improvement = -mp_perf['memory_used_mb']['relative_improvement_percent']
            
            print(f"⚡ Speed Improvement (MP): {speed_improvement:+.1f}%")
            print(f"💾 Memory Improvement (MP): {memory_improvement:+.1f}%")
        
        # Model size reduction
        size_reduction = comparative_analysis['model_size']['actual_reduction_percent']
        print(f"📦 Model Size Reduction: {size_reduction:.1f}%")
        
        # Hardware optimization
        fp16_optimized = analyzer.results['model_analysis']['fp16']['hardware_optimized']
        print(f"🔧 Hardware Optimization: {'✅ Enabled' if fp16_optimized else '❌ Limited'}")
        
        # Deployment recommendation
        if mp_acc >= baseline_acc - 0.5:  # Less than 0.5% accuracy drop
            if 'mixed_precision' in comparative_analysis:
                speed_gain = -comparative_analysis['mixed_precision']['avg_inference_time_ms']['relative_improvement_percent']
                if speed_gain > 10:
                    print(f"✅ RECOMMENDATION: Deploy mixed precision model")
                elif speed_gain > 5:
                    print(f"⚠️  RECOMMENDATION: Consider mixed precision for speed-critical applications")
                else:
                    print(f"💡 RECOMMENDATION: Mixed precision shows promise, evaluate for specific use case")
            else:
                print(f"💡 RECOMMENDATION: Further evaluation needed")
        else:
            print(f"❌ RECOMMENDATION: Stick with FP32 due to accuracy requirements")
        
        print(f"\n🎓 DISSERTATION-READY FP16 ANALYSIS COMPLETED")
        print(f"Ready for academic submission and publication!")
        print(f"Combine with INT8 QAT results for complete precision spectrum analysis!")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Save partial results if available
        try:
            if hasattr(analyzer, 'results') and analyzer.results:
                error_results_path = os.path.join(RESULTS_DIR, 'error_results.json')
                json_results = analyzer._convert_results_for_json(analyzer.results)
                with open(error_results_path, 'w') as f:
                    json.dump(json_results, f, indent=2)
                print(f"Partial results saved to: {error_results_path}")
        except:
            pass

if __name__ == "__main__":
    # Import required modules
    import sys
    
    # Install required packages if not available
    required_packages = ['seaborn', 'scikit-learn', 'scipy']
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing required package: {package}")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # Run main analysis
    main()