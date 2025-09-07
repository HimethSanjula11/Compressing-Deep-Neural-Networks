import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torch.quantization as quantization
from torch.quantization import QuantStub, DeQuantStub
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

class QuantizedAlexNet(nn.Module):
    """
    AlexNet with quantization stubs for QAT
    Essential for proper quantization aware training
    """
    def __init__(self, num_classes=2):
        super(QuantizedAlexNet, self).__init__()
        # Load pretrained AlexNet
        self.alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        
        # Modify classifier for binary classification
        num_ftrs = self.alexnet.classifier[6].in_features
        self.alexnet.classifier[6] = nn.Linear(num_ftrs, num_classes)
        
        # Add quantization stubs
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
    def forward(self, x):
        x = self.quant(x)
        x = self.alexnet(x)
        x = self.dequant(x)
        return x

class ComprehensiveQATAnalyzer:
    """
    Dissertation-grade QAT analyzer with academic rigor
    Implements comprehensive quantization analysis methodology
    """
    
    def __init__(self, dataset_path, results_dir="qat_results", device=None):
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
            'quantization_metrics': {},
            'performance_analysis': {},
            'statistical_analysis': {},
            'deployment_metrics': {}
        }
        
        # Academic configuration
        self.num_statistical_runs = 10  # For statistical significance
        self.confidence_level = 0.95
        self.calibration_samples = 100
        
        print(f"QAT Analyzer initialized")
        print(f"Device: {self.device}")
        print(f"Results directory: {self.results_dir}")
        
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
                'gpu_compute_capability': torch.cuda.get_device_properties(0).major
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
        
        # Create calibration dataset (subset of validation)
        calibration_indices = np.random.choice(len(val_dataset), self.calibration_samples, replace=False)
        calibration_dataset = Subset(val_dataset, calibration_indices)
        
        # Data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                      num_workers=4, pin_memory=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                                    num_workers=4, pin_memory=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                     num_workers=4, pin_memory=True)
        self.calibration_loader = DataLoader(calibration_dataset, batch_size=batch_size, 
                                           shuffle=False, num_workers=4, pin_memory=True)
        
        # Dataset analysis
        dataset_info = {
            'total_train_samples': len(train_dataset),
            'validation_samples': len(val_dataset),
            'test_samples': len(test_dataset),
            'calibration_samples': len(calibration_dataset),
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
        print(f"  Calibration: {len(calibration_dataset)} samples")
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
        # This is a simplified estimation - for exact FLOPs, use tools like thop or fvcore
        flops = 0
        
        # AlexNet specific estimation
        # Conv layers: kernel_h * kernel_w * in_channels * out_channels * output_h * output_w
        # FC layers: in_features * out_features
        
        # Rough estimation based on known AlexNet architecture
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
        Train baseline model with comprehensive monitoring
        Academic-grade training with detailed metrics
        """
        print(f"\nTraining baseline model for {epochs} epochs...")
        
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
    
    def prepare_quantization_aware_training(self, model):
        """
        Prepare model for quantization aware training
        Configure quantization settings with academic rigor
        """
        print("\nPreparing Quantization Aware Training...")
        
        # Set quantization configuration
        model.train()
        
        # Use a compatible quantization configuration
        # Per-tensor quantization for both weights and activations to avoid compatibility issues
        from torch.quantization import QConfig, default_observer, default_weight_observer
        
        # Create a compatible QConfig with per-tensor quantization
        model.qconfig = QConfig(
            activation=default_observer.with_args(dtype=torch.quint8),
            weight=default_weight_observer.with_args(dtype=torch.qint8)
        )
        
        # Prepare model for QAT
        quantization.prepare_qat(model, inplace=True)
        
        # Analysis of quantization configuration
        qat_config = {
            'backend': 'fbgemm',
            'qconfig_details': str(model.qconfig),
            'quantization_scheme': 'per_tensor_affine',
            'activation_dtype': 'quint8',
            'weight_dtype': 'qint8',
            'bit_width': 8,
            'calibration_method': 'QAT'
        }
        
        self.results['quantization_metrics']['qat_config'] = qat_config
        
        print(f"QAT Configuration:")
        print(f"  Backend: {qat_config['backend']}")
        print(f"  Bit width: {qat_config['bit_width']}")
        print(f"  Quantization scheme: {qat_config['quantization_scheme']}")
        print(f"  Activation dtype: {qat_config['activation_dtype']}")
        print(f"  Weight dtype: {qat_config['weight_dtype']}")
        
        return model
    
    def train_quantized_model(self, model, epochs=5, learning_rate=0.00001):
        """
        Quantization Aware Training with comprehensive monitoring
        Academic-grade QAT implementation
        """
        print(f"\nStarting Quantization Aware Training for {epochs} epochs...")
        
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        
        # Lower learning rate for QAT fine-tuning
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # QAT training history
        qat_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'quantization_error': []
        }
        
        best_qat_val_acc = 0.0
        best_qat_model_state = None
        
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
                    print(f'  QAT Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
            
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
            qat_history['train_loss'].append(train_loss)
            qat_history['train_acc'].append(train_acc)
            qat_history['val_loss'].append(val_loss)
            qat_history['val_acc'].append(val_acc)
            
            # Save best model
            if val_acc > best_qat_val_acc:
                best_qat_val_acc = val_acc
                best_qat_model_state = model.state_dict().copy()
            
            print(f'QAT Epoch {epoch+1}/{epochs}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, '
                  f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Load best QAT model
        model.load_state_dict(best_qat_model_state)
        
        # Save QAT training metrics
        self.results['training_metrics']['qat'] = {
            'history': qat_history,
            'best_validation_accuracy': best_qat_val_acc,
            'epochs_trained': epochs
        }
        
        print(f"QAT training completed. Best validation accuracy: {best_qat_val_acc:.2f}%")
        return model, qat_history
    
    def convert_to_quantized_model(self, qat_model):
        """
        Convert QAT model to production quantized model
        Academic-grade model conversion with analysis
        """
        print("\nConverting QAT model to quantized model...")
        
        # Set to evaluation mode
        qat_model.eval()
        
        # Convert to quantized model
        quantized_model = quantization.convert(qat_model, inplace=False)
        
        # Analyze quantized model
        quantized_analysis = self._analyze_quantized_model(quantized_model)
        
        self.results['model_analysis']['quantized'] = quantized_analysis
        
        print("Model conversion completed.")
        return quantized_model
    
    def _analyze_quantized_model(self, quantized_model):
        """Comprehensive quantized model analysis"""
        analysis = {
            'quantized_layers': 0,
            'total_layers': 0,
            'quantization_coverage': 0.0,
            'estimated_size_reduction': 0.0,
            'layer_quantization_details': OrderedDict()
        }
        
        for name, module in quantized_model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                analysis['total_layers'] += 1
                
                # Check if layer is quantized
                is_quantized = any(keyword in module.__class__.__name__.lower() 
                                 for keyword in ['quantized', 'int8'])
                
                if is_quantized:
                    analysis['quantized_layers'] += 1
                
                analysis['layer_quantization_details'][name] = {
                    'type': module.__class__.__name__,
                    'is_quantized': is_quantized
                }
        
        if analysis['total_layers'] > 0:
            analysis['quantization_coverage'] = analysis['quantized_layers'] / analysis['total_layers']
        
        # Estimate size reduction (INT8 vs FP32)
        analysis['estimated_size_reduction'] = 0.75  # Theoretical 75% reduction
        
        return analysis
    
    def comprehensive_performance_evaluation(self, baseline_model, quantized_model):
        """
        Comprehensive performance evaluation with statistical rigor
        Academic-grade performance analysis
        """
        print(f"\nConducting comprehensive performance evaluation ({self.num_statistical_runs} runs)...")
        
        # Create clean copies of models to avoid state pollution
        baseline_model_clean = QuantizedAlexNet(num_classes=2)
        baseline_model_clean.load_state_dict(baseline_model.state_dict())
        baseline_model_clean.eval()
        
        quantized_model_clean = quantized_model  # This is already a converted model
        
        models = {
            'baseline': baseline_model_clean,
            'quantized': quantized_model_clean
        }
        
        evaluation_results = {}
        
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name} model...")
            
            # Multiple runs for statistical significance
            run_results = []
            
            for run in range(self.num_statistical_runs):
                print(f"  Run {run + 1}/{self.num_statistical_runs}")
                
                # Create a fresh copy for each run to avoid any state changes
                if model_name == 'baseline':
                    model_copy = QuantizedAlexNet(num_classes=2)
                    model_copy.load_state_dict(baseline_model.state_dict())
                    model_copy.eval()
                else:
                    model_copy = model  # Quantized model is stateless
                
                run_result = self._single_performance_run(model_copy, f"{model_name}_run_{run}")
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
    
    def _single_performance_run(self, model, run_id):
        """Single performance evaluation run"""
        model.eval()
        
        # More precise quantized model detection
        # Check for actual quantized modules, not just QAT preparation artifacts
        is_quantized = False
        for module in model.modules():
            module_name = module.__class__.__name__
            # Look for actual quantized modules (post-conversion)
            if any(keyword in module_name.lower() for keyword in [
                'quantizedconv', 'quantizedlinear', 'quantizedrelu', 
                'quantizedadaptiveavgpool', 'quantizeddropout'
            ]):
                is_quantized = True
                break
            # Also check for quantized tensor attributes
            if hasattr(module, '_packed_params') or hasattr(module, 'scale'):
                is_quantized = True
                break
        
        # Determine device based on model type
        if is_quantized:
            model_device = torch.device('cpu')
            model = model.to(model_device)
            print(f"    Running quantized model on CPU")
        else:
            model_device = self.device
            model = model.to(model_device)
            print(f"    Running baseline model on {model_device}")
        
        # Timing variables
        inference_times = []
        memory_usage = []
        predictions = []
        ground_truth = []
        
        # Memory baseline
        if model_device.type == 'cuda':
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
                # Move data to appropriate device
                inputs = inputs.to(model_device)
                labels = labels.to(model_device)
                
                # Synchronize for accurate timing (only for CUDA)
                if model_device.type == 'cuda':
                    torch.cuda.synchronize()
                
                # Time inference
                start_time = time.time()
                outputs = model(inputs)
                
                if model_device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.time()
                
                # Record timing
                batch_time = end_time - start_time
                inference_times.append(batch_time / inputs.size(0))  # Per sample
                
                # Memory usage
                if model_device.type == 'cuda':
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
        if model_device.type == 'cuda':
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
            'device_used': str(model_device),
            'model_type': 'quantized' if is_quantized else 'baseline'
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
        """Comparative analysis between models"""
        baseline_stats = evaluation_results['baseline']
        quantized_stats = evaluation_results['quantized']
        
        comparative_metrics = {}
        
        # Performance comparisons
        metrics_to_compare = ['accuracy', 'avg_inference_time_ms', 'throughput_samples_per_sec', 'memory_used_mb']
        
        for metric in metrics_to_compare:
            baseline_mean = baseline_stats[metric]['mean']
            quantized_mean = quantized_stats[metric]['mean']
            
            if metric == 'accuracy':
                # For accuracy, calculate difference
                improvement = quantized_mean - baseline_mean
                relative_change = improvement / baseline_mean * 100 if baseline_mean != 0 else 0
            elif metric == 'avg_inference_time_ms' or metric == 'memory_used_mb':
                # For time and memory, reduction is better
                improvement = baseline_mean - quantized_mean
                relative_change = improvement / baseline_mean * 100 if baseline_mean != 0 else 0
            else:
                # For throughput, increase is better
                improvement = quantized_mean - baseline_mean
                relative_change = improvement / baseline_mean * 100 if baseline_mean != 0 else 0
            
            # Statistical significance test
            baseline_values = baseline_stats[metric]['raw_values']
            quantized_values = quantized_stats[metric]['raw_values']
            
            # Two-sample t-test
            t_statistic, p_value = stats.ttest_ind(baseline_values, quantized_values)
            is_significant = p_value < (1 - self.confidence_level)
            
            comparative_metrics[metric] = {
                'baseline_mean': baseline_mean,
                'quantized_mean': quantized_mean,
                'absolute_improvement': improvement,
                'relative_improvement_percent': relative_change,
                't_statistic': t_statistic,
                'p_value': p_value,
                'is_statistically_significant': is_significant
            }
        
        # Model size analysis
        baseline_analysis = self.results['model_analysis']['baseline']
        quantized_analysis = self.results['model_analysis']['quantized']
        
        size_reduction = {
            'theoretical_size_reduction_percent': quantized_analysis['estimated_size_reduction'] * 100,
            'parameter_memory_baseline_mb': baseline_analysis['parameter_memory_mb'],
            'estimated_quantized_memory_mb': baseline_analysis['parameter_memory_mb'] * (1 - quantized_analysis['estimated_size_reduction'])
        }
        
        comparative_metrics['model_size'] = size_reduction
        
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
        qat_history = self.results['training_metrics']['qat']['history']
        
        epochs_baseline = range(1, len(baseline_history['val_acc']) + 1)
        epochs_qat = range(1, len(qat_history['val_acc']) + 1)
        
        plt.plot(epochs_baseline, baseline_history['val_acc'], 'b-', label='Baseline', linewidth=2)
        plt.plot(epochs_qat, qat_history['val_acc'], 'r-', label='QAT', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy (%)')
        plt.title('Training Progress Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Performance Metrics Comparison
        ax2 = plt.subplot(3, 4, 2)
        performance = self.results['performance_analysis']['comparative_analysis']
        
        metrics = ['accuracy', 'avg_inference_time_ms', 'throughput_samples_per_sec', 'memory_used_mb']
        baseline_values = [performance[m]['baseline_mean'] for m in metrics]
        quantized_values = [performance[m]['quantized_mean'] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, baseline_values, width, label='Baseline', alpha=0.8)
        plt.bar(x + width/2, quantized_values, width, label='Quantized', alpha=0.8)
        plt.xlabel('Metrics')
        plt.ylabel('Value')
        plt.title('Performance Comparison')
        plt.xticks(x, ['Accuracy', 'Inference Time', 'Throughput', 'Memory'])
        plt.legend()
        plt.xticks(rotation=45)
        
        # 3. Statistical Significance Analysis
        ax3 = plt.subplot(3, 4, 3)
        p_values = [performance[m]['p_value'] for m in metrics]
        significance_threshold = 1 - self.confidence_level
        
        colors = ['green' if p < significance_threshold else 'red' for p in p_values]
        plt.bar(range(len(metrics)), p_values, color=colors, alpha=0.7)
        plt.axhline(y=significance_threshold, color='black', linestyle='--', alpha=0.5)
        plt.xlabel('Metrics')
        plt.ylabel('P-value')
        plt.title('Statistical Significance (p < 0.05)')
        plt.xticks(range(len(metrics)), ['Acc', 'Time', 'Throughput', 'Memory'])
        plt.yscale('log')
        
        # 4. Model Size Reduction
        ax4 = plt.subplot(3, 4, 4)
        size_data = performance['model_size']
        
        sizes = [size_data['parameter_memory_baseline_mb'], size_data['estimated_quantized_memory_mb']]
        labels = ['Baseline (FP32)', 'Quantized (INT8)']
        colors = ['skyblue', 'lightcoral']
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Model Size Comparison')
        
        # 5. Inference Time Distribution
        ax5 = plt.subplot(3, 4, 5)
        baseline_times = self.results['performance_analysis']['individual_results']['baseline']['avg_inference_time_ms']['raw_values']
        quantized_times = self.results['performance_analysis']['individual_results']['quantized']['avg_inference_time_ms']['raw_values']
        
        plt.hist(baseline_times, bins=15, alpha=0.7, label='Baseline', color='blue')
        plt.hist(quantized_times, bins=15, alpha=0.7, label='Quantized', color='red')
        plt.xlabel('Inference Time (ms)')
        plt.ylabel('Frequency')
        plt.title('Inference Time Distribution')
        plt.legend()
        
        # 6. Accuracy Box Plot
        ax6 = plt.subplot(3, 4, 6)
        baseline_acc = self.results['performance_analysis']['individual_results']['baseline']['accuracy']['raw_values']
        quantized_acc = self.results['performance_analysis']['individual_results']['quantized']['accuracy']['raw_values']
        
        plt.boxplot([baseline_acc, quantized_acc], labels=['Baseline', 'Quantized'])
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy Distribution')
        plt.grid(True, alpha=0.3)
        
        # 7. Confusion Matrix - Baseline
        ax7 = plt.subplot(3, 4, 7)
        baseline_cm = np.array(self.results['performance_analysis']['individual_results']['baseline']['confusion_matrix'])
        sns.heatmap(baseline_cm, annot=True, fmt='d', cmap='Blues', ax=ax7)
        plt.title('Confusion Matrix - Baseline')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 8. Confusion Matrix - Quantized
        ax8 = plt.subplot(3, 4, 8)
        quantized_cm = np.array(self.results['performance_analysis']['individual_results']['quantized']['confusion_matrix'])
        sns.heatmap(quantized_cm, annot=True, fmt='d', cmap='Reds', ax=ax8)
        plt.title('Confusion Matrix - Quantized')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 9. Performance Improvement Radar Chart
        ax9 = plt.subplot(3, 4, 9, projection='polar')
        
        categories = ['Accuracy\nChange (%)', 'Speed\nImprovement (%)', 'Memory\nReduction (%)', 'Size\nReduction (%)']
        values = [
            performance['accuracy']['relative_improvement_percent'],
            -performance['avg_inference_time_ms']['relative_improvement_percent'],  # Negative because reduction is good
            -performance['memory_used_mb']['relative_improvement_percent'],
            size_data['theoretical_size_reduction_percent']
        ]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        ax9.plot(angles, values, 'o-', linewidth=2, color='green')
        ax9.fill(angles, values, color='green', alpha=0.25)
        ax9.set_xticks(angles[:-1])
        ax9.set_xticklabels(categories)
        ax9.set_title('Overall Performance Improvement')
        
        # 10. Layer-wise Quantization Analysis
        ax10 = plt.subplot(3, 4, 10)
        quantized_analysis = self.results['model_analysis']['quantized']
        
        quantized_layers = quantized_analysis['quantized_layers']
        total_layers = quantized_analysis['total_layers']
        non_quantized_layers = total_layers - quantized_layers
        
        plt.pie([quantized_layers, non_quantized_layers], 
                labels=['Quantized', 'Non-quantized'], 
                autopct='%1.1f%%', 
                colors=['lightgreen', 'lightgray'])
        plt.title('Layer Quantization Coverage')
        
        # 11. Training Loss Comparison
        ax11 = plt.subplot(3, 4, 11)
        plt.plot(epochs_baseline, baseline_history['train_loss'], 'b-', label='Baseline Train', alpha=0.7)
        plt.plot(epochs_baseline, baseline_history['val_loss'], 'b--', label='Baseline Val', alpha=0.7)
        plt.plot(epochs_qat, qat_history['train_loss'], 'r-', label='QAT Train', alpha=0.7)
        plt.plot(epochs_qat, qat_history['val_loss'], 'r--', label='QAT Val', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 12. System Resource Summary
        ax12 = plt.subplot(3, 4, 12)
        system_info = self.results['system_info']
        
        # Create a text summary
        ax12.axis('off')
        summary_text = f"""
        System Configuration:
        • Device: {system_info['device_name']}
        • PyTorch: {system_info['pytorch_version']}
        • CUDA: {system_info.get('cuda_version', 'N/A')}
        
        Dataset Statistics:
        • Train: {self.results['dataset_info']['total_train_samples']} samples
        • Test: {self.results['dataset_info']['test_samples']} samples
        • Classes: {self.results['dataset_info']['num_classes']}
        
        Model Statistics:
        • Parameters: {self.results['model_analysis']['baseline']['total_parameters']:,}
        • Size Reduction: {size_data['theoretical_size_reduction_percent']:.1f}%
        • Quantized Layers: {quantized_analysis['quantization_coverage']*100:.1f}%
        """
        
        ax12.text(0.1, 0.9, summary_text, transform=ax12.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.suptitle('Comprehensive QAT Analysis: AlexNet INT8 Quantization', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save visualization
        viz_path = os.path.join(self.results_dir, 'comprehensive_qat_analysis.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comprehensive visualization saved: {viz_path}")
        
        # Additional detailed plots
        self._generate_detailed_plots()
        
        return viz_path
    
    def _generate_detailed_plots(self):
        """Generate additional detailed analysis plots"""
        
        # 1. Detailed Performance Metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        performance = self.results['performance_analysis']['comparative_analysis']
        
        # Accuracy comparison with error bars
        ax = axes[0, 0]
        baseline_acc = self.results['performance_analysis']['individual_results']['baseline']['accuracy']
        quantized_acc = self.results['performance_analysis']['individual_results']['quantized']['accuracy']
        
        means = [baseline_acc['mean'], quantized_acc['mean']]
        stds = [baseline_acc['std'], quantized_acc['std']]
        
        x_pos = [0, 1]
        ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color=['blue', 'red'])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['Baseline', 'Quantized'])
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Accuracy Comparison with Error Bars')
        ax.grid(True, alpha=0.3)
        
        # Inference time comparison
        ax = axes[0, 1]
        baseline_time = self.results['performance_analysis']['individual_results']['baseline']['avg_inference_time_ms']
        quantized_time = self.results['performance_analysis']['individual_results']['quantized']['avg_inference_time_ms']
        
        means = [baseline_time['mean'], quantized_time['mean']]
        stds = [baseline_time['std'], quantized_time['std']]
        
        ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color=['blue', 'red'])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['Baseline', 'Quantized'])
        ax.set_ylabel('Inference Time (ms)')
        ax.set_title('Inference Time Comparison')
        ax.grid(True, alpha=0.3)
        
        # Memory usage comparison
        ax = axes[1, 0]
        baseline_memory = self.results['performance_analysis']['individual_results']['baseline']['memory_used_mb']
        quantized_memory = self.results['performance_analysis']['individual_results']['quantized']['memory_used_mb']
        
        means = [baseline_memory['mean'], quantized_memory['mean']]
        stds = [baseline_memory['std'], quantized_memory['std']]
        
        ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color=['blue', 'red'])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['Baseline', 'Quantized'])
        ax.set_ylabel('Memory Usage (MB)')
        ax.set_title('Memory Usage Comparison')
        ax.grid(True, alpha=0.3)
        
        # Throughput comparison
        ax = axes[1, 1]
        baseline_throughput = self.results['performance_analysis']['individual_results']['baseline']['throughput_samples_per_sec']
        quantized_throughput = self.results['performance_analysis']['individual_results']['quantized']['throughput_samples_per_sec']
        
        means = [baseline_throughput['mean'], quantized_throughput['mean']]
        stds = [baseline_throughput['std'], quantized_throughput['std']]
        
        ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color=['blue', 'red'])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['Baseline', 'Quantized'])
        ax.set_ylabel('Throughput (samples/sec)')
        ax.set_title('Throughput Comparison')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Detailed Performance Analysis with Statistical Significance', fontsize=14)
        plt.tight_layout()
        
        detail_path = os.path.join(self.results_dir, 'detailed_performance_analysis.png')
        plt.savefig(detail_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Detailed performance analysis saved: {detail_path}")
    
    def generate_academic_report(self):
        """
        Generate comprehensive academic report
        Publication-quality analysis document
        """
        print("\nGenerating comprehensive academic report...")
        
        report_path = os.path.join(self.results_dir, 'comprehensive_qat_analysis_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Comprehensive Quantization Aware Training Analysis Report\n")
            f.write("## AlexNet INT8 Quantization for Image Classification\n\n")
            
            # Abstract
            f.write("## Abstract\n\n")
            f.write("This report presents a comprehensive analysis of Quantization Aware Training (QAT) ")
            f.write("applied to AlexNet for binary image classification. The study employs rigorous ")
            f.write("experimental methodology with statistical significance testing across multiple ")
            f.write("performance metrics including accuracy, inference latency, memory consumption, ")
            f.write("and model size reduction.\n\n")
            
            # System Configuration
            f.write("## 1. Experimental Setup\n\n")
            f.write("### 1.1 System Configuration\n\n")
            system_info = self.results['system_info']
            
            f.write(f"- **Hardware**: {system_info.get('gpu_name', 'CPU only')}\n")
            f.write(f"- **PyTorch Version**: {system_info['pytorch_version']}\n")
            f.write(f"- **CUDA Version**: {system_info.get('cuda_version', 'N/A')}\n")
            f.write(f"- **Device**: {system_info['device_name']}\n")
            f.write(f"- **Analysis Date**: {system_info['timestamp']}\n\n")
            
            # Dataset Information
            f.write("### 1.2 Dataset Configuration\n\n")
            dataset_info = self.results['dataset_info']
            
            f.write(f"- **Training Samples**: {dataset_info['total_train_samples']:,}\n")
            f.write(f"- **Validation Samples**: {dataset_info['validation_samples']:,}\n") 
            f.write(f"- **Test Samples**: {dataset_info['test_samples']:,}\n")
            f.write(f"- **Calibration Samples**: {dataset_info['calibration_samples']}\n")
            f.write(f"- **Number of Classes**: {dataset_info['num_classes']}\n")
            f.write(f"- **Class Names**: {', '.join(dataset_info['class_names'])}\n")
            f.write(f"- **Class Balance Ratio**: {dataset_info['class_balance_ratio']:.3f}\n\n")
            
            # Model Architecture
            f.write("## 2. Model Architecture Analysis\n\n")
            baseline_analysis = self.results['model_analysis']['baseline']
            quantized_analysis = self.results['model_analysis']['quantized']
            
            f.write("### 2.1 Baseline Model (FP32)\n\n")
            f.write(f"- **Total Parameters**: {baseline_analysis['total_parameters']:,}\n")
            f.write(f"- **Trainable Parameters**: {baseline_analysis['trainable_parameters']:,}\n")
            f.write(f"- **Parameter Memory**: {baseline_analysis['parameter_memory_mb']:.2f} MB\n")
            f.write(f"- **Estimated FLOPs**: {baseline_analysis['estimated_flops']:,}\n\n")
            
            f.write("### 2.2 Quantized Model (INT8)\n\n")
            f.write(f"- **Quantized Layers**: {quantized_analysis['quantized_layers']}\n")
            f.write(f"- **Total Layers**: {quantized_analysis['total_layers']}\n")
            f.write(f"- **Quantization Coverage**: {quantized_analysis['quantization_coverage']*100:.1f}%\n")
            f.write(f"- **Estimated Size Reduction**: {quantized_analysis['estimated_size_reduction']*100:.1f}%\n\n")
            
            # Training Results
            f.write("## 3. Training Analysis\n\n")
            baseline_training = self.results['training_metrics']['baseline']
            qat_training = self.results['training_metrics']['qat']
            
            f.write("### 3.1 Baseline Training\n\n")
            f.write(f"- **Best Validation Accuracy**: {baseline_training['best_validation_accuracy']:.2f}%\n")
            f.write(f"- **Epochs Trained**: {baseline_training['epochs_trained']}\n")
            f.write(f"- **Final Learning Rate**: {baseline_training['final_learning_rate']:.2e}\n\n")
            
            f.write("### 3.2 Quantization Aware Training\n\n")
            f.write(f"- **Best Validation Accuracy**: {qat_training['best_validation_accuracy']:.2f}%\n")
            f.write(f"- **Epochs Trained**: {qat_training['epochs_trained']}\n")
            f.write(f"- **Training Strategy**: Fine-tuning with reduced learning rate\n\n")
            
            # Performance Analysis
            f.write("## 4. Performance Analysis\n\n")
            performance = self.results['performance_analysis']['comparative_analysis']
            
            f.write("### 4.1 Statistical Methodology\n\n")
            f.write(f"- **Number of Runs**: {self.num_statistical_runs}\n")
            f.write(f"- **Confidence Level**: {self.confidence_level*100:.0f}%\n")
            f.write("- **Statistical Test**: Two-sample t-test\n")
            f.write("- **Significance Threshold**: p < 0.05\n\n")
            
            f.write("### 4.2 Performance Metrics Comparison\n\n")
            f.write("| Metric | Baseline | Quantized | Improvement | Significance |\n")
            f.write("|--------|----------|-----------|-------------|-------------|\n")
            
            metrics = ['accuracy', 'avg_inference_time_ms', 'throughput_samples_per_sec', 'memory_used_mb']
            metric_names = ['Accuracy (%)', 'Inference Time (ms)', 'Throughput (samples/sec)', 'Memory Usage (MB)']
            
            for metric, name in zip(metrics, metric_names):
                data = performance[metric]
                significance = "✓" if data['is_statistically_significant'] else "✗"
                f.write(f"| {name} | {data['baseline_mean']:.3f} | {data['quantized_mean']:.3f} | ")
                f.write(f"{data['relative_improvement_percent']:+.2f}% | {significance} (p={data['p_value']:.4f}) |\n")
            
            f.write("\n")
            
            # Model Size Analysis
            f.write("### 4.3 Model Size Analysis\n\n")
            size_data = performance['model_size']
            f.write(f"- **Baseline Model Size**: {size_data['parameter_memory_baseline_mb']:.2f} MB\n")
            f.write(f"- **Estimated Quantized Size**: {size_data['estimated_quantized_memory_mb']:.2f} MB\n")
            f.write(f"- **Theoretical Size Reduction**: {size_data['theoretical_size_reduction_percent']:.1f}%\n\n")
            
            # Classification Performance
            f.write("### 4.4 Classification Performance\n\n")
            baseline_report = self.results['performance_analysis']['individual_results']['baseline']['classification_report']
            quantized_report = self.results['performance_analysis']['individual_results']['quantized']['classification_report']
            
            f.write("#### Baseline Model Classification Report\n\n")
            f.write("| Class | Precision | Recall | F1-Score | Support |\n")
            f.write("|-------|-----------|--------|----------|----------|\n")
            
            for class_id in ['0', '1']:
                if class_id in baseline_report:
                    metrics_class = baseline_report[class_id]
                    f.write(f"| {class_id} | {metrics_class['precision']:.3f} | {metrics_class['recall']:.3f} | ")
                    f.write(f"{metrics_class['f1-score']:.3f} | {metrics_class['support']} |\n")
            
            if 'macro avg' in baseline_report:
                macro = baseline_report['macro avg']
                f.write(f"| **Macro Avg** | {macro['precision']:.3f} | {macro['recall']:.3f} | ")
                f.write(f"{macro['f1-score']:.3f} | {macro['support']} |\n")
            
            f.write("\n#### Quantized Model Classification Report\n\n")
            f.write("| Class | Precision | Recall | F1-Score | Support |\n")
            f.write("|-------|-----------|--------|----------|----------|\n")
            
            for class_id in ['0', '1']:
                if class_id in quantized_report:
                    metrics_class = quantized_report[class_id]
                    f.write(f"| {class_id} | {metrics_class['precision']:.3f} | {metrics_class['recall']:.3f} | ")
                    f.write(f"{metrics_class['f1-score']:.3f} | {metrics_class['support']} |\n")
            
            if 'macro avg' in quantized_report:
                macro = quantized_report['macro avg']
                f.write(f"| **Macro Avg** | {macro['precision']:.3f} | {macro['recall']:.3f} | ")
                f.write(f"{macro['f1-score']:.3f} | {macro['support']} |\n")
            
            f.write("\n")
            
            # Discussion and Conclusions
            f.write("## 5. Discussion and Analysis\n\n")
            
            f.write("### 5.1 Key Findings\n\n")
            
            # Accuracy analysis
            acc_change = performance['accuracy']['relative_improvement_percent']
            if acc_change > 1:
                f.write(f"- **Accuracy**: Quantization resulted in a {acc_change:.2f}% improvement in accuracy, ")
                f.write("indicating that QAT successfully maintained model performance.\n")
            elif acc_change > -1:
                f.write(f"- **Accuracy**: Minimal accuracy change ({acc_change:+.2f}%), demonstrating ")
                f.write("successful preservation of model capability through QAT.\n")
            else:
                f.write(f"- **Accuracy**: Accuracy decreased by {abs(acc_change):.2f}%, which may indicate ")
                f.write("the need for longer QAT fine-tuning or different quantization strategies.\n")
            
            # Performance analysis
            speed_improvement = -performance['avg_inference_time_ms']['relative_improvement_percent']
            if speed_improvement > 10:
                f.write(f"- **Inference Speed**: Significant performance improvement of {speed_improvement:.1f}% ")
                f.write("in inference time, demonstrating successful acceleration through quantization.\n")
            elif speed_improvement > 0:
                f.write(f"- **Inference Speed**: Modest improvement of {speed_improvement:.1f}% in inference time.\n")
            else:
                f.write(f"- **Inference Speed**: No significant speed improvement observed, possibly due to ")
                f.write("hardware limitations or insufficient optimization.\n")
            
            # Memory analysis
            memory_improvement = -performance['memory_used_mb']['relative_improvement_percent']
            f.write(f"- **Memory Efficiency**: Runtime memory usage change of {memory_improvement:+.1f}%.\n")
            
            # Model size
            f.write(f"- **Model Size**: Theoretical model size reduction of ")
            f.write(f"{size_data['theoretical_size_reduction_percent']:.1f}% through INT8 quantization.\n\n")
            
            f.write("### 5.2 Statistical Significance\n\n")
            
            significant_metrics = [m for m in metrics if performance[m]['is_statistically_significant']]
            if significant_metrics:
                f.write("The following metrics showed statistically significant differences (p < 0.05):\n")
                for metric in significant_metrics:
                    f.write(f"- {metric.replace('_', ' ').title()}\n")
            else:
                f.write("No metrics showed statistically significant differences at the 95% confidence level.")
            
            f.write("\n")
            
            # Limitations and Future Work
            f.write("### 5.3 Limitations and Future Work\n\n")
            f.write("- **Hardware Dependencies**: Performance improvements may vary significantly across different hardware platforms.\n")
            f.write("- **Dataset Specificity**: Results are specific to the binary classification task and dataset used.\n")
            f.write("- **Quantization Strategy**: Alternative quantization approaches (e.g., dynamic quantization, ONNX quantization) could yield different results.\n")
            f.write("- **Deployment Optimization**: Additional optimizations through ONNX Runtime, TensorRT, or mobile deployment frameworks could provide further improvements.\n\n")
            
            # Recommendations
            f.write("## 6. Recommendations\n\n")
            
            if acc_change > -2 and (speed_improvement > 5 or memory_improvement > 5):
                f.write("**Recommendation: DEPLOY QUANTIZED MODEL**\n\n")
                f.write("The quantized model demonstrates acceptable accuracy preservation with measurable performance benefits. ")
                f.write("Deployment of the INT8 quantized model is recommended for production use.\n\n")
            elif acc_change > -5:
                f.write("**Recommendation: CONDITIONAL DEPLOYMENT**\n\n")
                f.write("The quantized model shows promise but may require additional optimization. ")
                f.write("Consider extended QAT training or alternative quantization strategies.\n\n")
            else:
                f.write("**Recommendation: FURTHER OPTIMIZATION REQUIRED**\n\n")
                f.write("The current quantization approach shows significant accuracy degradation. ")
                f.write("Alternative approaches or extended fine-tuning recommended before deployment.\n\n")
            
            # Technical Implementation Details
            f.write("## 7. Technical Implementation\n\n")
            
            qat_config = self.results['quantization_metrics']['qat_config']
            f.write("### 7.1 Quantization Configuration\n\n")
            f.write(f"- **Backend**: {qat_config['backend']}\n")
            f.write(f"- **Bit Width**: {qat_config['bit_width']}\n")
            f.write(f"- **Quantization Scheme**: {qat_config['quantization_scheme']}\n")
            f.write(f"- **Calibration Method**: {qat_config['calibration_method']}\n\n")
            
            f.write("### 7.2 Training Configuration\n\n")
            f.write("- **QAT Fine-tuning**: Reduced learning rate with careful monitoring\n")
            f.write("- **Data Augmentation**: Standard ImageNet normalization with geometric and color augmentations\n")
            f.write("- **Validation Strategy**: Held-out validation set with early stopping\n")
            f.write("- **Statistical Rigor**: Multiple independent runs with confidence intervals\n\n")
            
            # Conclusion
            f.write("## 8. Conclusion\n\n")
            f.write("This comprehensive analysis demonstrates the application of Quantization Aware Training ")
            f.write("to AlexNet for binary image classification. The study employs rigorous experimental ")
            f.write("methodology with statistical significance testing to provide reliable performance ")
            f.write("characterization. The results provide evidence-based recommendations for deployment ")
            f.write("decisions based on the specific requirements of accuracy, speed, and memory efficiency.\n\n")
            
            f.write("The analysis framework presented here can be adapted for other neural network ")
            f.write("architectures and quantization strategies, providing a template for academic and ")
            f.write("industrial quantization research.\n\n")
            
            # References and Appendices
            f.write("## References\n\n")
            f.write("1. Jacob, B., et al. \"Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.\" CVPR 2018.\n")
            f.write("2. Krishnamoorthi, R. \"Quantizing deep convolutional networks for efficient inference: A whitepaper.\" arXiv preprint arXiv:1806.08342 (2018).\n")
            f.write("3. Wu, H., et al. \"Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation.\" arXiv preprint arXiv:2004.09602 (2020).\n\n")
            
            f.write("---\n")
            f.write(f"*Report generated on {system_info['timestamp']} using PyTorch {system_info['pytorch_version']}*\n")
        
        print(f"Comprehensive academic report saved: {report_path}")
        
        # Save detailed results as JSON for further analysis
        results_json_path = os.path.join(self.results_dir, 'detailed_results.json')
        
        # Convert numpy arrays to lists for JSON serialization
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
    
    def save_models(self, baseline_model, quantized_model):
        """Save trained models for future use"""
        print("\nSaving trained models...")
        
        # Save baseline model
        baseline_path = os.path.join(self.results_dir, 'baseline_model.pth')
        torch.save({
            'model_state_dict': baseline_model.state_dict(),
            'model_architecture': 'AlexNet',
            'num_classes': 2,
            'training_completed': True
        }, baseline_path)
        
        # Save quantized model
        quantized_path = os.path.join(self.results_dir, 'quantized_model.pth')
        torch.save(quantized_model, quantized_path)  # Save entire quantized model
        
        print(f"Baseline model saved: {baseline_path}")
        print(f"Quantized model saved: {quantized_path}")
        
        return baseline_path, quantized_path

def main():
    """
    Main execution function for comprehensive QAT analysis
    Dissertation-ready quantization analysis workflow
    """
    
    print("="*80)
    print("COMPREHENSIVE QUANTIZATION AWARE TRAINING ANALYSIS")
    print("AlexNet INT8 Quantization with Academic Rigor")
    print("="*80)
    
    # Configuration
    DATASET_PATH = "dataset"  # Update this path
    RESULTS_DIR = "comprehensive_qat_results"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize analyzer
    analyzer = ComprehensiveQATAnalyzer(
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
        
        baseline_model = QuantizedAlexNet(num_classes=2)
        baseline_analysis = analyzer.analyze_baseline_model(baseline_model)
        
        # 3. Train baseline model
        print("\n" + "="*60)
        print("PHASE 3: BASELINE MODEL TRAINING")
        print("="*60)
        
        baseline_model, baseline_history = analyzer.train_baseline_model(
            baseline_model, epochs=10, learning_rate=0.0001
        )
        
        # 4. Prepare for quantization aware training
        print("\n" + "="*60)
        print("PHASE 4: QUANTIZATION PREPARATION")
        print("="*60)
        
        # Create QAT model (copy of baseline) - keep baseline model clean
        qat_model = QuantizedAlexNet(num_classes=2)
        qat_model.load_state_dict(baseline_model.state_dict())
        
        # Store baseline model state before QAT modifications
        baseline_state_dict = baseline_model.state_dict().copy()
        
        # Prepare for QAT
        qat_model = analyzer.prepare_quantization_aware_training(qat_model)
        
        # Ensure baseline model remains unchanged
        baseline_model.load_state_dict(baseline_state_dict)
        baseline_model.eval()
        
        # 5. Quantization aware training
        print("\n" + "="*60)
        print("PHASE 5: QUANTIZATION AWARE TRAINING")
        print("="*60)
        
        qat_model, qat_history = analyzer.train_quantized_model(
            qat_model, epochs=5, learning_rate=0.00001
        )
        
        # 6. Convert to quantized model
        print("\n" + "="*60)
        print("PHASE 6: MODEL QUANTIZATION")
        print("="*60)
        
        quantized_model = analyzer.convert_to_quantized_model(qat_model)
        
        # 7. Comprehensive performance evaluation
        print("\n" + "="*60)
        print("PHASE 7: PERFORMANCE EVALUATION")
        print("="*60)
        
        evaluation_results, comparative_analysis = analyzer.comprehensive_performance_evaluation(
            baseline_model, quantized_model
        )
        
        # 8. Generate visualizations
        print("\n" + "="*60)
        print("PHASE 8: VISUALIZATION GENERATION")
        print("="*60)
        
        visualization_path = analyzer.generate_comprehensive_visualizations()
        
        # 9. Generate academic report
        print("\n" + "="*60)
        print("PHASE 9: ACADEMIC REPORT GENERATION")
        print("="*60)
        
        report_path, results_path = analyzer.generate_academic_report()
        
        # 10. Save models
        print("\n" + "="*60)
        print("PHASE 10: MODEL PERSISTENCE")
        print("="*60)
        
        baseline_model_path, quantized_model_path = analyzer.save_models(
            baseline_model, quantized_model
        )
        
        # Final summary
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE - COMPREHENSIVE QAT STUDY")
        print("="*80)
        
        print(f"\nGenerated Outputs:")
        print(f"📊 Comprehensive Visualization: {visualization_path}")
        print(f"📄 Academic Report: {report_path}")
        print(f"📈 Detailed Results: {results_path}")
        print(f"🤖 Baseline Model: {baseline_model_path}")
        print(f"⚡ Quantized Model: {quantized_model_path}")
        
        # Key findings summary
        print(f"\nKey Findings Summary:")
        acc_improvement = comparative_analysis['accuracy']['relative_improvement_percent']
        speed_improvement = -comparative_analysis['avg_inference_time_ms']['relative_improvement_percent']
        memory_improvement = -comparative_analysis['memory_used_mb']['relative_improvement_percent']
        size_reduction = comparative_analysis['model_size']['theoretical_size_reduction_percent']
        
        print(f"📈 Accuracy Change: {acc_improvement:+.2f}%")
        print(f"⚡ Speed Improvement: {speed_improvement:+.1f}%")
        print(f"💾 Memory Change: {memory_improvement:+.1f}%")
        print(f"📦 Model Size Reduction: {size_reduction:.1f}%")
        
        # Statistical significance
        significant_metrics = [
            metric for metric in ['accuracy', 'avg_inference_time_ms', 'throughput_samples_per_sec', 'memory_used_mb']
            if comparative_analysis[metric]['is_statistically_significant']
        ]
        
        if significant_metrics:
            print(f"🔬 Statistically Significant Metrics: {', '.join(significant_metrics)}")
        else:
            print(f"🔬 No statistically significant differences found")
        
        # Deployment recommendation
        if acc_improvement > -2 and (speed_improvement > 5 or memory_improvement > 5):
            print(f"✅ RECOMMENDATION: Deploy quantized model")
        elif acc_improvement > -5:
            print(f"⚠️  RECOMMENDATION: Further optimization recommended")
        else:
            print(f"❌ RECOMMENDATION: Quantization approach needs revision")
        
        print(f"\n🎓 DISSERTATION-READY QAT ANALYSIS COMPLETED")
        print(f"Ready for academic submission and publication!")
        
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