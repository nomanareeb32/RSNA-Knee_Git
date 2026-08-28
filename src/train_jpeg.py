"""
Training Loop for RSNA Knee Abnormality Detection
Optimized for RTX 4050
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import time


class Trainer:
    """Training wrapper for knee abnormality model"""
    
    def __init__(self, model, device, config: dict):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model
            device: torch.device
            config: Dictionary with training config
        """
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('num_epochs', 30),
            eta_min=config.get('min_lr', 1e-6)
        )
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Tracking
        self.best_auc = 0.0
        self.best_epoch = 0
        self.train_losses = []
        self.val_aucs = []
        self.val_losses = []
        
        # Checkpoints directory
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        print(f"✓ Trainer initialized")
        print(f"  Learning rate: {config.get('learning_rate', 1e-4)}")
        print(f"  Batch size: {config.get('batch_size', 4)}")
        print(f"  Epochs: {config.get('num_epochs', 30)}")
        print(f"  Checkpoint dir: {self.checkpoint_dir}")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc='Training', leave=False)
        
        for batch in pbar:
            slices = batch['slices'].to(self.device)  # (B, num_slices, H, W)
            labels = batch['labels'].to(self.device)  # (B, 12)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(slices)  # (B, 12)
            
            # Loss
            loss = self.criterion(predictions, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> tuple:
        """
        Validate on validation set
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Tuple of (mean_auc, val_loss)
        """
        self.model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(val_loader, desc='Validating', leave=False)
        
        with torch.no_grad():
            for batch in pbar:
                slices = batch['slices'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                predictions = self.model(slices)
                
                loss = self.criterion(predictions, labels)
                total_loss += loss.item()
                num_batches += 1
                
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Compute AUC-ROC
        predictions = np.concatenate(all_predictions, axis=0)  # (N, 12)
        labels = np.concatenate(all_labels, axis=0)  # (N, 12)
        
        auc_scores = []
        label_names = [
            'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
            'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
            'Synovitis', "Baker's", 'Contusion', 'Fracture'
        ]
        
        for i, label_name in enumerate(label_names):
            # Only compute AUC where labels are not 0.5 (unlabeled)
            valid_idx = labels[:, i] != 0.5
            
            if valid_idx.sum() > 1:
                try:
                    auc = roc_auc_score(labels[valid_idx, i], predictions[valid_idx, i])
                    auc_scores.append(auc)
                except:
                    auc_scores.append(0.5)
        
        mean_auc = np.mean(auc_scores) if auc_scores else 0.0
        avg_loss = total_loss / max(num_batches, 1)
        
        return mean_auc, avg_loss
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            num_epochs: int, patience: int = 5) -> dict:
        """
        Full training loop
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            num_epochs: Number of epochs to train
            patience: Early stopping patience
            
        Returns:
            Dictionary with training history
        """
        print(f"\n{'='*70}")
        print("STARTING TRAINING")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_auc, val_loss = self.validate(val_loader)
            
            # Schedule
            self.scheduler.step()
            
            # Track
            self.train_losses.append(train_loss)
            self.val_aucs.append(val_auc)
            self.val_losses.append(val_loss)
            
            epoch_time = time.time() - epoch_start
            
            # Print
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Time: {epoch_time:5.1f}s | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val AUC: {val_auc:.4f}")
            
            # Save best model
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.best_epoch = epoch
                
                checkpoint_path = self.checkpoint_dir / f'best_model_auc_{val_auc:.4f}.pt'
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"  ✓ Saved best model (AUC: {val_auc:.4f})")
            
            # Early stopping
            if epoch - self.best_epoch > patience:
                print(f"\n⚠ Early stopping at epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        
        # Print summary
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Best AUC: {self.best_auc:.4f} (Epoch {self.best_epoch+1})")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Average time per epoch: {total_time/(epoch+1):.1f} seconds")
        print(f"{'='*70}\n")
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_aucs': self.val_aucs,
            'best_auc': float(self.best_auc),
            'best_epoch': int(self.best_epoch),
            'total_time_seconds': total_time
        }
        
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        return history
    
    def load_best_model(self):
        """Load best checkpoint"""
        best_files = list(self.checkpoint_dir.glob('best_model_auc_*.pt'))
        if best_files:
            latest = max(best_files, key=lambda p: p.stat().st_mtime)
            self.model.load_state_dict(torch.load(latest, map_location=self.device))
            print(f"✓ Loaded best model: {latest.name}")
        else:
            print("⚠ No checkpoint found")


if __name__ == "__main__":
    print("Training script - import this module and use Trainer class")