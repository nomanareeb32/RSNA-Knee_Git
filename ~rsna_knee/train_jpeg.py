"""
Training Script for RSNA Knee Abnormality Detection
Complete end-to-end training - just run: python train_jpeg.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import time
import sys

sys.path.insert(0, 'src')

from image_loader import ImageLoader
from models import KneeAbnormalityModel


class KneeDataset(Dataset):
    """PyTorch Dataset for knee abnormality detection"""
    
    def __init__(self, study_uids, loader, mode='train', max_images=64, augment=False):
        self.study_uids = study_uids
        self.loader = loader
        self.mode = mode
        self.max_images = max_images
        self.augment = augment
        
        self.label_cols = [
            'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
            'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
            'Synovitis', "Baker's", 'Contusion', 'Fracture'
        ]
    
    def __len__(self):
        return len(self.study_uids)
    
    def __getitem__(self, idx):
        study_uid = self.study_uids[idx]
        
        # Load images
        images, _ = self.loader.load_study(study_uid, mode=self.mode, max_images=None)
        
        if images is None:
            images = np.zeros((self.max_images, 224, 224))
            labels = torch.tensor([0.5] * 12, dtype=torch.float32)
            return {
                'study_uid': study_uid,
                'images': torch.from_numpy(images).float(),
                'labels': labels,
                'num_images': 0
            }
        
        num_actual = len(images)
        
        # Pad or truncate
        if len(images) < self.max_images:
            pad_size = self.max_images - len(images)
            images = np.pad(images, ((0, pad_size), (0, 0), (0, 0)), mode='constant')
        else:
            idx_sample = np.linspace(0, len(images)-1, self.max_images, dtype=int)
            images = images[idx_sample]
        
        images_tensor = torch.from_numpy(images).float()
        
        # Get labels
        labels_dict = self.loader.get_labels(study_uid)
        labels = torch.tensor([labels_dict.get(col, 0.5) for col in self.label_cols], dtype=torch.float32)
        
        return {
            'study_uid': study_uid,
            'images': images_tensor,
            'labels': labels,
            'num_images': num_actual
        }


def collate_batch(batch):
    """Collate function for DataLoader"""
    study_uids = [item['study_uid'] for item in batch]
    images = torch.stack([item['images'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    num_images = torch.tensor([item['num_images'] for item in batch])
    
    return {
        'study_uids': study_uids,
        'images': images,
        'labels': labels,
        'num_images': num_images
    }


class Trainer:
    """Training wrapper"""
    
    def __init__(self, model, device, config):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_epochs'],
            eta_min=config['min_lr']
        )
        
        self.criterion = nn.BCELoss()
        
        self.best_auc = 0.0
        self.best_epoch = 0
        self.train_losses = []
        self.val_aucs = []
        self.val_losses = []
        
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        print(f"✓ Trainer initialized")
        print(f"  Learning rate: {config['learning_rate']}")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Epochs: {config['num_epochs']}")
    
    def train_epoch(self, train_loader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc='Training', leave=False)
        
        for batch in pbar:
            images = batch['images'].to(self.device)  # (B, N, H, W, 3)
            labels = batch['labels'].to(self.device)  # (B, 12)
            
            self.optimizer.zero_grad()
            predictions = self.model(images)  # Model handles all dimensions
            loss = self.criterion(predictions, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / max( num_batches, 1)
        
    def validate(self, val_loader):
        """Validate"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(val_loader, desc='Validating', leave=False)
        
        with torch.no_grad():
            for batch in pbar:
                images = batch['images'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                predictions = self.model(images)
                loss = self.criterion(predictions, labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Compute AUC
        predictions = np.concatenate(all_predictions, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        auc_scores = []
        for i in range(12):
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
    
    def fit(self, train_loader, val_loader, num_epochs, patience):
        """Full training loop"""
        print(f"\n{'='*70}")
        print("STARTING TRAINING")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            train_loss = self.train_epoch(train_loader)
            val_auc, val_loss = self.validate(val_loader)
            
            self.scheduler.step()
            
            self.train_losses.append(train_loss)
            self.val_aucs.append(val_auc)
            self.val_losses.append(val_loss)
            
            epoch_time = time.time() - epoch_start
            
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Time: {epoch_time:5.1f}s | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val AUC: {val_auc:.4f}")
            
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.best_epoch = epoch
                
                checkpoint_path = self.checkpoint_dir / f'best_model_auc_{val_auc:.4f}.pt'
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"  ✓ Saved best model (AUC: {val_auc:.4f})")
            
            if epoch - self.best_epoch > patience:
                print(f"\n⚠ Early stopping at epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Best AUC: {self.best_auc:.4f} (Epoch {self.best_epoch+1})")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Average time per epoch: {total_time/(epoch+1):.1f} seconds")
        print(f"{'='*70}\n")
        
        # Save history
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


def main():
    """Main training pipeline"""
    
    print("\n" + "="*70)
    print("RSNA KNEE ABNORMALITY DETECTION - TRAINING")
    print("="*70 + "\n")
    
    # Config (optimized for RTX 4050 6GB VRAM)
    config = {
        'batch_size': 2,          # Reduced from 4 to save VRAM
        'num_epochs': 30,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'min_lr': 1e-6,
        'patience': 5,
        'checkpoint_dir': 'checkpoints',
    }
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load data
    print("Loading dataset...")
    loader = ImageLoader('data/')
    
    if loader.train_csv is None:
        print("❌ train.csv not found!")
        return
    
    study_uids = loader.train_csv['StudyInstanceUID'].values
    print(f"Found {len(study_uids)} studies\n")
    
    # Create dataset (reduced max_images to fit 6GB VRAM)
    dataset = KneeDataset(study_uids, loader, mode='train', max_images=32, augment=True)
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_batch,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_batch,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}\n")
    
    # Model
    print("Creating model...")
    model = KneeAbnormalityModel(
        backbone='resnet50',
        pretrained=True,
        aggregation='max',
        num_abnormalities=12
    )
    
    print(f"Model parameters: {model.get_num_parameters():,}\n")
    
    # Train
    trainer = Trainer(model, device, config)
    trainer.fit(
        train_loader,
        val_loader,
        num_epochs=config['num_epochs'],
        patience=config['patience']
    )
    
    print(f"\n✓ Training complete! Best AUC: {trainer.best_auc:.4f}")


if __name__ == "__main__":
    main()