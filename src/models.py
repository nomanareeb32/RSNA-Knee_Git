"""
Model Architectures for RSNA Knee Abnormality Detection
Optimized for RTX 4050 (6GB VRAM)
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class SliceEncoder(nn.Module):
    """
    2D CNN encoder for individual MRI slices
    Converts image (H, W) → feature vector (feature_dim,)
    """
    
    def __init__(self, backbone: str = 'resnet50', pretrained: bool = True):
        """
        Initialize encoder
        
        Args:
            backbone: 'resnet50', 'resnet34', 'efficientnet_b2'
            pretrained: Whether to use ImageNet pretrained weights
        """
        super().__init__()
        
        if backbone == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            self.model.fc = nn.Identity()
            self.feature_dim = 2048
            
        elif backbone == 'resnet34':
            self.model = models.resnet34(pretrained=pretrained)
            self.model.fc = nn.Identity()
            self.feature_dim = 512
            
        elif backbone == 'efficientnet_b2':
            self.model = models.efficientnet_b2(pretrained=pretrained)
            self.model.classifier = nn.Identity()
            self.feature_dim = 1408
            
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images to features
        
        Args:
            x: Input tensor of shape (B, H, W) or (B, 3, H, W)
        
        Returns:
            Feature tensor of shape (B, feature_dim)
        """
        # Expand grayscale to RGB if needed
        if x.dim() == 3:
            x = x.unsqueeze(1).repeat(1, 3, 1, 1)
        elif x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        
        return self.model(x)


class StudyAggregator(nn.Module):
    """
    Aggregate slice-level features to study-level predictions
    Takes (B, num_slices, feature_dim) → (B, 12)
    """
    
    def __init__(self, feature_dim: int, num_abnormalities: int = 12,
                 aggregation: str = 'max'):
        """
        Initialize aggregator
        
        Args:
            feature_dim: Dimension of input features
            num_abnormalities: Number of output classes (default 12)
            aggregation: 'max', 'mean', or 'attention'
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_abnormalities = num_abnormalities
        self.aggregation = aggregation
        
        # Aggregation
        if aggregation == 'max':
            self.pool = nn.AdaptiveMaxPool1d(1)
        elif aggregation == 'mean':
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif aggregation == 'attention':
            # Learnable attention weights
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
            self.pool = None
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_abnormalities),
            nn.Sigmoid()  # Binary classification for each abnormality
        )
    
    def forward(self, slice_features: torch.Tensor) -> torch.Tensor:
        """
        Aggregate and classify
        
        Args:
            slice_features: (B, num_slices, feature_dim)
        
        Returns:
            Predictions: (B, num_abnormalities)
        """
        # Aggregate across slices
        if self.aggregation == 'max':
            # (B, num_slices, feature_dim) → (B, feature_dim, num_slices) 
            # → (B, feature_dim, 1) → (B, feature_dim)
            study_features = self.pool(slice_features.transpose(1, 2)).squeeze(-1)
            
        elif self.aggregation == 'mean':
            # Simple average across slices
            study_features = slice_features.mean(dim=1)  # (B, feature_dim)
            
        elif self.aggregation == 'attention':
            # Apply learned attention weights
            attn_logits = self.attention(slice_features)  # (B, num_slices, 1)
            attn_weights = torch.softmax(attn_logits, dim=1)  # (B, num_slices, 1)
            study_features = (slice_features * attn_weights).sum(dim=1)  # (B, feature_dim)
        
        # Classify
        predictions = self.classifier(study_features)  # (B, num_abnormalities)
        
        return predictions


class KneeAbnormalityModel(nn.Module):
    """
    Complete model: Slice Encoder + Study Aggregator
    Processes (B, num_slices, H, W) → (B, 12)
    """
    
    def __init__(self, backbone: str = 'resnet50', pretrained: bool = True,
                 aggregation: str = 'max', num_abnormalities: int = 12):
        """
        Initialize complete model
        
        Args:
            backbone: Encoder backbone ('resnet50', 'resnet34', 'efficientnet_b2')
            pretrained: Use ImageNet pretrained weights
            aggregation: Aggregation method ('max', 'mean', 'attention')
            num_abnormalities: Number of output classes
        """
        super().__init__()
        
        self.encoder = SliceEncoder(backbone=backbone, pretrained=pretrained)
        self.aggregator = StudyAggregator(
            feature_dim=self.encoder.feature_dim,
            num_abnormalities=num_abnormalities,
            aggregation=aggregation
        )
        
        print(f"✓ Model initialized")
        print(f"  Backbone: {backbone}")
        print(f"  Aggregation: {aggregation}")
        print(f"  Feature dim: {self.encoder.feature_dim}")
        print(f"  Abnormalities: {num_abnormalities}")
    
    def forward(self, slices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            slices: (B, num_slices, H, W)
        
        Returns:
            Predictions: (B, 12)
        """
        B, N, H, W = slices.shape
        
        # Flatten batch and slice dimensions
        slices_flat = slices.reshape(B * N, H, W)
        
        # Encode each slice independently
        slice_features = self.encoder(slices_flat)  # (B*N, feature_dim)
        
        # Reshape back to (B, N, feature_dim)
        slice_features = slice_features.reshape(B, N, -1)
        
        # Aggregate to study level and classify
        predictions = self.aggregator(slice_features)  # (B, 12)
        
        return predictions
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test models
    print("Testing model architectures...\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Test different configurations
    configs = [
        {'backbone': 'resnet50', 'aggregation': 'max'},
        {'backbone': 'resnet50', 'aggregation': 'mean'},
        {'backbone': 'resnet50', 'aggregation': 'attention'},
        {'backbone': 'resnet34', 'aggregation': 'max'},
    ]
    
    for config in configs:
        print(f"\nTesting: {config['backbone']} + {config['aggregation']}")
        
        model = KneeAbnormalityModel(**config).to(device)
        num_params = model.get_num_parameters()
        print(f"  Parameters: {num_params:,}")
        
        # Test forward pass
        dummy_input = torch.randn(2, 32, 512, 512).to(device)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        # Estimate memory usage
        param_memory = (num_params * 4) / (1024**2)  # 4 bytes per float32
        print(f"  Parameter memory: {param_memory:.1f} MB")
        
        del model
    
    print(f"\n✓ Model tests passed!")
