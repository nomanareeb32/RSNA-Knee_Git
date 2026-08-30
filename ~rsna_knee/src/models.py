"""
Model architectures for RSNA Knee Abnormality Detection
Handles (B, N, H, W, 3) RGB input from JPE grayscale-to-RGB conversion
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50Encoder(nn.Module):
    """ResNet50 encoder for slice feature extraction"""
    
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet50(weights='DEFAULT' if pretrained else None)
        
        # Remove classification head, keep only feature extraction
        self.model = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 2048
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) - RGB images
        Returns:
            features: (B, 2048)
        """
        x = self.model(x)
        x = x.reshape(x.size(0), -1)
        return x
    
    def get_feature_dim(self):
        return self.feature_dim


class MaxAggregator(nn.Module):
    """Max pooling aggregation across slices"""
    
    def __init__(self, feature_dim, num_abnormalities=12):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_abnormalities),
            nn.Sigmoid()
        )
    
    def forward(self, slice_features):
        """
        Args:
            slice_features: (B, N, feature_dim)
        Returns:
            predictions: (B, num_abnormalities)
        """
        # Max pool over slice dimension
        aggregated, _ = torch.max(slice_features, dim=1)  # (B, feature_dim)
        predictions = self.fc(aggregated)
        return predictions


class MeanAggregator(nn.Module):
    """Mean pooling aggregation across slices"""
    
    def __init__(self, feature_dim, num_abnormalities=12):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_abnormalities),
            nn.Sigmoid()
        )
    
    def forward(self, slice_features):
        """
        Args:
            slice_features: (B, N, feature_dim)
        Returns:
            predictions: (B, num_abnormalities)
        """
        # Mean pool over slice dimension
        aggregated = torch.mean(slice_features, dim=1)  # (B, feature_dim)
        predictions = self.fc(aggregated)
        return predictions


class KneeAbnormalityModel(nn.Module):
    """
    Complete model for knee abnormality detection
    Input: (B, N, H, W, 3) - Multiple 3-channel RGB slices per study
    Output: (B, 12) - Predictions for 12 abnormalities per study
    """
    
    def __init__(self, backbone='resnet50', pretrained=True, 
                 aggregation='max', num_abnormalities=12):
        super().__init__()
        
        self.backbone = backbone
        self.aggregation = aggregation
        self.num_abnormalities = num_abnormalities
        
        # Initialize encoder
        if backbone == 'resnet50':
            self.encoder = ResNet50Encoder(pretrained=pretrained)
        else:
            raise ValueError(f"Backbone {backbone} not supported")
        
        # Initialize aggregator
        feature_dim = self.encoder.get_feature_dim()
        
        if aggregation == 'max':
            self.aggregator = MaxAggregator(feature_dim, num_abnormalities)
        elif aggregation == 'mean':
            self.aggregator = MeanAggregator(feature_dim, num_abnormalities)
        else:
            raise ValueError(f"Aggregation {aggregation} not supported")
        
        print(f"✓ Model initialized")
        print(f"  Backbone: {backbone}")
        print(f"  Aggregation: {aggregation}")
        print(f"  Feature dim: {feature_dim}")
        print(f"  Abnormalities: {num_abnormalities}")
    
    def forward(self, slices):
        """
        Forward pass
        
        Args:
            slices: (B, N, H, W, 3) - Batch of studies with N slices each (RGB)
        
        Returns:
            predictions: (B, 12) - Abnormality predictions
        """
        # Input shape validation
        if len(slices.shape) != 5:
            raise ValueError(f"Expected 5D input (B, N, H, W, 3), got shape {slices.shape}")
        
        B, N, H, W, C = slices.shape
        
        if C != 3:
            raise ValueError(f"Expected 3 channels, got {C}")
        
        # Reshape (B, N, H, W, 3) -> (B*N, 3, H, W) for ResNet
        # First permute to move channel to position 1: (B, N, 3, H, W)
        slices = slices.permute(0, 1, 4, 2, 3)
        # Then reshape to flatten batch and slice dims: (B*N, 3, H, W)
        slices_flat = slices.reshape(B * N, C, H, W)
        
        # Encode each slice independently through ResNet
        slice_features = self.encoder(slices_flat)  # (B*N, 2048)
        
        # Reshape back to (B, N, 2048)
        slice_features = slice_features.reshape(B, N, -1)
        
        # Aggregate across slices and classify to study level
        predictions = self.aggregator(slice_features)  # (B, 12)
        
        return predictions
    
    def get_num_parameters(self):
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("\nTesting KneeAbnormalityModel...\n")
    
    model = KneeAbnormalityModel(
        backbone='resnet50',
        pretrained=True,
        aggregation='max',
        num_abnormalities=12
    )
    
    print(f"Model parameters: {model.get_num_parameters():,}\n")
    
    # Test with correct 5D RGB input
    print("Testing with (B=2, N=32, H=224, W=224, C=3) RGB input...")
    test_input = torch.randn(2, 32, 224, 224, 3)
    output = model(test_input)
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    assert output.shape == (2, 12), f"Expected output (2, 12), got {output.shape}"
    
    print("\n✓ Model test passed!")