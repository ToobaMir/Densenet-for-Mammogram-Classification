import torch
import torch.nn as nn
import torchvision.models as models

class MultiViewDenseNet121(nn.Module):
    """Multi-view DenseNet121 for mammogram classification
    
    Architecture:
    - Two parallel DenseNet121 branches for CC and MLO views
    - Feature fusion layer
    - Additional dense layers for final classification
    
    Args:
        num_classes (int): Number of classes for classification. Default: 1
        pretrained (bool): If True, use pretrained DenseNet121 weights. Default: True
        drop_rate (float): Dropout rate. Default: 0.0
    """
    def __init__(self, num_classes=1, pretrained=True, drop_rate=0.0):
        super(MultiViewDenseNet121, self).__init__()
        
        # Create two DenseNet121 models for CC and MLO views
        self.densenet_cc = models.densenet121(pretrained=pretrained)
        self.densenet_mlo = models.densenet121(pretrained=pretrained)
        
        # Remove the original classifier layers
        feature_dim = self.densenet_cc.classifier.in_features
        self.densenet_cc.classifier = nn.Identity()
        self.densenet_mlo.classifier = nn.Identity()
        
        # Feature fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )

    def forward_single_view(self, x, model):
        """Forward pass for a single view"""
        # Initial convolution and pooling
        x = model.features(x)
        
        # Global average pooling
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        
        return x

    def forward_features(self, x_cc, x_mlo):
        """Extract and fuse features from both views"""
        # Process each view separately
        cc_features = self.forward_single_view(x_cc, self.densenet_cc)
        mlo_features = self.forward_single_view(x_mlo, self.densenet_mlo)
        
        # Concatenate features from both views
        combined_features = torch.cat((cc_features, mlo_features), dim=1)
        
        # Fuse features
        fused_features = self.fusion(combined_features)
        
        return fused_features

    def forward(self, x_cc, x_mlo):
        """Forward pass for both views"""
        # Get fused features
        fused_features = self.forward_features(x_cc, x_mlo)
        
        # Final classification
        output = self.classifier(fused_features)
        
        return output

def create_model(num_classes=1, pretrained=True, drop_rate=0.0):
    """Create a multi-view DenseNet121 model instance"""
    model = MultiViewDenseNet121(
        num_classes=num_classes,
        pretrained=pretrained,
        drop_rate=drop_rate
    )
    return model

