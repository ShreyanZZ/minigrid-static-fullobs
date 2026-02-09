import torch
import torch.nn as nn
import torch.nn.functional as F

class CertificateNet(nn.Module):
    def _initialize_weights(self):
        """
        Initializes weights to break symmetry while ensuring the starting 
        output V(x) is very close to 0 (approx 0.006) for stability.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming init is best for Conv2d with ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: 
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Standard init for hidden linear layers
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
                
        last_layer = self.actor[-1] 
        nn.init.normal_(last_layer.weight, mean=0, std=0.001)
        nn.init.constant_(last_layer.bias, -5.0)
        
    def __init__(self, obs_space):
        super().__init__()
        # Input is 8x8 with 3 channels
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)), # Output: 16 x 7 x 7
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)), # Output: 32 x 6 x 6
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)), # Output: 64 x 5 x 5
            nn.ReLU()
        )
        
        # Flatten size: 64 channels * 5 * 5 = 1600
        self.feature_size = 64 * 13 * 13 

        self.actor = nn.Sequential(
            nn.Linear(self.feature_size, 64),
            nn.Tanh(),                         
            nn.Linear(64, 1)
        )

        self._initialize_weights()

    def forward(self, obs):
        # Input obs is (Batch, Height, Width, Channels) -> (B, 8, 8, 3)
        # Permute to (Batch, Channels, Height, Width) -> (B, 3, 8, 8)
        x = obs.permute(0, 3, 1, 2).float()
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1) # Flatten
        v = self.actor(x)
        return F.softplus(v)