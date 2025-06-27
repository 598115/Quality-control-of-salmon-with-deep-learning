import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet34_Weights


class AttentionGate(nn.Module):
    """Attention Gate (AG) for the UNet architecture.
    Helps the model focus on relevant features during upsampling.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):

        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Ensure spatial dimensions match for addition
        if g1.size()[2:] != x1.size()[2:]:
            g1 = F.interpolate(g1, size=x1.size()[2:], mode='bilinear', align_corners=True)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DecoderBlock(nn.Module):
    """Enhanced decoder block for the UNet architecture with optional attention gates."""

    def __init__(self, in_channels, skip_channels, out_channels, attention_skip_channels=0, use_attention=True, dropout_p=0.0, main_path=True, upsample=2):
        super(DecoderBlock, self).__init__()

        self.main_path = main_path

        self.use_attention = use_attention
        if self.main_path == False:
            self.use_attention = False

        # Attention Gate
        if self.use_attention and main_path:
            self.attention_gate = AttentionGate(in_channels, attention_skip_channels, in_channels//2)

        # Upsampling
        self.upsample = nn.Upsample(scale_factor=upsample, mode='bilinear', align_corners=True)

        # First conv block after concatenation
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Second conv block
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p)
        )

    def forward(self, x, skip):
        # Upsample the input
        x = self.upsample(x)

        # Check if skip is a sequence/array type
        if not isinstance(skip, (list, tuple)):
            skip = [skip]

        # Apply attention mechanism if enabled
        if self.use_attention and skip is not None and self.main_path:
            skip[0] = self.attention_gate(x, skip[0])

        # Concatenate with skip connections
        if skip is not None:
                skip_tensors = [s for s in skip if s is not None]  # Filter out None skips
                
                if skip_tensors:  # Only proceed if there are valid skip connections
                    # Ensure spatial dimensions match
                    for s in skip_tensors:
                        if x.size()[2:] != s.size()[2:]:
                            x = F.interpolate(x, size=s.size()[2:], mode='bilinear', align_corners=True)
                    
                    # Concatenate all skip tensors with x
                    x = torch.cat([x] + skip_tensors, dim=1)

        # Apply convolution blocks
        x = self.conv1(x)
        if self.main_path:
            x = self.conv2(x)

        return x



class UNetPlusPlus(nn.Module): 
    """UNet++ architecture with ResNet34 backbone, attention gates, and deep supervision capabilities.
    This class implements an advanced UNet++ architecture with the following features:
    - ResNet34 backbone as encoder (can be pretrained or not)
    - Nested dense skip connections
    - Optional attention gates
    - Optional deep supervision
    - Dual output heads for zone and spot segmentation
    Args:
        n_classes_zone (int, optional): Number of classes for zone segmentation. Defaults to 4.
        n_classes_spot (int, optional): Number of classes for spot segmentation. Defaults to 3.
        pretrained (bool, optional): Whether to use pretrained ResNet34 weights. Defaults to True.
        use_attention (bool, optional): Whether to use attention gates in decoder. Defaults to True.
        use_deep_supervision (bool, optional): Whether to use deep supervision. Defaults to True.
        dropout_p (float, optional): Dropout probability. Defaults to 0.2.
    Architecture Details:
        Encoder: ResNet34 backbone with 5 encoding stages
        Bridge: Double convolutional block with dropout
        Decoder: Nested dense skip connections forming UNet++ architecture
        Output: Dual heads for zone and spot segmentation
    Forward Pass:
        Input: RGB image tensor of shape (B, 3, H, W)
        Output:
            - Training mode with deep supervision:
                ((zone_output, spot_output), deep_outputs)
                where deep_outputs is a list of (zone, spot) pairs from intermediate layers
            - Inference mode or without deep supervision:
                (zone_output, spot_output)
                where outputs are of shape (B, n_classes, H, W)
    Notes:
        - Uses Kaiming initialization for non-pretrained layers
        - Implements nested skip connections for better feature reuse
        - Deep supervision helps with gradient flow and intermediate predictions
        - Attention mechanism helps focus on relevant features
    """
    def __init__(self, n_classes_zone=4, n_classes_spot=3, pretrained=True, 
                 use_attention=True, use_deep_supervision=True, dropout_p=0.2):
        super(UNetPlusPlus, self).__init__()

        self.use_deep_supervision = use_deep_supervision

        # Load pre-trained ResNet34 encoder
        if pretrained:
            resnet = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        else:
            resnet = models.resnet34(weights=None)

        # Encoder layers
        self.encoder0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        ) 
        
        self.encoder1 = nn.Sequential(
            resnet.maxpool
        )  # 64 channels
        
        self.encoder2 = resnet.layer1  # 64 channels
        self.encoder3 = resnet.layer2  # 128 channels
        self.encoder4 = resnet.layer3  # 256 channels
        self.encoder5 = resnet.layer4  # 512 channels

        # Bridge 
        self.bridge = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Nested UNet++ decoder paths
        # Main decoder path
        self.decoder01 = DecoderBlock(512, 256, 256, 256, use_attention, dropout_p)
        self.decoder11 = DecoderBlock(256, 128+128, 128, 128, use_attention, dropout_p)
        self.decoder21 = DecoderBlock(128, 64+64+128, 64, 64, use_attention, dropout_p)
        self.decoder31 = DecoderBlock(64, 64+32+64+128, 64, 64, use_attention, dropout_p)

        # Level 1 decoders 
        self.decoder02 = DecoderBlock(64, 64, 32, main_path=False)  # Encoder block 1, 2 in. To supervised out
        self.decoder12 = DecoderBlock(128, 64, 64, main_path=False)
        self.decoder22 = DecoderBlock(256, 128, 128, main_path=False)

        # Level 2 decoders
        self.decoder03 = DecoderBlock(64, 32+64, 64, main_path=False)  # Changed from (512, 256, 256)
        self.decoder13 = DecoderBlock(128, 64+64, 128, main_path=False)

        # Level 3 decoder
        self.decoder04 = DecoderBlock(128, 64+32+64, 128, main_path=False)  # Changed from (512, 0, 256)


        # Task-specific final layers
        self.final_zone = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_classes_zone, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 
        )

        self.final_spot = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_classes_spot, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 
        )

        # Deep supervision outputs
        if self.use_deep_supervision:
            self.deep_zone1 = nn.Sequential(
                nn.Conv2d(32, n_classes_zone, kernel_size=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # Changed from 8 to 4
            )
            self.deep_zone2 = nn.Sequential(
                nn.Conv2d(64, n_classes_zone, kernel_size=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # Changed from 16 to 8
            )

            self.deep_zone3 = nn.Sequential(
                nn.Conv2d(128, n_classes_zone, kernel_size=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # Changed from 16 to 8
            )

            self.deep_spot1 = nn.Sequential(
                nn.Conv2d(32, n_classes_spot, kernel_size=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # Changed from 8 to 4
            )
            self.deep_spot2 = nn.Sequential(
                nn.Conv2d(64, n_classes_spot, kernel_size=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # Changed from 16 to 8
            )

            self.deep_spot3 = nn.Sequential(
                nn.Conv2d(128, n_classes_spot, kernel_size=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # Changed from 16 to 8
            )


        # Initialize weights for better convergence
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for non-pretrained layers using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                # Check if tensor has elements before initializing
                if m.weight.numel() > 0:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None and m.bias.numel() > 0:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight.numel() > 0:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None and m.bias.numel() > 0:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x):
        # Encoder
        x0 = self.encoder0(x)       # 64 channels, 1/2 resolution
        x1 = self.encoder1(x0)      # 64 channels, 1/4 resolution, maxpool
        x2 = self.encoder2(x1)      # 64 channels, 1/4 resolution
        x3 = self.encoder3(x2)      # 128 channels, 1/8 resolution
        x4 = self.encoder4(x3)      # 256 channels, 1/16 resolution
        x5 = self.encoder5(x4)      # 512 channels, 1/32 resolution

        # Bridge
        bridge = self.bridge(x5)    # 512 channels, 1/32 resolution

        # Decoder with nested skip connections (UNet++)

        # Level 2 (more dense connections)
        d02 = self.decoder02(x2, x0)    # 32 channels, 1/2 resolution
        d12 = self.decoder12(x3, x2)    # 64 channels, 1/4 resolution
        d22 = self.decoder22(x4, x3)    # 128 channels, 1/8 resolution

        # Level 3
        d03 = self.decoder03(d12, [x0, d02])    # 64 channels, 1/2 resolution
        d13 = self.decoder13(d22, [x2, d12])    # 128 channels, 1/4 resolution

        # Level 4
        d04 = self.decoder04(d13, [x0, d02, d03])   # 128 channels, 1/2 resolution

        # Level 1 (direct connections)
        d01 = self.decoder01(bridge, x4)  # 256 channels, 1/16 resolution
        d11 = self.decoder11(d01, [x3, d22])     # 128 channels, 1/8 resolution
        d21 = self.decoder21(d11, [x2, d12, d13])     # 64 channels, 1/4 resolution
        d31 = self.decoder31(d21, [x0, d02, d03, d04])     # 64 channels, 1/2 resolution

        # Deep supervision outputs
        deep_outputs = []
        if self.use_deep_supervision:
            zone_out1 = self.deep_zone1(d02)  # From level 3
            zone_out2 = self.deep_zone2(d03)  # From level 4
            zone_out3 = self.deep_zone3(d04)

            spot_out1 = self.deep_spot1(d02)  # From level 3
            spot_out2 = self.deep_spot2(d03)  # From level 4
            spot_out3 = self.deep_spot3(d04)

            deep_outputs = [
                (zone_out1, spot_out1), 
                (zone_out2, spot_out2),
                (zone_out3, spot_out3)
            ]

        # Final outputs (main branch)
        output_zone = self.final_zone(d31)
        output_spot = self.final_spot(d31)

        if self.use_deep_supervision and self.training:
            return (output_zone, output_spot), deep_outputs
        else:
            return output_zone, output_spot

