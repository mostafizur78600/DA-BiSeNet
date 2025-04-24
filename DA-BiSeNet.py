import torch
import torch.nn as nn
import torch.nn.functional as F
import timm  # Import timm for Xception backbone based on the architecture mentioned in the paper
from torchsummary import summary


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SAM(nn.Module):
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, bias=self.bias)

    def forward(self, x):
        max = torch.max(x,1)[0].unsqueeze(1)
        avg = torch.mean(x,1).unsqueeze(1)
        concat = torch.cat((max,avg), dim=1)
        output = self.conv(concat)
        output = F.sigmoid(output) * x 
        return output 

class CAM(nn.Module):
    def __init__(self, channels, r):
        super(CAM, self).__init__()
        self.channels = channels
        self.r = r
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.channels, out_features=self.channels//self.r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.channels//self.r, out_features=self.channels, bias=True))

    def forward(self, x):
        max = F.adaptive_max_pool2d(x, output_size=1)
        avg = F.adaptive_avg_pool2d(x, output_size=1)
        b, c, _, _ = x.size()
        linear_max = self.linear(max.view(b,c)).view(b, c, 1, 1)
        linear_avg = self.linear(avg.view(b,c)).view(b, c, 1, 1)
        output = linear_max + linear_avg
        output = F.sigmoid(output) * x
        return output
    
class CBAM(nn.Module):
    def __init__(self, channels, r):
        super(CBAM, self).__init__()
        self.channels = channels
        self.r = r
        self.sam = SAM(bias=False)
        self.cam = CAM(channels=self.channels, r=self.r)

    def forward(self, x):
        output = self.cam(x)
        output = self.sam(output)
        return output + x



class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out




class SpatialPath(nn.Module):
    def __init__(self):
        super(SpatialPath, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        # Coordinate Attention is added here
        self.coordatt = CoordAtt(256, 256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.coordatt(x)  # Apply Coordinate Attention
        return x 

class XceptionBackbone(nn.Module):
    def __init__(self):
        super(XceptionBackbone, self).__init__()
        # Load Xception backbone from timm
        self.backbone = timm.create_model('xception', pretrained=True, features_only=True, out_indices=(2, 3, 4))

    def forward(self, x):
        # Extract features from different stages of the Xception backbone
        features = self.backbone(x)
        return features[-1]  # Return the deepest layer for context

class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionRefinementModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        attention = torch.mean(x, dim=(2, 3), keepdim=True)
        attention = self.sigmoid(attention)
        return x * attention

class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels):
        super(FeatureFusionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.cbam = CBAM(256, r=16)  # Provide the 'r' argument


    def forward(self, spatial_out, context_out):
        # Ensure that the spatial and context outputs have the same size
        if spatial_out.size()[2:] != context_out.size()[2:]:
            context_out = F.interpolate(context_out, size=spatial_out.size()[2:], mode='bilinear', align_corners=False)

        fusion = torch.cat([spatial_out, context_out], dim=1)
        fusion = F.relu(self.bn1(self.conv1(fusion)))
        fusion = F.relu(self.bn2(self.conv2(fusion)))
        fusion = self.cbam(fusion)  # Apply CBAM
        return fusion

class BiSeNet(nn.Module):
    def __init__(self, num_classes):
        super(BiSeNet, self).__init__()
        self.spatial_path = SpatialPath()
        self.context_path = XceptionBackbone()
        self.arm = AttentionRefinementModule(2048, 128)  # Adjust channels to match Xception's output
        self.ffm = FeatureFusionModule(128 + 256)  # Combine spatial and context paths
        self.output_conv = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        spatial_out = self.spatial_path(x)  # Spatial details
        context_out = self.context_path(x)  # Xception context

        # Apply Attention Refinement Module on the context features
        context_out = self.arm(context_out)

        # Fuse spatial and context outputs
        fusion_out = self.ffm(spatial_out, context_out)

        # Final segmentation output
        out = self.output_conv(fusion_out)
        out = F.interpolate(out, size=x.size()[2:], mode='bilinear', align_corners=False)
        return out

# Example of creating the model
if __name__ == "__main__":
    model = BiSeNet(num_classes=7).to(device)
    sample_input = torch.randn(1, 3, 227, 320).to(device)  # Example input based on the image properties you provided
    output = model(sample_input)
    print(f"Output shape: {output.shape}")
    summary(model, (3, 227, 320))  # Print model summary for input shape (3, 227, 320)
