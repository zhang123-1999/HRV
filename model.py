import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """
    CBAM 通道注意力模块 (1D版)
    关注 'What'：什么样的特征是重要的？
    """
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # 防止除以0或通道过小，确保至少有1个中间通道
        mid_planes = max(1, in_planes // ratio)
        
        # 共享感知层 (Shared MLP)
        self.fc1 = nn.Conv1d(in_planes, mid_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(mid_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """
    CBAM 空间注意力模块 (1D版)
    关注 'Where'：哪个时间段的信号是重要的？
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # padding 保证输出尺寸不变
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 在通道维度上做平均和最大化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    """
    完整 CBAM 模块：串联通道注意力和空间注意力
    """
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        # 先进行通道加权
        out = x * self.ca(x)
        # 再进行空间加权
        result = out * self.sa(out)
        return result

class InceptionBlock(nn.Module):
    """
    Inception Block based on RPnet paper.
    Uses kernels of sizes 15, 17, 19, 21 in parallel.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Reduce dimension first to save computation (1x1 conv)
        bottleneck_channels = max(1, in_channels // 4)
        self.branch1x1 = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1)

        # Parallel convolutions with different kernel sizes
        self.conv15 = nn.Conv1d(bottleneck_channels, out_channels // 4, kernel_size=15, padding=7)
        self.conv17 = nn.Conv1d(bottleneck_channels, out_channels // 4, kernel_size=17, padding=8)
        self.conv19 = nn.Conv1d(bottleneck_channels, out_channels // 4, kernel_size=19, padding=9)
        # Ensure the last branch fills the remaining channels
        rem_channels = out_channels - 3 * (out_channels // 4)
        self.conv21 = nn.Conv1d(bottleneck_channels, rem_channels, kernel_size=21, padding=10)

        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x_reduced = self.branch1x1(x)
        out1 = self.conv15(x_reduced)
        out2 = self.conv17(x_reduced)
        out3 = self.conv19(x_reduced)
        out4 = self.conv21(x_reduced)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        out = self.bn(out)
        out = self.activation(out)
        return out

class ResInceptionBlock(nn.Module):
    """
    Residual Inception Block.
    Adds a skip connection to the Inception Block.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.inception = InceptionBlock(in_channels, out_channels)
        # 1x1 conv to match dimensions if needed for the skip connection
        if in_channels != out_channels:
            self.skip_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x):
        identity = self.skip_conv(x)
        out = self.inception(x)
        return out + identity

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Downsample using Stride 2
        self.down_conv = nn.Conv1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.res_inception = ResInceptionBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.down_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.res_inception(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # Upsample using Transpose Conv
        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        # Skip connection concatenation happens in forward
        total_channels = out_channels + skip_channels
        self.res_inception = ResInceptionBlock(total_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        
        # Handle padding issues if dimensions don't match exactly due to odd input lengths
        if x.size(2) != skip.size(2):
            diff = skip.size(2) - x.size(2)
            x = F.pad(x, (diff // 2, diff - diff // 2))
            
        x = torch.cat([skip, x], dim=1) #
        x = self.res_inception(x)
        return x

class IncResUnet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=16):
        super().__init__()
        
        # ... (Encoder 和 CBAM 定义保持不变) ...
        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_filters),
            nn.LeakyReLU(0.2, inplace=True),
            ResInceptionBlock(base_filters, base_filters)
        )
        self.pool1 = EncoderBlock(base_filters, base_filters * 2)
        self.cbam1 = CBAM(base_filters * 2)
        self.pool2 = EncoderBlock(base_filters * 2, base_filters * 4)
        self.cbam2 = CBAM(base_filters * 4)
        self.pool3 = EncoderBlock(base_filters * 4, base_filters * 8)
        self.cbam3 = CBAM(base_filters * 8)
        self.pool4 = EncoderBlock(base_filters * 8, base_filters * 16)
        self.cbam4 = CBAM(base_filters * 16)

        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = ResInceptionBlock(base_filters * 16, base_filters * 16)

        # ... (Decoder 定义保持不变) ...
        self.up4 = DecoderBlock(base_filters * 16, base_filters * 8, base_filters * 8)
        self.up3 = DecoderBlock(base_filters * 8, base_filters * 4, base_filters * 4)
        self.up2 = DecoderBlock(base_filters * 4, base_filters * 2, base_filters * 2)
        self.up1 = DecoderBlock(base_filters * 2, base_filters, base_filters)
        self.final_conv = nn.Conv1d(base_filters, out_channels, kernel_size=1)

        # === 新增：计数回归头 (Count Head) ===
        # 输入是瓶颈层特征 [B, 256, 75] -> 全局平均池化 -> [B, 256] -> FC -> [B, 1]
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.count_head = nn.Sequential(
            nn.Linear(base_filters * 16, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 1) # 输出标量：R峰数量
        )

    def forward(self, x):
        # Encoder Path
        e1 = self.enc1(x)
        e2 = self.cbam1(self.pool1(e1))
        e3 = self.cbam2(self.pool2(e2))
        e4 = self.cbam3(self.pool3(e3))
        e5 = self.cbam4(self.pool4(e4))

        # Bottleneck
        b = self.bottleneck(self.dropout(e5))

        # === 分支 1: 计数头 ===
        # 使用 bottleneck 的特征来预测数量
        b_pooled = self.global_pool(b).flatten(1) # [B, 256, 1] -> [B, 256]
        pred_count = self.count_head(b_pooled)

        # === 分支 2: 解码器 (生成距离图) ===
        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        dist_map = self.final_conv(d1)

        # 返回两个输出
        return dist_map, pred_count