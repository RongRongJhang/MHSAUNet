import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
 
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        assert embed_size % num_heads == 0
        self.head_dim = embed_size // num_heads
        self.query_dense = nn.Linear(embed_size, embed_size)
        self.key_dense = nn.Linear(embed_size, embed_size)
        self.value_dense = nn.Linear(embed_size, embed_size)
        self.combine_heads = nn.Linear(embed_size, embed_size)
        self._init_weights()

    def split_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        x = x.reshape(batch_size, height * width, -1)

        query = self.split_heads(self.query_dense(x), batch_size)
        key = self.split_heads(self.key_dense(x), batch_size)
        value = self.split_heads(self.value_dense(x), batch_size)

        attention_weights = F.softmax(torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5), dim=-1)
        attention = torch.matmul(attention_weights, value)
        attention = attention.permute(0, 2, 1, 3).contiguous().reshape(batch_size, -1, self.embed_size)

        output = self.combine_heads(attention)
        return output.reshape(batch_size, height, width, self.embed_size).permute(0, 3, 1, 2)

    def _init_weights(self):
        init.xavier_uniform_(self.query_dense.weight)
        init.xavier_uniform_(self.key_dense.weight)
        init.xavier_uniform_(self.value_dense.weight)
        init.xavier_uniform_(self.combine_heads.weight)
        init.constant_(self.query_dense.bias, 0)
        init.constant_(self.key_dense.bias, 0)
        init.constant_(self.value_dense.bias, 0)
        init.constant_(self.combine_heads.bias, 0)
 
class Denoiser(nn.Module):
    def __init__(self, num_filters, kernel_size=3, activation='relu'):
        super(Denoiser, self).__init__()
        # 定義卷積層，輸入通道數改為 48 以匹配 F.pixel_unshuffle 的輸出
        self.conv1 = nn.Conv2d(48, num_filters, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.conv4 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        # 加入 MultiHeadSelfAttention
        self.bottleneck = MultiHeadSelfAttention(embed_size=num_filters, num_heads=4)
        # 上採樣層
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        # 輸出層與殘差層
        self.output_layer = nn.Conv2d(3, 3, kernel_size=kernel_size, padding=1)
        self.res_layer = nn.Conv2d(num_filters, 3, kernel_size=kernel_size, padding=1)
        self.activation = getattr(F, activation)
        self._init_weights()

    def forward(self, x):
        # 下採樣路徑
        x1 = self.activation(self.conv1(x))
        x2 = self.activation(self.conv2(x1))
        x3 = self.activation(self.conv3(x2))
        x4 = self.activation(self.conv4(x3))
        # 通過 MultiHeadSelfAttention
        x = self.bottleneck(x4)
        # 上採樣路徑與跳躍連接
        x = self.up4(x)
        # 調整 x 的尺寸以匹配 x3
        if x.shape[2:] != x3.shape[2:]:
            x = F.interpolate(x, size=x3.shape[2:], mode='nearest')
        x = self.up3(x + x3)
        # 調整 x 的尺寸以匹配 x2
        if x.shape[2:] != x2.shape[2:]:
            x = F.interpolate(x, size=x2.shape[2:], mode='nearest')
        x = self.up2(x + x2)
        # 調整 x 的尺寸以匹配 x1
        if x.shape[2:] != x1.shape[2:]:
            x = F.interpolate(x, size=x1.shape[2:], mode='nearest')
        x = x + x1
        x = self.res_layer(x)
        return torch.tanh(self.output_layer(x + x))

    def _init_weights(self):
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.output_layer, self.res_layer]:
            init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                init.constant_(layer.bias, 0)

class MHSAUNet(nn.Module):
    def __init__(self, num_filters=32):
        super(MHSAUNet, self).__init__()
        # 為每個分支定義 Denoiser 模組（包含 MultiHeadSelfAttention）
        self.denoiser_ycbcr = Denoiser(num_filters)

        # 新增的下採樣前 3x3 卷積層
        self.pre_shuffle_conv = nn.Conv2d(3, 12, kernel_size=3, padding=1)  # 輸出通道設為 12，因為下採樣 factor=2

        # 最終的 3x3 卷積層
        self.final_conv = nn.Conv2d(3, 12, kernel_size=3, padding=1)  # 調整輸出通道為 12 以配合上採樣

        self.gamma = 0.4  # Gamma 校正參數

        self._init_weights()

    def _rgb_to_ycbcr(self, image):
        # 將 RGB 轉換為 YCbCr
        r, g, b = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.14713 * r - 0.28886 * g + 0.436 * b + 0.5
        cr = 0.615 * r - 0.51499 * g - 0.10001 * b + 0.5
        return torch.stack([y, cb, cr], dim=1)

    def _rgb_to_oklab(self, image):
        # 分離 r, g, b 通道
        r, g, b = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :]

        # 將線性 sRGB 轉換至中間表徵 l, m, s
        l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
        m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
        s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

        # 分別取立方根 (使用 torch.sign 來正確處理正負值)
        eps = 1e-6
        l_ = torch.sign(l) * (torch.abs(l) + eps).pow(1/3)
        m_ = torch.sign(m) * (torch.abs(m) + eps).pow(1/3)
        s_ = torch.sign(s) * (torch.abs(s) + eps).pow(1/3)

        # 計算 Oklab 各通道
        L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
        a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
        b_out = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

        # 合併 L, a, b 三個通道
        oklab = torch.stack((L, a, b_out), dim=1)
        return oklab

    def _gamma_correction(self, image, gamma):
        # 確保輸入在 [0, 1] 範圍內，並添加數值穩定性
        eps = 1e-8  # 避免零值問題
        image = torch.clamp(image, 0, 1)  # 限制範圍
        return torch.pow(image + eps, gamma)

    def forward(self, x):
        # 將 RGB 轉換為 YCbCr 或 Oklab
        ycbcr = self._rgb_to_oklab(x)
        y, cb, cr = torch.split(ycbcr, 1, dim=1)

        # 對 Y 和 Cb 分支進行 Gamma 校正
        y = self._gamma_correction(y, self.gamma)
        cb = self._gamma_correction(cb, self.gamma)

        # 將處理後的三個分支合併
        combined = torch.cat([y, cb, cr], dim=1)

        # 新增：PixelShuffle 下採樣前的 3x3 卷積
        combined = self.pre_shuffle_conv(combined)
        # 使用 F.pixel_unshuffle 進行下採樣 (空間尺寸減半，通道數增加 4 倍)
        combined = F.pixel_unshuffle(combined, downscale_factor=2)

        # 對合併後的分支進行去噪處理
        ycbcr_denoised = self.denoiser_ycbcr(combined)

        # 通過最終的 3x3 卷積層
        output = self.final_conv(ycbcr_denoised)

        # 使用 F.pixel_shuffle 進行上採樣 (空間尺寸放大 2 倍，通道數減少 4 倍)
        output = F.pixel_shuffle(output, upscale_factor=2)

        # 確保輸出範圍在 [0, 1]
        return torch.sigmoid(output)

    def _init_weights(self):
        # 初始化新增的卷積層
        init.kaiming_uniform_(self.pre_shuffle_conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.pre_shuffle_conv.bias, 0)
        init.kaiming_uniform_(self.final_conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.final_conv.bias, 0)