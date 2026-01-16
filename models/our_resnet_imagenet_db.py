
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck, ResNet


# --- 1. 复用 DynamicBiasNetwork (保持不变) ---
class DynamicBiasNetwork(nn.Module):
    def __init__(self, num_classes, hidden_dim=64):
        super(DynamicBiasNetwork, self).__init__()
        self.num_classes = num_classes
        self.class_embed = nn.Embedding(num_classes, hidden_dim)

        self.bias_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)
        )
        nn.init.xavier_uniform_(self.class_embed.weight)

    def forward(self, class_idx):
        class_emb = self.class_embed(class_idx)
        bias = self.bias_mlp(class_emb)
        return bias


# --- 2. 适配 ImageNet 的 Decoder ---
class DecoderImageNet(nn.Module):
    def __init__(self, input_dim=2048):
        super(DecoderImageNet, self).__init__()
        # Input: [B, 2048, 7, 7] -> Output: [B, 3, 224, 224]
        # 需要 5 次 stride=2 的上采样 (7->14->28->56->112->224)

        # 7 -> 14
        self.deconv1 = nn.ConvTranspose2d(input_dim, 1024, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(1024)

        # 14 -> 28
        self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(512)

        # 28 -> 56
        self.deconv3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)

        # 56 -> 112
        self.deconv4 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)

        # 112 -> 224
        self.deconv5 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=True)

    def forward(self, x):
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = F.relu(self.bn4(self.deconv4(x)))
        x = self.deconv5(x)  # ImageNet 需归一化, 这里输出原始数值让 loss 去拟合
        return x


# --- 3. DB-MoRE ResNet50 ---
class ResNet50AE(ResNet):
    def __init__(self, num_classes=1000, class_freq=None,
                 recon_weight=1.0, sparse_weight=0.01, dcb_weight=0.1,
                 alpha=1.0, beta=1.0, gamma=1.0, **kwargs):

        # 初始化标准 ResNet50 骨干
        # layers=[3, 4, 6, 3] 对应 ResNet50
        super(ResNet50AE, self).__init__(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)

        # --- DB-MoRE 参数 ---
        self.feat_dim = 2048  # ResNet50 layer4 output channels
        self.num_classes = num_classes
        self.recon_weight = recon_weight
        self.sparse_weight = sparse_weight
        self.dcb_weight = dcb_weight
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # 类别频率处理
        if class_freq is not None:
            probs = torch.tensor(class_freq, dtype=torch.float32)
            self.class_probs = probs / probs.sum()
        else:
            self.class_probs = None

        # --- DB-MoRE 模块 ---
        # 1. 替换原有 fc 为适配层 (虽然 ResNet 自带 fc，但我们显式定义更清晰)
        self.fc = nn.Linear(self.feat_dim, num_classes)

        # 2. 专家解码器 (针对 ImageNet 适配)
        self.decoder0 = DecoderImageNet(input_dim=self.feat_dim)
        self.decoder1 = DecoderImageNet(input_dim=self.feat_dim)
        self.decoder2 = DecoderImageNet(input_dim=self.feat_dim)

        # 3. 门控网络
        self.gate = nn.Sequential(
            nn.Linear(self.feat_dim, 512), nn.ReLU(),  # 稍微加大隐层
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 3)
        )

        # 4. DCB 网络
        self.bias_net = DynamicBiasNetwork(num_classes)

    def compute_dcb_loss(self, device):
        if self.class_probs is None:
            return torch.tensor(0.0).to(device)

        all_classes = torch.arange(self.num_classes).to(device)
        all_biases = self.bias_net(all_classes)
        avg_bias = all_biases.mean(dim=1)

        class_probs = self.class_probs.to(device)

        # 严格对应文本公式 (带 log)
        prob_bias = torch.sigmoid(avg_bias)
        dcb_loss = torch.sum(class_probs * torch.log(prob_bias + 1e-6))

        return dcb_loss

    def forward_features(self, x):
        # 标准 ResNet 前向流程，直到 layer4
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        h4 = self.layer4(x)  # [B, 2048, 7, 7]

        p4 = self.avgpool(h4)  # [B, 2048, 1, 1]
        p4 = torch.flatten(p4, 1)  # [B, 2048]

        return p4, h4

    def forward(self, x, class_idx=None):
        p4, h4 = self.forward_features(x)
        device = x.device

        # --- 1. Classification & DCB Adjustment ---
        logits = self.fc(p4)

        if self.training:
            all_classes = torch.arange(self.num_classes).to(device)
            all_biases = self.bias_net(all_classes).mean(dim=1)
            # Logit Adjustment: f(x) + eta * b(y)
            logits = logits + self.gamma * all_biases

        # --- 2. MoRE Reconstruction ---
        gate_logits = self.gate(p4)

        rec0 = self.decoder0(h4)
        rec1 = self.decoder1(h4)
        rec2 = self.decoder2(h4)

        # 计算重构误差 [B, C, H, W] -> [B]
        loss_rec0 = F.mse_loss(rec0, x, reduction='none').mean(dim=[1, 2, 3])
        loss_rec1 = F.mse_loss(rec1, x, reduction='none').mean(dim=[1, 2, 3])
        loss_rec2 = F.mse_loss(rec2, x, reduction='none').mean(dim=[1, 2, 3])

        # Quality Score
        q0 = -torch.log(loss_rec0 + 1e-8)
        q1 = -torch.log(loss_rec1 + 1e-8)
        q2 = -torch.log(loss_rec2 + 1e-8)
        quality_logits = torch.stack([q0, q1, q2], dim=1)

        # Routing Bias
        if class_idx is not None:
            bias_logits = self.bias_net(class_idx)
        else:
            bias_logits = torch.zeros_like(gate_logits)

        final_gate_logits = (self.alpha * gate_logits +
                             self.beta * quality_logits +
                             self.gamma * bias_logits)

        weights = F.softmax(final_gate_logits, dim=1)

        # 加权融合
        w_expanded = weights.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        recs = torch.stack([rec0, rec1, rec2], dim=1)
        rec_final = (w_expanded * recs).sum(dim=1)

        # --- 3. Loss Calculation ---
        recon_loss = F.mse_loss(rec_final, x) * self.recon_weight

        mean_usage = weights.mean(dim=0)
        loss_balance = F.mse_loss(mean_usage, torch.ones(3).to(device) / 3)
        loss_sparse = -torch.sum(weights * torch.log(weights + 1e-8), dim=1).mean()
        gate_loss = loss_balance + self.sparse_weight * loss_sparse

        dcb_reg_loss = self.compute_dcb_loss(device) * self.dcb_weight

        return logits, recon_loss, gate_loss, dcb_reg_loss


# 实例化函数
def ResNet50DBMoRE(num_classes=1000, class_freq=None, **kwargs):
    return ResNet50AE(num_classes=num_classes, class_freq=class_freq, **kwargs)


