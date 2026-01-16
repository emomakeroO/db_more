import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, mid_planes, out_planes, norm, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm(mid_planes)
        self.conv2 = nn.Conv2d(mid_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm(out_planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                norm(out_planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Decoder(nn.Module):
    def __init__(self, input_dim=512):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(input_dim, 256, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = self.deconv4(x)
        return x


class DynamicBiasNetwork(nn.Module):
    """
    """
    def __init__(self, num_classes, hidden_dim=64):
        super(DynamicBiasNetwork, self).__init__()
        self.num_classes = num_classes
        self.class_embed = nn.Embedding(num_classes, hidden_dim)

        self.bias_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # 输出3个专家的 Bias
        )
        nn.init.xavier_uniform_(self.class_embed.weight)

    def forward(self, class_idx):
        class_emb = self.class_embed(class_idx)
        bias = self.bias_mlp(class_emb)  # [B, 3]
        return bias


class ResNetAE(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, pooling='avgpool',
                 norm=nn.BatchNorm2d, recon_weight=1.0, class_freq=None, sparse_weight=0.01,
                 dcb_weight=0.1, alpha=1.0, beta=1.0, gamma=1.0):
        super(ResNetAE, self).__init__()

        self.in_planes = 64
        self.num_classes = num_classes
        self.recon_weight = recon_weight
        self.sparse_weight = sparse_weight
        self.dcb_weight = dcb_weight
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        if class_freq is not None:
            probs = torch.tensor(class_freq, dtype=torch.float32)
            self.class_probs = probs / probs.sum()
        else:
            self.class_probs = None

        if pooling == 'avgpool':
            self.pooling = nn.AvgPool2d(4)
        elif pooling == 'maxpool':
            self.pooling = nn.MaxPool2d(4)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, norm=norm)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, norm=norm)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, norm=norm)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, norm=norm)
        self.feat_dim = 512 * block.expansion
        self.classifier = nn.Linear(self.feat_dim, num_classes)
                     
        self.decoder0 = Decoder(input_dim=self.feat_dim)
        self.decoder1 = Decoder(input_dim=self.feat_dim)
        self.decoder2 = Decoder(input_dim=self.feat_dim)

        self.gate = nn.Sequential(
            nn.Linear(self.feat_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 3)
        )

        self.bias_net = DynamicBiasNetwork(num_classes)

    def _make_layer(self, block, planes, num_blocks, norm, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, planes, norm, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def compute_dcb_loss(self, device):
        """
        """
        if self.class_probs is None:
            return torch.tensor(0.0).to(device)

        all_classes = torch.arange(self.num_classes).to(device)
        # 获取所有类别的 Bias, 并取3个专家的均值作为该类的 scalar bias
        all_biases = self.bias_net(all_classes)
        avg_bias = all_biases.mean(dim=1)

        class_probs = self.class_probs.to(device)

        prob_bias = torch.sigmoid(avg_bias)
        dcb_loss = torch.sum(class_probs * torch.log(prob_bias + 1e-6))

        return dcb_loss

    def forward_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        h4 = self.layer4(out)
        p4 = self.pooling(h4)
        p4 = p4.view(p4.size(0), -1)
        return p4, h4

    def forward(self, x, class_idx=None):
        p4, h4 = self.forward_features(x)
        device = x.device

        # --- 1. Bias-aware Classification (对应文本 Eq. 1 的第一部分) ---
        logits = self.classifier(p4)

        if self.training:
            all_classes = torch.arange(self.num_classes).to(device)
            all_biases = self.bias_net(all_classes).mean(dim=1)
            logits = logits + self.gamma * all_biases

        gate_logits = self.gate(p4)

        rec0 = self.decoder0(h4)
        rec1 = self.decoder1(h4)
        rec2 = self.decoder2(h4)

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

        w_expanded = weights.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        recs = torch.stack([rec0, rec1, rec2], dim=1)
        rec_final = (w_expanded * recs).sum(dim=1)

        recon_loss = F.mse_loss(rec_final, x) * self.recon_weight

        mean_usage = weights.mean(dim=0)
        loss_balance = F.mse_loss(mean_usage, torch.ones(3).to(device) / 3)
        loss_sparse = -torch.sum(weights * torch.log(weights + 1e-8), dim=1).mean()
        gate_loss = loss_balance + self.sparse_weight * loss_sparse

        dcb_reg_loss = self.compute_dcb_loss(device) * self.dcb_weight

        return logits, recon_loss, gate_loss, dcb_reg_loss
        
def ResNet18AE(num_classes, class_freq, **kwargs):
    return ResNetAE(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, class_freq=class_freq, **kwargs)


def ResNet34AE(num_classes, class_freq, **kwargs):
    return ResNetAE(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, class_freq=class_freq, **kwargs)
