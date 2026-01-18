import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
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
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.tanh(self.deconv4(x))
        return x


class ResNetAE(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, pooling='avgpool',
                 norm=nn.BatchNorm2d, recon_weight=1, class_freq=None):
        super(ResNetAE, self).__init__()
        if pooling == 'avgpool':
            self.pooling = nn.AvgPool2d(4)
        elif pooling == 'maxpool':
            self.pooling = nn.MaxPool2d(4)
        else:
            raise Exception('Unsupported pooling: %s' % pooling)
        self.in_planes = 64
        self.recon_weight = recon_weight

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, norm=norm)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, norm=norm)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, norm=norm)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, norm=norm)

        self.linear_classifier0 = nn.Linear(512, num_classes)
        self.linear_classifier1 = nn.Linear(512, num_classes)
        self.linear_classifier2 = nn.Linear(512, num_classes)

        self.decoder0 = Decoder()
        self.decoder1 = Decoder()
        self.decoder2 = Decoder()

        self.class_freq = class_freq  
        self.recon_weight = recon_weight
        if class_freq is not None:
            self.freq_embed = nn.Embedding(len(class_freq), 512)
            gate_input_dim = 512 + 64  
        else:
            gate_input_dim = 512

        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def _make_layer(self, block, planes, num_blocks, norm, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, planes, norm, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward_features(self, x):
        c1 = F.relu(self.bn1(self.conv1(x)))
        h1 = self.layer1(c1)
        h2 = self.layer2(h1)
        h3 = self.layer3(h2)
        h4 = self.layer4(h3)
        p4 = self.pooling(h4)
        p4 = p4.view(p4.size(0), -1)
        return p4, h4

    def forward_classifier(self, p4):
        logits0 = self.linear_classifier0(p4)
        logits1 = self.linear_classifier1(p4)
        logits2 = self.linear_classifier2(p4)
        return logits0, logits1, logits2

    def forward(self, x, class_idx=None):
        p4, h4 = self.forward_features(x)

        logits0, logits1, logits2 = self.forward_classifier(p4)

        gate_input = p4

        if hasattr(self, 'freq_embed') and class_idx is not None:
            freq_emb = self.freq_embed(class_idx)
            gate_input = torch.cat([gate_input, freq_emb], dim=1)

        gate_weights = F.softmax(self.gate(gate_input), dim=1)
        if class_idx is not None and self.class_freq is not None:
            freq_weight = torch.sqrt(1.0 / (self.class_freq[class_idx] + 1e-8))
            gate_weights = gate_weights * freq_weight.unsqueeze(1)
            gate_weights = F.normalize(gate_weights, p=1, dim=1)
        reconstructed0 = self.decoder0(h4)
        reconstructed1 = self.decoder1(h4)
        reconstructed2 = self.decoder2(h4)
        
        expanded_weights = gate_weights.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        reconstructions = torch.stack([reconstructed0, reconstructed1, reconstructed2], dim=1)
        reconstructed = (expanded_weights * reconstructions).sum(dim=1)

        recon_loss = F.mse_loss(reconstructed, x) * self.recon_weight

        return logits0, logits1, logits2, recon_loss


def ResNet18AE(num_classes=10, pooling='avgpool', norm=nn.BatchNorm2d, class_freq=None):
    return ResNetAE(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, pooling=pooling, norm=norm,
                    class_freq=class_freq)


def ResNet34AE(num_classes=10, pooling='avgpool', norm=nn.BatchNorm2d, class_freq=None):
    return ResNetAE(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, pooling=pooling, norm=norm,
                    class_freq=class_freq)


# 损失函数
def loss_fn(logits, targets, recon_loss, recon_weight=1):
    ce_loss = F.cross_entropy(logits, targets)
    total_loss = ce_loss + recon_loss * recon_weight
    return total_loss
