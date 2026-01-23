from torch import nn


class SEBlock(nn.Module):
    def __init__(self, channels: int, r: int = 8):
        super().__init__()
        hidden = max(1, channels // r)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, use_se: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_ch) if use_se else None

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.se is not None:
            out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class HabitatCNN(nn.Module):
    def __init__(self, in_ch=16, num_classes=71, widths=(32, 64, 128), blocks=(1, 1, 1), use_se=False, mixer=False):
        super().__init__()
        stem_in = widths[0]
        if mixer:
            self.mixer = nn.Sequential(
                nn.Conv2d(in_ch, stem_in, kernel_size=1, bias=False),
                nn.BatchNorm2d(stem_in),
                nn.ReLU(inplace=True),
            )
            in_ch = stem_in
        else:
            self.mixer = None

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, widths[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )

        self.stage1 = self._make_stage(widths[0], widths[0], blocks[0], stride=1, use_se=use_se)
        self.stage2 = self._make_stage(widths[0], widths[1], blocks[1], stride=2, use_se=use_se)
        self.stage3 = self._make_stage(widths[1], widths[2], blocks[2], stride=2, use_se=use_se)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(widths[2], num_classes)

    def _make_stage(self, in_ch, out_ch, blocks, stride, use_se):
        layers = [BasicBlock(in_ch, out_ch, stride=stride, use_se=use_se)]
        for _ in range(blocks - 1):
            layers.append(BasicBlock(out_ch, out_ch, stride=1, use_se=use_se))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.mixer is not None:
            x = self.mixer(x)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


def make_cnn(arch: str, in_ch=16, num_classes=71) -> HabitatCNN:
    if arch == "resnet_small":
        return HabitatCNN(in_ch, num_classes, widths=(32, 64, 128), blocks=(1, 1, 1), use_se=False, mixer=False)
    if arch == "resnet_medium":
        return HabitatCNN(in_ch, num_classes, widths=(48, 96, 192), blocks=(2, 2, 2), use_se=False, mixer=False)
    if arch == "resnet_small_se_mixer":
        return HabitatCNN(in_ch, num_classes, widths=(32, 64, 128), blocks=(1, 1, 1), use_se=True, mixer=True)
    raise ValueError(f"Unknown arch: {arch}")
