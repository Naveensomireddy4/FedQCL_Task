import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1  # Ensures consistency

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)

        # Fixing downsampling condition
        if downsample is None and (stride != 1 or in_channels != out_channels):
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)  # Ensure identity has the correct shape

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # Ensure out and identity have matching shapes
        out = self.relu(out)

        return out


class ResNetTIL(nn.Module):
    def __init__(self, block, num_blocks, in_channels=1, num_classes_per_task=2, num_tasks=5):
        super(ResNetTIL, self).__init__()
        self.in_planes = 64
        self.num_tasks = num_tasks
        self.current_task = 0  # Track the current task index

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Adaptive average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Dropout for generalization
        self.dropout = nn.Dropout(p=0.5)

        # Task-specific classification heads
        self.task_heads = nn.ModuleList([
            nn.Linear(512 * block.expansion, num_classes_per_task) for _ in range(num_tasks)
        ])

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(block(self.in_planes, out_channels, stride, downsample))
        self.in_planes = out_channels  # Update for next layers
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x, task_id=None):
        device = next(self.parameters()).device
        x = x.to(device)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)  # Flatten
        out = self.dropout(out)  # Apply dropout before classification

        if task_id is None:
            task_id = self.current_task  # Use the current task index
        elif task_id < 0 or task_id >= len(self.task_heads):
            raise ValueError(f"Invalid task_id: {task_id}. Must be between 0 and {len(self.task_heads) - 1}.")

        return self.task_heads[task_id].to(device)(out)  # Ensure classifier is on correct device

    def set_task(self, task_id):
        """Set the current task for inference or training."""
        if 0 <= task_id < len(self.task_heads):
            self.current_task = task_id
        else:
            raise ValueError(f"Invalid task_id: {task_id}. Maximum is {len(self.task_heads) - 1}.")

# Function to create a ResNet-18 for TIL
def ResNet18_TIL(in_channels=1, num_classes_per_task=8, num_tasks=4):
    return ResNetTIL(BasicBlock, [2, 2, 2, 2], in_channels, num_classes_per_task, num_tasks).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

