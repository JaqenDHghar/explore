import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
print('开始了哦！')

# ==========================================
# 1. 核心原理：残差块 (Residual Block)
# ==========================================
# 这是一个精妙的工科设计：
# 如果网络这一层学不到东西，它至少可以把输入原样传过去 (x + out)，
# 保证了“不会比不加这一层更差”。这就解决了深层网络的退化问题。
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        # 卷积层 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) # 批归一化：防止数据分布跑偏
        self.relu = nn.ReLU(inplace=True)
        
        # 卷积层 2
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut (捷径/短路连接)
        # 如果输入输出维度不一致，需要用 1x1 卷积调整一下 shortcut 的维度，才能相加
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        identity = x                    # 1. 保留原始输入
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)  # 2. 核心步骤：原始输入 + 卷积结果
        out = self.relu(out)            # 3. 激活
        return out

# ==========================================
# 2. 组装 ResNet 骨架
# ==========================================
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # 初始处理层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 堆叠残差层 (ResNet-18 的结构: 2, 2, 2, 2)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2) # 尺寸减半
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2) # 尺寸减半
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2) # 尺寸减半

        # 输出层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2]) # 经典的 ResNet-18 结构

# ==========================================
# 3. 数据准备 (CIFAR-10)
# ==========================================
# 这里的预处理非常关键，加入了标准归一化
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4), # 随机裁剪
    transforms.RandomHorizontalFlip(),    # 随机翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # CIFAR-10 的均值和方差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

print("正在下载/加载 CIFAR-10 数据集 (约 160MB)...")
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0) # 批次调大，因为你有 5060

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

classes = ('飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车')

# ==========================================
# 4. 训练循环 (感受 5060 的力量)
# ==========================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device} (准备起飞)")

model = ResNet18().to(device)
criterion = nn.CrossEntropyLoss()
# 使用 SGD + 动量，这是训练 ResNet 的标准配置，比 Adam 更能跑出高分
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
# 学习率衰减：每 10 轮把学习率乘 0.1，模拟“越学越细致”
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

epochs = 30 # 跑 30 轮看看
print(f"开始训练 ResNet-18 on CIFAR-10...")

for epoch in range(epochs):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    # 更新学习率
    scheduler.step()
    
    acc = 100. * correct / total
    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {train_loss/(batch_idx+1):.4f} | Acc: {acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.4f}")

print("训练完成！开始最终测试...")

# 最终测试
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, targets in testloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

print(f"最终测试集准确率: {100. * correct / total:.2f}%")

torch.save(model.state_dict(), "cifar10_resnet18.pth")
print("模型已保存为 cifar10_resnet18.pth")