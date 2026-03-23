import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 必须重写一遍 ResNet 模型结构
# (因为 .pth 文件只存了参数，没存结构，就像只有肌肉没有骨头不行)
# ==========================================
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(identity)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
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
    return ResNet(BasicBlock, [2, 2, 2, 2])

# ==========================================
# 2. 定义预测逻辑
# ==========================================
# CIFAR-10 的 10 个类别（顺序绝对不能乱）
classes = ('飞机 (Plane)', '汽车 (Car)', '鸟 (Bird)', '猫 (Cat)', '鹿 (Deer)', 
           '狗 (Dog)', '青蛙 (Frog)', '马 (Horse)', '船 (Ship)', '卡车 (Truck)')

# 预处理：必须和训练时完全一致！
# 注意：我们必须把任意尺寸的图 resize 到 32x32，因为模型只见过这个尺寸的世界
predict_transform = transforms.Compose([
    transforms.Resize((32, 32)), 
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def predict_local_image(image_path, model_path="cifar10_resnet18.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载模型
    model = ResNet18().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    else:
        print(f"❌ 错误：找不到模型文件 {model_path}，请先运行训练脚本！")
        return

    # 2. 读取图片
    try:
        print(f"📂 正在读取: {image_path}")
        img_original = Image.open(image_path).convert('RGB')
        
        # 预处理
        img_tensor = predict_transform(img_original).unsqueeze(0).to(device)
        
        # 3. 推理
        with torch.no_grad():
            outputs = model(img_tensor)
            
            # 计算概率分布（用 Softmax 把分数变成百分比）
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, predicted = torch.max(probs, 1)
            
        class_name = classes[predicted.item()]
        confidence = conf.item() * 100
        
        print(f"🎉 预测结果: {class_name} | 置信度: {confidence:.2f}%")
        
        # 4. 显示结果
        plt.imshow(img_original) # 显示原图，不显示那张被缩小的模糊图
        plt.title(f"AI: {class_name} ({confidence:.1f}%)")
        plt.axis('off')
        plt.show()

    except Exception as e:
        print(f"❌ 图片处理失败: {e}")

# ==========================================
# 3. 执行
# ==========================================
# 把你要测的图片重命名为 test.jpg 放在同一目录下
if __name__ == '__main__':
    image_name = "test12.jpg" 
    if os.path.exists(image_name):
        predict_local_image(image_name)
    else:
        print("⚠️ 请找一张图片，重命名为 test.jpg 放在此文件夹下，然后再次运行！")