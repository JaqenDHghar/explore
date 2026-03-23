import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 模型骨架 (必须完全一致)
# ==========================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ==========================================
# 2. 加载模型
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

model_path = "cat_dog_model.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ 成功加载模型: {model_path}")
else:
    print(f"❌ 找不到 {model_path}，请确认文件名正确！")
    exit()

# ==========================================
# 3. 预测本地图片函数
# ==========================================
predict_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

def predict_local_image(image_path):
    print(f"\n📂 正在读取图片: {image_path} ...")
    
    try:
        # 打开本地图片
        img = Image.open(image_path).convert('RGB')
        
        # 预处理
        img_tensor = predict_transform(img)
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # 推理
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            
        # 0是猫(cats_set), 1是狗(dogs_set)
        # 注意：这取决于 ImageFolder 读取时的字母顺序，通常 'c' 在 'd' 前面
        result = "🐶 狗 (Dog)" if predicted.item() == 1 else "🐱 猫 (Cat)"
        
        print(f"🎉 预测结果: {result}")
        
        # 显示图片
        plt.imshow(img)
        plt.title(f"AI: {result}")
        plt.axis('off')
        plt.show()

    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 '{image_path}'。请确认图片放在了代码目录下！")
    except Exception as e:
        print(f"❌ 图片读取出错: {e}")

# ==========================================
# 4. 执行预测
# ==========================================
# 这里填写你刚才放进去的图片名字
image_name = "test.jpg" 

predict_local_image(image_name)