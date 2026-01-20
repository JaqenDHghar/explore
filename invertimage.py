import numpy as np
from PIL import Image  # 负责读取图片
import matplotlib.pyplot as plt # 负责显示图片

# --- 1. 数据采集阶段 ---
# 读取图片文件
# 这里的 'test.jpg' 必须和你代码在同一个目录下，如果你的图片是png，请修改这里
try:
    original_image = Image.open('test.jpg')
except FileNotFoundError:
    print("错误：找不到 'test.jpg'。请确保图片文件和代码文件在同一个文件夹里！")
    exit()

# --- 2. 数据转换阶段 (最核心的一步!) ---
# 工科视角：将“图片对象”强制转换为“NumPy三维数组”
# 计算机此刻不再把它看作图片，而是一个巨大的数字方阵
image_array = np.array(original_image)

# 打印一下这个矩阵的“骨架”看看
# 输出格式通常是 (高度Height, 宽度Width, 通道数Channels)
# 通道数 3 代表 RGB (红绿蓝) 三个颜色层
print("-" * 30)
print(f"图像矩阵的形状 (H, W, C): {image_array.shape}")
print(f"图像数据类型: {image_array.dtype}") # 通常是 uint8 (0-255 的整数)
print("-" * 30)


# --- 3. 数据处理阶段 (算法实现) ---
# 实验目标：做一个“底片反色”效果。
# 原理：RGB颜色是0-255。白色是255，黑色是0。
# 反色就是用 255 减去当前的颜色值。
# 在 C 语言里，你需要写三个嵌套的 for 循环来遍历每一个像素。
# 在 NumPy 里，只需要一行代码，它会自动对矩阵里的几百万个数字同时做减法。
inverted_array = 255 - image_array


# --- 4. 数据可视化阶段 ---
# 使用 Matplotlib 把两个矩阵画出来对比

plt.figure(figsize=(10, 5)) # 创建一个画布

# 画左边的图（原图）
plt.subplot(1, 2, 1) # 1行2列，第1个位置
plt.title("Original Image")
plt.imshow(image_array)
plt.axis('off') # 关掉坐标轴看着更清爽

# 画右边的图（处理后的图）
plt.subplot(1, 2, 2) # 1行2列，第2个位置
plt.title("Inverted Matrix (NumPy)")
plt.imshow(inverted_array)
plt.axis('off')

print("正在显示图像窗口，请查看...")
plt.show() # 把画好的图弹窗显示出来