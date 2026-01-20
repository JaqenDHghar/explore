import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 1. 读取图片
# 读进来是 RGB，为了严谨，强制转一下
img = np.array(Image.open('test.jpg').convert('RGB'))
H, W, C = img.shape

print(f"图片形状: {H}x{W}, 通道: {C}")

# 2. 拆分通道（为了写判断条件方便）
# 取出所有行、所有列的第0、1、2个通道
R_channel = img[:, :, 0]
G_channel = img[:, :, 1]
B_channel = img[:, :, 2]

# --- 核心技术：复合条件造模版 (Multi-condition Mask) ---
# 我们定义“比较红”的阈值：
# R > 160  且  G < 80  且  B < 80
# (这些数字需要根据实际图片的光照情况微调)

# 重要细节：在 NumPy 数组进行逻辑“与”操作时，必须用 `&` 符号，不能用 `and`。
# 并且每个条件最好用括号 () 括起来，防止优先级错误。
mask_red = (R_channel > 10) & (G_channel < 140) & (B_channel < 230)

print("红色掩膜形状:", mask_red.shape) # 这是一个二维的 True/False 矩阵

# --- 3. 制造二值化图像 ---
# 创建一个全黑的画布，大小和原图一样，但只要单通道 (灰度图)
binary_result = np.zeros((H, W), dtype=np.uint8)

# 喷漆：掩膜为 True 的地方，喷成白色 (255)
binary_result[mask_red] = 255


# 4. 可视化对比
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Red Segmentation Mask (Binary)")
# 显示灰度图时，要告诉 matplotlib 用黑白模式，否则它可能会自作聪明上伪彩色
plt.imshow(binary_result, cmap='gray')
plt.axis('off')

print("显示结果...")
plt.show()