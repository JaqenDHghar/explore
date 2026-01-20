import numpy as np
import matplotlib.pyplot as plt

# 1. 准备参数
H, J= 400, 600  # 画布高宽
center_y, center_x = 200, 300 # 圆心坐标 (注意先 y 后 x)
radius = 100     # 半径

# 2. 创建黑色画布
# dtype=np.uint8 表示 0-255 的整数
canvas = np.zeros((H, J, 3), dtype=np.uint8)

# --- 核心技术：制造“坐标系” ---
# 我们需要两个矩阵：
# 一个叫 Y_matrix，里面存着每个像素的 Y 坐标
# 一个叫 X_matrix，里面存着每个像素的 X 坐标

# 生成 0 到 H-1 的行索引向量，和 0 到 W-1 的列索引向量
y_indices = np.arange(H)
x_indices = np.arange(J)

# meshgrid 会把这两个向量“编织”成两个大的二维矩阵
# Y_grid 矩阵里每一行都是一样的，都是当前的行号
# X_grid 矩阵里每一列都是一样的，都是当前的列号
X_grid, Y_grid = np.meshgrid(x_indices, y_indices)

# 打印看看这两个网格是啥样的，辅助理解
print("Y 坐标网格形状:", Y_grid.shape)
print(Y_grid) # 如果想看具体数据可以取消注释

# --- 核心技术：数学公式造模版 (Mask) ---
# 这一步是 NumPy 的精髓。
# 我们直接对整个矩阵进行数学运算。
# 这一行代码，实际上对矩阵里的 24万个点 (400*600) 同时进行了距离计算。
dist_squared = (X_grid - center_x)**2 + (Y_grid - center_y)**2

# 生成掩膜：判断距离是否小于半径的平方
# 这是一个全是 True/False 的二维矩阵
mask_circle = dist_squared < radius**2

print("掩膜形状:", mask_circle.shape)
print("掩膜数据类型:", mask_circle.dtype) # bool 类型

# --- 3. 喷漆操作 ---
# 语法：canvas[掩膜为True的地方] = 新颜色
# 将满足条件的像素点设为白色 [255, 255, 255]
canvas[mask_circle] = [255, 255, 255]


# 4. 显示
plt.figure(figsize=(6, 4))
plt.title("Math Circle (Boolean Masking)")
plt.imshow(canvas)
plt.axis('on') # 打开坐标轴看看圆心对不对
plt.show()