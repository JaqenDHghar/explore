import kagglehub

# Download latest version
path = kagglehub.dataset_download("demonllord/indian-smartphones-under-20000specs-prices")

print("Path to dataset files:", path)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import re

# ==========================================
# 第一步：智能加载数据 (演示 glob 和 os 的用法)
# ==========================================
# 使用 glob 模糊匹配文件名，防止文件名变动导致找不到
csv_pattern = "*indian_smartphones*.csv"
files = glob.glob(csv_pattern)

if not files:
    # 如果找不到，尝试直接读取你上传的具体文件名
    file_path = "indian_smartphones_under_20000_raw(1).csv"
else:
    file_path = files[0] # 取第一个匹配到的文件

print(f"正在读取文件: {file_path}")
df = pd.read_csv(file_path)

# ==========================================
# 第二步：数据清洗 (演示 re 的强大之处)
# ==========================================
def clean_currency(value):
    """清洗价格：'₹12,999' -> 12999.0"""
    if isinstance(value, str):
        # re.sub 用正则表达式替换掉所有非数字和小数点的字符
        clean_str = re.sub(r'[^\d.]', '', value)
        return float(clean_str) if clean_str else None
    return value

def extract_brand(model_name):
    """提取品牌：'vivo T4x 5G...' -> 'vivo'"""
    if isinstance(model_name, str):
        return model_name.split()[0] # 取第一个单词作为品牌
    return "Unknown"

# 1. 清洗价格列
df['Price_Clean'] = df['Price'].apply(clean_currency)

# 2. 清洗评分列 (Ratings 通常含有逗号 '1,234')
if 'Ratings' in df.columns:
    df['Ratings_Clean'] = df['Ratings'].astype(str).apply(clean_currency)

# 3. 提取品牌
df['Brand'] = df['Model Name'].apply(extract_brand)

# ==========================================
# 第三步：专业绘图 (Matplotlib + Seaborn)
# ==========================================
# 设置画布大小
plt.figure(figsize=(20, 6))
sns.set_style("whitegrid") # 设置背景样式

# 图1: 价格分布 (直方图 + 核密度曲线)
plt.subplot(1, 3, 1)
sns.histplot(df['Price_Clean'].dropna(), kde=True, color='skyblue', bins=20)
plt.title('Smartphone Price Distribution (价格分布)', fontsize=14)
plt.xlabel('Price (INR)')
plt.ylabel('Frequency')

# 图2: 各品牌市场占有率 (条形图)
plt.subplot(1, 3, 2)
# 统计前10大品牌
top_brands = df['Brand'].value_counts().head(10)
sns.barplot(x=top_brands.index, y=top_brands.values, palette='viridis')
plt.title('Top 10 Brands by Model Count (品牌机型数量)', fontsize=14)
plt.xticks(rotation=45)
plt.ylabel('Count')

# 图3: 价格与评分的关系 (散点图)
plt.subplot(1, 3, 3)
if 'Ratings_Clean' in df.columns:
    # 过滤掉评论数过少的极端数据，使图表更清晰
    sns.scatterplot(x='Price_Clean', y='Ratings_Clean', data=df, 
                    hue='Brand', alpha=0.7, size='Ratings_Clean', sizes=(20, 200), legend=False)
    plt.title('Price vs. Ratings Count (价格与热度)', fontsize=14)
    plt.xlabel('Price (INR)')
    plt.ylabel('Number of Ratings')
    plt.yscale('log') # 评分数量差异巨大，使用对数坐标轴显示更友好

plt.tight_layout()
plt.show()

print("\n--- 清洗后的数据预览 ---")
print(df[['Brand', 'Price_Clean', 'Ratings_Clean']].head())