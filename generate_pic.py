import matplotlib.pyplot as plt
import numpy as np
import torch

# 分类图中 2 表示不变（黑色，函数中被设为0）；1 表示变化（白色）

def generate_png(label, name: str, scale: float = 4.0, dpi: int = 400):
    fig, ax = plt.subplots()
    numlabel = np.array(label)
    numlabel = numlabel.astype(np.int16)  # 数组类型转换
    numlabel = np.where(numlabel > 1, 0, 1) # 数组中大于1的元素设为0，否则设为1
    # 灰度映射，数组中的值自动归一化到[0,1]区间，最小值0映射为黑色（不变），最大值1映射为白色（变化）
    plt.imshow(numlabel, cmap='gray')
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
    foo_fig = plt.gcf()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    foo_fig.savefig(name + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0)
    
    return numlabel

def generate_png_low(classification_map, gt, name: str, scale: float = 4.0, dpi: int = 400):
    # 如果 classification_map 和 gt 是 torch.Tensor 类型，将其转换为 NumPy 数组
    if isinstance(classification_map, torch.Tensor):
        classification_map = classification_map.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
        
    # 创建一个与 classification_map 相同形状的数组，用于存储标记后的结果
    marked_image = np.zeros(classification_map.shape, dtype=np.uint8)
    
    # 逐像素对比 classification_map 和 gt
    for i in range(classification_map.shape[0]):
        for j in range(classification_map.shape[1]):
            if gt[i, j] == 0:
                # 该像素标记为黑色（背景，值为 0）
                marked_image[i, j] = 0
            elif classification_map[i,j] == 1 :
                # 该像素标记为灰色（变化，值为 1）
                marked_image[i, j] = 1
            else:
                # 该像素标记为白色（不变，值为2）
                marked_image[i, j] = 2
            
    
    # 创建一个新的图形和坐标轴对象
    fig, ax = plt.subplots()
    
    # 使用灰度映射将标记后的数组显示为图像
    plt.imshow(marked_image, cmap='gray')
    
    # 隐藏坐标轴
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    # 根据输入的缩放因子和图像分辨率设置图形的大小
    fig.set_size_inches(classification_map.shape[1] * scale / dpi, classification_map.shape[0] * scale / dpi)
    
    # 获取当前图形对象
    foo_fig = plt.gcf()
    
    # 隐藏坐标轴上的刻度
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    
    # 调整子图的布局，使其充满整个图形
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    
    # 将图形保存为 PNG 文件，设置透明背景、分辨率和边距
    foo_fig.savefig(name + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0)

    return marked_image