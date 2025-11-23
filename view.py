import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

# 读取 .mat 文件
mat_file_path = 'C:/Users/Think/Downloads/fractal_label.mat'  # 替换为你的 .mat 文件路径
mat_data = sio.loadmat(mat_file_path)

# 假设 .mat 文件中存储图像数据的变量名为 'image_data'
# 你需要根据实际情况修改这个变量名
image_data = mat_data.get('hyperdata_gt')

if image_data is not None:
    # 如果数据是多维的，取第一个通道（假设是灰度图像）
    if len(image_data.shape) > 2:
        image_data = image_data[:, :, 0]

    # 确保数据类型为 uint8
    if image_data.dtype != np.uint8:
        # 归一化到 [0, 255] 范围
        image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data)) * 255
        image_data = image_data.astype(np.uint8)

    # 显示灰度图像
    plt.imshow(image_data, cmap='gray')
    plt.axis('off')

    # 保存图像
    output_image_path = 'C:/Users/Think/Downloads/3.png'  # 替换为你想要保存的图像路径
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    print(f"图像已保存到 {output_image_path}")
else:
    print("未找到图像数据，请检查 .mat 文件中的变量名。")