import numpy as np
import matplotlib.pyplot as plt
from generate_pic import generate_png

def mark_errors(detection_result, ground_truth, name: str, scale: float = 4.0, dpi: int = 400):
    """
    标记检测结果中的漏检和误检像素，并保存为PNG图片。
    :param detection_result: 检测结果图，二维数组，1表示变化，0表示不变
    :param ground_truth: 真值变化图，二维数组，1表示变化，0表示不变
    :param name: 保存图片的文件名
    :param scale: 图片缩放比例
    :param dpi: 图片分辨率
    """
    print(f"检测结果数据类型: {detection_result.dtype}")
    print(f"真值数据类型: {ground_truth.dtype}")
    print(f"检测结果取值范围: {np.min(detection_result)} - {np.max(detection_result)}")
    print(f"真值取值范围: {np.min(ground_truth)} - {np.max(ground_truth)}")

    # 初始化彩色图像，初始为黑色
    colored_image = np.zeros((detection_result.shape[0], detection_result.shape[1], 3), dtype=np.uint8)

    # 标记漏检像素（真值为变化，检测结果为不变）为绿色
    missed_detections = np.logical_and(ground_truth == 1, detection_result == 0)
    colored_image[missed_detections] = [0, 255, 0]

    # 标记误检像素（真值为不变，检测结果为变化）为红色
    false_alarms = np.logical_and(ground_truth == 0, detection_result == 1)
    print(f"误检像素数量: {np.sum(false_alarms)}")
    colored_image[false_alarms] = [255, 0, 0]

    # 正确检测的像素保持为黑白（根据检测结果）
    correct_non_detections = np.logical_and(ground_truth == detection_result, ground_truth == 1)
    colored_image[correct_non_detections] = [255, 255, 255]  # 白色表示变化
    correct_non_detections = np.logical_and(ground_truth == detection_result, ground_truth == 0)
    colored_image[correct_non_detections] = [0, 0, 0]  # 黑色表示不变


    # 保存标记后的图片
    fig, ax = plt.subplots()
    ax.imshow(colored_image)
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.set_size_inches(detection_result.shape[1] * scale / dpi, detection_result.shape[0] * scale / dpi)
    foo_fig = plt.gcf()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    foo_fig.savefig(name + '_marked.png', format='png', transparent=True, dpi=dpi, pad_inches=0)
    plt.close(fig)

def mark_errors_low(detection_result, ground_truth, name: str, scale: float = 4.0, dpi: int = 400):
    """
    标记检测结果中的漏检和误检像素，并保存为PNG图片。
    :param detection_result: 检测结果图，二维数组，1表示变化，0表示不变
    :param ground_truth: 真值变化图，二维数组，1表示变化，0表示不变
    :param name: 保存图片的文件名
    :param scale: 图片缩放比例
    :param dpi: 图片分辨率
    """
    # 初始化彩色图像，初始为黑色
    colored_image = np.zeros((detection_result.shape[0], detection_result.shape[1], 3), dtype=np.uint8)

    # 标记漏检像素（真值为变化，检测结果为不变）为绿色
    missed_detections = np.logical_and(ground_truth == 1, detection_result == 2)
    colored_image[missed_detections] = [0, 255, 0]

    # 标记误检像素（真值为不变，检测结果为变化）为红色
    false_alarms = np.logical_and(ground_truth == 2, detection_result == 1)
    colored_image[false_alarms] = [255, 0, 0]

    # 正确检测的像素保持为黑白（根据检测结果）
    correct_detections = np.logical_and(ground_truth == detection_result, ground_truth == 1)
    colored_image[correct_detections] = [128, 128, 128]  # 灰色表示变化
    correct_non_detections = np.logical_and(ground_truth == detection_result, ground_truth == 2)
    colored_image[correct_non_detections] = [255, 255, 255]  # 白色表示不变
    correct_non_detections = np.logical_and(ground_truth == detection_result, ground_truth == 0)
    colored_image[correct_non_detections] = [0, 0, 0]  # 黑色表示背景

    # 保存标记后的图片
    fig, ax = plt.subplots()
    ax.imshow(colored_image)
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.set_size_inches(detection_result.shape[1] * scale / dpi, detection_result.shape[0] * scale / dpi)
    foo_fig = plt.gcf()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    foo_fig.savefig(name + '_marked.png', format='png', transparent=True, dpi=dpi, pad_inches=0)
    plt.close(fig)