import numpy as np
import scipy.io as sio
from sklearn import preprocessing
import random

def load_dataset(Dataset):

    if Dataset== "RV":
        data_b = sio.loadmat('datasets/river/river_before.mat')
        data_a = sio.loadmat('datasets/river/river_after.mat')
        data_before = data_b['river_before']
        data_after = data_a['river_after']
        gt_mat = sio.loadmat('datasets/river/groundtruth.mat')
        gt = gt_mat['lakelabel_v1']
        dataset_name = "river"

    if Dataset== "FM":
        data_b = sio.loadmat('datasets/farmland/farm06.mat')
        data_a = sio.loadmat('datasets/farmland/farm07.mat')
        data_before = data_b['imgh']
        data_after = data_a['imghl']
        gt_mat = sio.loadmat('datasets/farmland/label.mat')
        gt = gt_mat['label']   #与MATLAB打开.mat文件的变量名称一致
        dataset_name = "farmland"

    if Dataset== "USA":
        data_b = sio.loadmat('datasets/usa/usa_before.mat')
        data_a = sio.loadmat('datasets/usa/usa_after.mat')
        data_before = data_b['T1']
        data_after = data_a['T2']
        gt_mat = sio.loadmat('datasets/usa/groundtruth.mat')
        gt = gt_mat['Binary']   #与MATLAB打开.mat文件的变量名称一致
        dataset_name = "USA"
    
    if Dataset== "YC":
        data_b = sio.loadmat('datasets/yc/yc_before.mat')
        data_a = sio.loadmat('datasets/yc/yc_after.mat')
        data_before = data_b['T1']
        data_after = data_a['T2']
        gt_mat = sio.loadmat('datasets/yc/groundtruth.mat')
        gt = gt_mat['Binary']   #与MATLAB打开.mat文件的变量名称一致
        dataset_name = "YC"

    if Dataset== "SANTA":
        data_b = sio.loadmat('datasets/santa/barbara_2013.mat')
        data_a = sio.loadmat('datasets/santa/barbara_2014.mat')
        data_before = data_b['HypeRvieW']
        data_after = data_a['HypeRvieW']
        gt_mat = sio.loadmat('datasets/santa/barbara_gtChanges.mat')
        gt = gt_mat['HypeRvieW']   #与MATLAB打开.mat文件的变量名称一致
        dataset_name = "SANTA"

    if Dataset== "BAY":
        data_b = sio.loadmat('datasets/bay/Bay_Area_2013.mat')
        data_a = sio.loadmat('datasets/bay/Bay_Area_2015.mat')
        data_before = data_b['HypeRvieW']
        data_after = data_a['HypeRvieW']
        gt_mat = sio.loadmat('datasets/bay/bayArea_gtChanges2.mat')
        gt = gt_mat['HypeRvieW']   #与MATLAB打开.mat文件的变量名称一致
        dataset_name = "BAY"
        
    height, width, bands = data_before.shape
    if height < 500 :
        gt = 2 - gt

    print(dataset_name)
    print(height, width, bands)
    data_concat = np.concatenate((data_before, data_after), axis=-1)
    data_before = np.reshape(data_before, [height * width, bands])
    data_after = np.reshape(data_after, [height * width, bands])
    data_concat = np.reshape(data_concat, [height * width, 2 * bands])
    minMax = preprocessing.StandardScaler()
    data_before = minMax.fit_transform(data_before)
    data_after = minMax.fit_transform(data_after)
    data_concat = minMax.fit_transform(data_concat)
    data_before = np.reshape(data_before, [height, width, bands])
    data_after = np.reshape(data_after, [height, width, bands])
    data_concat = np.reshape(data_concat, [height, width, 2 * bands])

    return data_before, data_after, data_concat, gt, dataset_name

def sampling(sampling_mode, train_rate, gt):

    train_rand_idx = []      # 实际训练样本索引
    gt_1d = np.reshape(gt, [-1])  # 展平后的gt

    if sampling_mode == 'random':

        idx = np.where(gt_1d < 3)[-1]
        # 这行代码使用 np.where 函数找出 gt_1d 数组中值小于 3 的元素的索引。
        # np.where 返回一个元组，元组中的每个元素对应一个维度的索引数组，这里取最后一个元素（通常对于一维数组就是所需的索引数组）
        samplesCount = len(idx)
        # 计算满足 gt_1d < 3 条件的样本数量，即符合条件的索引数量
        rand_list = [i for i in range(samplesCount)]
        # 生成一个【0-samplescount-1】的整数列表，用于后续随机采样
        rand_idx = random.sample(rand_list, np.ceil(samplesCount * train_rate).astype('int32'))
        # 从 rand_list 中随机无放回地抽取 np.ceil(samplesCount * train_rate) 个元素作为随机索引。
        # np.ceil 是向上取整函数，将 samplesCount * train_rate 的结果向上取整，然后转换为 int32 类型
        rand_real_idx_per_class = idx[rand_idx]
        # 根据随机抽取的索引 rand_idx 从 idx 数组中选取对应的索引，得到实际的样本索引
        train_rand_idx.append(rand_real_idx_per_class)
         # 将选取的实际样本索引添加到 train_rand_idx 列表中

        train_rand_idx = np.array(train_rand_idx)   # 将 train_rand_idx 列表转换为 numpy 数组
        train_index = []   # 初始化一个空列表，用于存储训练集的索引
        for c in range(train_rand_idx.shape[0]):  # 遍历 train_rand_idx 数组的每一行，将每一行的元素添加到 train_index 列表中
            a = train_rand_idx[c]
            for j in range(a.shape[0]):
                train_index.append(a[j])
        train_index = np.array(train_index)  # 将 train_index 列表转换为 numpy 数组

        train_index = set(train_index)              # 将训练集索引数组转换为集合，方便后续进行集合运算
        all_index = [i for i in range(len(gt_1d))]  # 生成一个从 0 到 len(gt_1d) - 1 的整数列表，表示所有样本的索引
        all_index = set(all_index)     # 将所有样本索引列表转换为集合

        background_idx = np.where(gt_1d == 0)[-1]  # 找出 gt_1d 数组中值为 0 的元素的索引（背景像素）
        background_idx = set(background_idx)   # 将背景样本的索引数组转换为集合
        test_index = all_index - train_index - background_idx 
        # 通过集合的差运算，得到测试集的索引集合，即所有样本索引集合减去训练集索引集合和背景样本索引集合

        val_count = int(0.01 * (len(test_index) + len(train_index)))
        # 计算验证集的样本数量，为训练集和测试集样本总数的 1%
        val_index = random.sample(test_index, val_count)
         # 从测试集索引集合中随机抽取 val_count 个元素作为验证集的索引
        val_index = set(val_index)  # 将验证集索引列表转换为集合
        test_index = test_index - val_index  # 从测试集索引集合中移除验证集的索引，得到最终的测试集索引集合

        test_index = list(test_index)   # 将三个集的索引集合转换为列表
        train_index = list(train_index)
        val_index = list(val_index)

    if sampling_mode == 'low':  #此模式中用不到采样率，已在代码中固定

        # 找出非背景元素（值为 1 和 2）的索引
        non_background_idx = np.where((gt_1d == 1) | (gt_1d == 2))[-1]
        non_background_count = len(non_background_idx)

        # 计算训练集的样本数量，为非背景元素总数的 5%
        train_count = int(0.05 * non_background_count)

        # 随机抽取 5% 的非背景元素作为训练集
        train_rand_idx = random.sample(list(non_background_idx), train_count)
        train_index = set(train_rand_idx)

        # 剩下的 95% 作为验证集
        val_index = set(non_background_idx) - train_index

        # 所有非背景元素的集合减去训练集 作为测试集
        test_index = set(non_background_idx)- train_index

        # 剩下的 95% 作为验证集
        val_index = set(non_background_idx) - train_index

        test_index = list(test_index)
        train_index = list(train_index)
        val_index = list(val_index)

    return train_index, val_index, test_index

def one_hot(gt_mask, height, width):
    # print(f"\n输入gt_mask形状: {gt_mask.shape}, 内容示例:\n{gt_mask[:3, :3] if gt_mask.size > 0 else gt_mask}\n")

    gt_one_hot = []
    for i in range(gt_mask.shape[0]):
        for j in range(gt_mask.shape[1]):
            temp = np.zeros(2, dtype=np.float32)
            if gt_mask[i, j] != 0:
                temp[int(gt_mask[i, j]) - 1] = 1
            # print(f"gt_mask: {gt_mask}")
            gt_one_hot.append(temp)
    gt_one_hot = np.reshape(gt_one_hot, [height, width, 2])
    return gt_one_hot

"""
    print("\ngt_mask的详细内容（带坐标）：")
    for i in range(gt_mask.shape[0]):
    for j in range(gt_mask.shape[1]):
        print(f"位置({i},{j}): {gt_mask[i, j]}")
"""
    # print(f"重塑后的one-hot数组形状: {gt_one_hot.shape}")
    

def make_mask(gt_mask, height, width):

    label_mask = np.zeros([height * width, 2])
    temp_ones = np.ones([2])
    gt_mask_1d = np.reshape(gt_mask, [height * width])
    for i in range(height * width):
        if gt_mask_1d[i] != 0:
            label_mask[i] = temp_ones
    label_mask = np.reshape(label_mask, [height * width, 2])
    return label_mask

def get_mask_onehot(gt, index):
    height, width = gt.shape
    gt_1d = np.reshape(gt, [-1])
    gt_mask = np.zeros_like(gt_1d)
    for i in range(len(index)):
        gt_mask[index[i]] = gt_1d[index[i]]
        pass
    gt_mask = np.reshape(gt_mask, [height, width])
    sampling_gt = gt_mask
    gt_onehot = one_hot(gt_mask, height, width)
    gt_mask = make_mask(gt_mask, height, width)

    return gt_onehot, gt_mask, sampling_gt
