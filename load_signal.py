import os
import numpy as np
from glob import glob
from extract_features import extract_features

def load_signal(file_path,fs=96000,f_signal=50,len_signal=1920,sensor_num=2):
    all_features = []
    img_path_list = glob(os.path.join(file_path, '*.txt'))
    img_num = len(img_path_list)
    for num in range(img_num):
        file_name = img_path_list[num]
        counter = 1
        with open(file_name, 'rb') as f:
            # 获取文件长度
            f.seek(0, os.SEEK_END)
            data_len = f.tell()
            f.seek(0, os.SEEK_SET)
            # 读取数据
            data = np.fromfile(f, dtype=np.int32)
        # 假设每个传感器的数据占 4 字节
        CHData = np.zeros((data_len // sensor_num // 4, sensor_num))
        channel_len = 10 * 1920  # 假设每个通道的长度为 10 * 1920
        # 将数据划分到传感器
        for i in range(2):
            CHData[:channel_len, i] = data[i * channel_len:(i + 1) * channel_len]
        CHData = CHData.T  # 转置
        seg = 0
        for start in range(0, channel_len, len_signal):
            for k in range(sensor_num):
                # 按照实验后的参数直接生成数据
                raw_signal = CHData[k, start:start + len_signal]
                features = extract_features(raw_signal,fs)
                all_features.append(features)  # 将数据添加到列表
    merged_dataset = np.vstack(all_features)
    # print(all_features)
    # np.savetxt(f"features.csv", merged_dataset, delimiter=",")
    # print(f'模型已加载')
    return merged_dataset