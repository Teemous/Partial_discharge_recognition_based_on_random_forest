import os
import numpy as np
import librosa
#import librosa.display
from scipy.stats import skew, kurtosis
from glob import glob
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="librosa")


# 提取时域和频域特征(单帧信号特征提取)
def extract_features(y, sr=96000):
    # 时域特征
    mean = np.mean(y)
    std = np.std(y)
    max_value = np.max(y)
    min_value = np.min(y)
    zero_crossings = librosa.feature.zero_crossing_rate(y)[0]
    skewness = skew(y)
    kurt = kurtosis(y)
    peak_value = np.max(np.abs(y))  # 峰值（信号的最大绝对值
    # 频域特征
    D = np.abs(librosa.stft(y))  # 短时傅里叶变换 (STFT)
    power_spectrum = np.square(D)  # 功率谱

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    # 时域特征的总特征值
    time_domain_features = [
        np.mean(y),
        np.std(y),
        np.max(y),
        np.min(y),
        np.mean(zero_crossings),
        skewness,
        kurt,
        peak_value,
    ]
    # 频域特征的总特征值
    freq_domain_features = [
        np.mean(power_spectrum),
        np.mean(spectral_centroid),
        np.mean(spectral_flatness),
        np.mean(spectral_rolloff),]
    # 合并所有特征
    total_features = time_domain_features + freq_domain_features
    return total_features


if __name__ == '__main__':
    file_path = './signal/01_悬浮'
    all_features = []
    img_path_list = glob(os.path.join(file_path, '*.txt'))
    img_num = len(img_path_list)
    for num in range(img_num):
        file_name = img_path_list[num]
        #file_path_full = os.path.join(file_path)
        counter = 1
        with open(file_name, 'rb') as f:
            # 获取文件长度
            f.seek(0, os.SEEK_END)
            data_len = f.tell()
            f.seek(0, os.SEEK_SET)

            # 读取数据
            data = np.fromfile(f, dtype=np.int32)

        # 假设每个传感器的数据占 4 字节
        CHData = np.zeros((data_len // 2 // 4, 2))
        channel_len = 10 * 1920  # 假设每个通道的长度为 10 * 1920

        # 将数据划分到传感器
        for i in range(2):
            CHData[:channel_len, i] = data[i * channel_len:(i + 1) * channel_len]

        CHData = CHData.T  # 转置
        seg = 0
        for start in range(0, channel_len, 1920):
            for k in range(2):
                #按照实验后的参数直接生成数据
                raw_signal = CHData[k, start:start + 1920]

                features = extract_features(raw_signal)
                all_features.append(features)  # 将数据添加到列表
    merged_dataset = np.vstack(all_features)
    print(all_features)
    np.savetxt(f"features.csv", merged_dataset, delimiter=",")