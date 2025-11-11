import time
import numpy as np
import pickle
import os
import matplotlib
from load_signal import load_signal
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from glob import glob
from extract_features import extract_features
from PRPD import product_map
from PIL import Image
from scipy import signal
print(os.getcwd())  # 打印当前工作目录

model_path = './model/model.pkl'
data_path = './signal/01_悬浮'
PRPD_path = './PRPD'

fs = 96000  # 采样频率
f_signal = 50  # 工频信号频率
len_signal = 1920   # 假设每个信号的长度为 1920 * 2
sensor_num = 2  # 传感器数量

pre_possessing = False #是否进行预处理

plt.rcParams['font.sans-serif'] = ['SimHei']  # 推荐使用SimHei字体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

try:#加载模型
    with open(model_path, 'rb') as file:
        rf_model = pickle.load(file)
        print(f'模型已加载')
except Exception as e:
    print(f'模型加载错误：{e}')

# 加载数据集
try:
    #dataset = np.loadtxt(data_path, delimiter=',')
    all_features = []
    img_path_list = glob(os.path.join(data_path, '*.txt'))
    img_num = len(img_path_list)
    seg = 0
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
        for start in range(0, channel_len, len_signal):
            for k in range(sensor_num):
                # 按照实验后的参数直接生成数据
                start_time = time.time()
                raw_signal = CHData[0, start:start + len_signal]
                raw_signal_for_prpd = raw_signal
                if pre_possessing:
                    x_n = raw_signal
                    X_z = signal.czt(x_n)
                    z = np.exp(2j * np.pi * np.arange(len(X_z)) / len(X_z))  # 单位圆上的点
                    H_z = 1 - 0.97 / z  # 计算 H(z)
                    # 在频域上进行乘法
                    Y_z = X_z * H_z  # 信号经过系统后的频域表示
                    raw_signal = np.fft.ifft(Y_z).real
                features = extract_features(raw_signal, fs)
                features = np.array(features)
                features = features.reshape(1, -1)
                y_pred = rf_model.predict(features)  # 预测类别
                y_pred_proba = rf_model.predict_proba(features)  # 预测概率
                #plt.close()  # 显式关闭图形，防止内存泄漏
                # 标签映射
                labels = ['悬浮', '沿面', '电晕', '无放电']
                # 输出结果和置信度
                for idx,(y, prob) in enumerate(zip(y_pred, y_pred_proba)):
                    label = labels[int(y)]
                    confidence = max(prob) * 100  # 获取最高置信度
                    print(f"样本 { seg + 1} - 预测结果: {label}, 置信度: {confidence:.2f}%")
                    prpd_map = product_map(raw_signal_for_prpd, fs, f_signal)
                    prpd_map = np.resize(prpd_map, (256, 256))  # 使用 numpy.resize 调整大小
                    # 创建一个新的图形，并显式设置尺寸为 256x256 像素
                    #plt.figure(figsize=(2.56, 2.56), dpi=100)  # figsize 设置为 256x256 像素，dpi=100 使其成为 256x256 图像
                    plt.imshow(prpd_map, cmap='gray', aspect='auto')  # 使用 imshow 显示二维数据
                    end_time = time.time()
                    spend_time = end_time - start_time
                    print(f'消耗时间:{spend_time}')
                    plt.title(f"消耗时间:{spend_time}秒 样本 { seg + 1} 预测结果: {label} 置信度: {confidence:.2f}%")
                    plt.xlabel("X 轴", fontsize=8)
                    plt.ylabel("Y 轴", fontsize=8)
                    plt.colorbar(label="强度")  # 添加颜色条
                    plt.grid(False)  # 如果需要网格可以设置为 True
                    plt.axis('on')  # 确保显示坐标轴
                    #Image.fromarray(prpd_map.astype(np.uint8)).save(f'{PRPD_path}/{file_name[-24:-4]}_{seg}.png', bbox_inches='tight', pad_inches=0.1)
                    plt.savefig(f"{PRPD_path}/{file_name[-24:-4]}_{seg}.png", bbox_inches='tight', pad_inches=0.2, transparent=False)
                    #plt.show()
                    plt.close()
                    seg+=1
                #all_features.append(features)  # 将数据添加到列表
    #merged_dataset = np.vstack(all_features)
except Exception as e:
    print(f'数据集加载错误：{e}')
'''
# 创建一个 DataFrame
#X = dataset
X = merged_dataset
Y = []

y_pred = rf_model.predict(X) # 预测类别
y_pred_proba = rf_model.predict_proba(X)  # 预测概率
end_time = time.time()
spend_time = end_time - start_time

print(f'消耗时间:{spend_time}')

# 标签映射
labels = ['悬浮', '沿面', '电晕', '无放电']

# 输出结果和置信度
for idx, (y, prob) in enumerate(zip(y_pred, y_pred_proba)):
    label = labels[int(y)]
    confidence = max(prob) * 100  # 获取最高置信度
    print(f"样本 {idx + 1} - 预测结果: {label}, 置信度: {confidence:.2f}%")
'''


