import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def product_map(raw_signal, fs, f_signal):
    """
    raw_signal: 输入的原始信号数据
    fs: 采样频率
    f_signal: 工频信号频率
    返回：生成的PRPD图像
    """
    raw_signal = raw_signal[raw_signal >= 0]
    #归一化信号(获取最大幅值后将每个采样点除以最大幅值)
    normalized_signal = raw_signal / np.max(np.abs(raw_signal))
    #参考电压 1 mV
    V_ref = 1e-3
    #转换为电压
    data_V = normalized_signal * V_ref
    #转换为 mV (1 V = 1000 mV)
    data_mV = data_V * 1000
    #处理无穷大或NaN值（将无穷大值和NAN值替换为数据中的最小值）
    data_mV[np.isnan(data_mV) | np.isinf(data_mV)] = np.min(data_mV[np.isfinite(data_mV)])
    #生成时间轴和相位
    len_signal = len(raw_signal)
    '''
    原程序：
    时间轴计算：从0到2个工频周期
    t = 0:1/fs:(1/fs)*(len-1)   生成从0到(1/fs)*(len-1)的数组，步长为1/fs,每次增加一个采样点的时间间隔
    不使用np.arange(0, len_signal, 1/fs)的原因：得到的是浮点数数组，计算开销太大
    '''
    t = np.arange(0,len_signal) / fs  #生成时间轴，长度为 len_signal（即 3840 个点）
    #工频信号相位计算，跨越0到4π
    '''
    np.mod:取模运算
    工频信号相位计算：Phase = 2*pi*fsignal*t
    '''
    phase = np.mod(2 * np.pi * f_signal * t, 2 * np.pi)  #相位从0到π
    #将相位转换为0到720度
    phase_deg = np.degrees(phase)  #转换为度数
    phase_deg = (phase_deg / np.max(phase_deg)) * 360

    '''
    如果 phase_deg 小于 720，np.mod(phase_deg, 720) 的结果就是 phase_deg 本身。
    如果 phase_deg 大于 720，则会减去 720，直到结果落入 0 到 720 的范围内。
    '''
    phase_deg = np.mod(phase_deg, 1 * 360)  #确保相位范围从0到360度
    #print(max(phase_deg))
    #绘制PRPD图缩放到256x256，图像尺寸
    box_len = 256
    PRPD = np.zeros((box_len, box_len))
    for i in range(len_signal):
        #将相位映射到图像的x坐标
        x = int(np.floor(phase_deg[i] / (1 * 360) * box_len))
        #将幅值映射到图像的y坐标
        y = box_len - int(np.floor((data_mV[i] - np.min(data_mV)) / 1 * box_len))
        #限制x和y使其在有效范围内(0<=x,y<box_len)
        x = min(max(x,0),box_len - 1)
        y = min(max(y,0),box_len - 1)
        #增加图像上的对应点
        PRPD[y, x] += 1
        #对图像进行邻域扩展平滑处理
        PRPD[max(0, y - 1):min(y + 2, box_len), max(0, x - 1):min(x + 2, box_len)] += 1
    #将图像中的零值点设为 255（表示空白区域）
    PRPD[PRPD == 0] = 255
    return PRPD