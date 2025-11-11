# **基于随机森林的局部放电识别**

### **一、环境要求**

所需环境

Anaconda3

python 3.9

pycharm (IDE)

numpy~=1.26.4
matplotlib~=3.7.5
scikit-learn~=1.5.2
pillow~=11.3.0
scipy~=1.11.4
librosa~=0.10.1

### **二、项目文件介绍**

predict.py： 是模型的预测文件

train.py： 模型训练文件，可修改文件路径

extract_features.py： 提取时域与频域特征，不需修改

load_signal.py：加载信号，不需修改

PCA.py:数据降维，不需修改

PRPD:生成PRPD图谱，不需修改


data文件夹：存放训练数据,dataset.csv为训练所用数据(均值、标准差、最大值、最小值、过零率、偏度、峰度、峰值、功率谱均值、频谱质心、频谱平坦度、频谱滚降点)

model文件夹：该文件夹是用来存放训练好模型的目录，模型文件为model.pkl

PRPD文件夹：用于存放预测输出的PRPD图谱

signal:存放预测使用的原始局部放电信号数据

## **使用步骤如下：**

### **训练：**

（1）在此 ./data 文件夹中放入原始信号数据经过时域与频域特征提取后的结果，保存为csv文件作为训练数据集（本项目文件夹中dataset.csv文件为整理好的数据，其余csv文件为每种局部放电信号类别各自特征提取后的结果）

（2）修改train.py文件中的model_path为训练模型的指定输出位置、data_path为训练数据集位置，执行后可在指定路径下得到训练后的模型model.pkl

### **预测：**

（1）将predict.py文件中的model_path修改为模型的保存路径，修改data_path为需要进行预测的原始局部放电信号数据，修改PRPD_path为PRPD图谱输出位置，执行后可输出各样本的预测结果与置信度
