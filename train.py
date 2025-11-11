import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import export_graphviz, plot_tree
from PCA import pca
import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

model_path = './model/model.pkl'
data_path = './data/dataset.csv'
tree_plot_path = './model/tree_visualization.pdf'

pca_flag = False #是否降维

plt_tree = False #是否生成树图

plt.rcParams['font.sans-serif'] = ['SimHei']  # 推荐使用SimHei字体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 加载数据集
dataset = np.loadtxt(data_path, delimiter=',')

# 创建一个 DataFrame
if pca_flag == False:
    X = dataset[:,0:-1] #特征
else:
    X = pca(dataset[:, 0:-1],5)  # 特征

Y = dataset[:,-1] #标签

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=50)

# 创建随机森林分类器
rf_model = RandomForestClassifier(
    n_estimators=100,  # 树的数量
    max_depth=None,    # 树的最大深度
    random_state=50    # 随机种子
)

# 训练模型
rf_model.fit(X_train, y_train)

# 保存模型到文件
with open(model_path, 'wb') as file:
    pickle.dump(rf_model, file)

print(f'模型已保存为 {model_path}')

# 预测测试集
y_pred = rf_model.predict(X_test)
print(f'true label:{y_test}')
print(f'predict:{y_pred}')

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy:.2f}")

# 打印分类报告
print("分类报告:")
print(classification_report(y_test, y_pred))

if plt_tree:
    # 决策树可视化
    # 选择随机森林中的第一棵树
    first_tree = rf_model.estimators_[0]

    # 使用 matplotlib 可视化
    plt.figure(figsize=(20, 10))
    plot_tree(first_tree, feature_names=[f"Feature {i}" for i in range(X.shape[1])],
              class_names=[str(int(cls)) for cls in np.unique(Y)],
              filled=True, rounded=True)
    plt.title("决策树可视化")
    plt.savefig(tree_plot_path)
    plt.show()
    print(f"决策树可视化已保存为 {tree_plot_path}")