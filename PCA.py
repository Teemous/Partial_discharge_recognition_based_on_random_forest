import numpy as np
def pca(X, num_components):
    '''

    :param X: 需要降维的数组
    :param num_components: 降维后的维度
    :return: 降维后的数组
    '''
    # 数据中心化
    mean = np.mean(X, axis=0)
    # 目的是将数据矩阵 X 中的每一列减去该列的均值，使得数据的均值为 0。
    X_centered = X - mean

    # 计算协方差矩阵
    cov_matrix = np.cov(X_centered.T)
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    # 对特征值进行排序并选取前 num_components 个
    idx = np.argsort(eigenvalues)[::-1]
    # np.argsort(eigenvalues) 对特征值进行升序排序，并返回排序后的索引。
    # [::-1] 切片操作将排序后的索引倒序，实现降序排序。
    eigenvalues = eigenvalues[idx]
    # 根据排序后的索引更新特征值，使其按降序排列。
    eigenvectors = eigenvectors[:, idx]
    # 根据排序后的索引更新特征向量，确保特征向量与特征值的对应关系。
    top_eigenvectors = eigenvectors[:, :num_components]
    # 选取前 num_components 个特征向量，形成一个 n x num_components 的矩阵。

    # 数据投影
    X_pca = np.dot(X_centered, top_eigenvectors)
    # np.dot 函数用于矩阵乘法，将中心化的数据矩阵 X_centered 与 top_eigenvectors 相乘。
    # 实现将数据从原始的高维空间投影到由前 num_components 个主成分构成的低维空间。
    # 得到的 X_pca 是一个 m x num_components 的矩阵，表示降维后的数据。
    return X_pca