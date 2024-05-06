import numpy as np
from sklearn.metrics import pairwise_distances

def silhouette_coefficient(X, labels):
    """
    计算轮廓系数。

    :param X: 数据点的数组。
    :param labels: 每个数据点的聚类标签数组。
    :return: 所有数据点轮廓系数的平均值。
    """
    # 计算所有点的成对距离
    distances = pairwise_distances(X)
    silhouette_scores = []

    # 遍历所有数据点
    for i in range(len(X)):
        # 同一聚类内的距离
        same_cluster = labels == labels[i]
        a = np.mean(distances[i, same_cluster])

        # 不同聚类的距离和标签
        other_clusters = [labels == label for label in set(labels) if label != labels[i]]
        b = min([np.mean(distances[i, other]) for other in other_clusters])

        # 计算轮廓系数
        s = (b - a) / max(a, b)
        silhouette_scores.append(s)

    # 返回所有轮廓系数的平均值
    return np.mean(silhouette_scores)


def soft_silhouette_coefficient(X, membership_degrees, centroids):
    """
    计算软聚类的轮廓系数。

    :param X: 数据点的数组。
    :param membership_degrees: 每个数据点对于每个聚类的隶属度矩阵。
    :param centroids: 聚类中心点的数组。
    :return: 所有数据点轮廓系数的平均值。
    """
    # 计算所有点的成对距离
    distances = pairwise_distances(X, centroids)

    # 计算每个点的a和b值
    a_values = np.sum(membership_degrees * distances, axis=1) / np.sum(membership_degrees, axis=1)

    # 初始化b_values为无穷大
    b_values = np.full(shape=a_values.shape, fill_value=np.inf)

    # 计算每个点的b值
    for i in range(b_values.shape[0]):
        # 排除自己所在的聚类
        for j in range(centroids.shape[0]):
            if j != np.argmax(membership_degrees[i]):
                b_values[i] = min(b_values[i], distances[i, j])

    # 计算轮廓系数
    silhouette_scores = (b_values - a_values)

class KMeans:
    def __init__(self, k=3, epochs=300, tol=1e-6):
        self.k = k
        self.epochs = epochs
        self.tol = tol
        self.centroids = np.array([[]])

    def cal_dis(self, x,y):
        d = 0.0
        for i in range(len(x)):
            d += (x[i]-y[i])**2
        return d**0.5

    def calculate_distances(self, X, centroids):
        # 手动计算欧几里得距离矩阵
        distances = np.zeros((X.shape[0], len(centroids)))
        for i, x in enumerate(X):
            for j, c in enumerate(centroids):
                distances[i, j] = np.sqrt(np.sum((x - c) ** 2))
        return distances

    def fit(self, X):
        if self.k > len(X):
            raise ValueError("k cannot be greater than the number of data points.")

        # 初始化质心，随机选取输入样本。
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]
        for i in range(self.epochs):
            # 分配样本到最近的质心
            distances = self.calculate_distances(X, self.centroids)
            labels = np.argmin(distances, axis=1)

            # 计算新的质心
            new_centroids = []
            for j in range(self.k):
                # 获取属于质心j的所有点
                assigned_points = X[labels == j]
                # 如果有点分配给质心，则计算平均值；否则重新选择一个点作为质心
                if assigned_points.size > 0:
                    new_centroids.append(assigned_points.mean(axis=0))
                else:
                    new_centroids.append(X[np.random.choice(X.shape[0])])

            # 将列表转换为NumPy数组
            new_centroids = np.array(new_centroids)

            # 检查质心是否变化很小
            if np.all(np.abs(new_centroids - self.centroids) <= self.tol):
                break

            self.centroids = new_centroids

        self.labels_ = labels
        return self

    def predict(self, X):
        distances = self.calculate_distances(X, self.centroids)
        return np.argmin(distances, axis=1)

    def loss(self, X):
        distances = np.min(self.calculate_distances(X, self.centroids), axis=1)
        return np.sum(distances ** 2)


class KMeansPlus(KMeans):
    def __init__(self, k=3, epochs=300, tol=1e-6):
        super().__init__(k, epochs, tol)

    def init_centroids(self, X):
        # 随机选择第一个质心
        centroids = [X[np.random.randint(X.shape[0])]]
        # print('in kmeans plus(base function):')
        # print(centroids)
        for _ in range(1, self.k):
            # 计算每个点到最近质心的距离
            distances = np.min(self.calculate_distances(X, centroids), axis=1)

            # 选择下一个质心的概率与距离平方成正比
            probabilities = distances ** 2
            #             probabilities /= probabilities.sum()
            #             cumulative_probabilities = np.cumsum(probabilities)

            # 直接选择最大概率对应的点作为新质心
            next_centroid_idx = np.argmax(probabilities)
            centroids.append(X[next_centroid_idx])


            # 随机选择下一个质心
        #             r = np.random.rand()
        #             next_centroid_idx = np.searchsorted(cumulative_probabilities, r)
        #             centroids.append(X[next_centroid_idx])

        self.centroids = centroids

    def fit(self, X ,n_init = 10):
        best_inertia = np.inf
        best_centroids = None
        best_labels = None

        for _ in range(n_init):
            self.init_centroids(X)
            super().fit(X)
            inertia = self.loss(X)

            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = self.centroids
                best_labels = self.labels_

        self.centroids = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        return self


class SoftKMeans(KMeansPlus):
    def __init__(self, k=3, epochs=300, tol=1e-6, beta=1.0):
        super().__init__(k, epochs, tol)
        self.beta = beta  # Soft assignment parameter

    def calculate_responsibilities(self, X):
        # # 计算每个点到每个质心的距离
        # distances = self.calculate_distances(X, self.centroids)
        # # 计算软分配概率（责任）
        # # 使用负的beta值乘以距离来计算指数
        # exp_distances = np.exp(-self.beta * distances)
        # # 计算归一化的责任
        # responsibilities = exp_distances / np.sum(exp_distances, axis=1, keepdims=True)
        responsibilities = np.zeros((len(X), self.k))
        for i in range(len(X)):
            # distances = np.sum((X[i] - self.centroids)**2, axis=1)
            distances = np.linalg.norm(X[i] - self.centroids, axis=1)
            for k in range(self.k):
                numerator = distances[k]
                if numerator == 0:
                    responsibilities[i] = np.zeros(self.k)
                    responsibilities[i, k] = 1
                    continue
                denominator = np.sum((numerator / distances) ** (2 / (self.beta - 1)))
                responsibilities[i, k] = 1 / denominator
        return responsibilities

        # return responsibilities

    def update_centroids(self, X, responsibilities):
        # 使用责任值作为权重来更新质心
        new_centroids = np.dot(responsibilities.T, X) / np.sum(responsibilities, axis=0)[:, np.newaxis]
        return new_centroids

    def fit(self, X, n_init = 10):
        # self.centroids = self.init_centroids(X)
        self.init_centroids(X)
        # self.centroids = np.array([[-2,0],[2,0]])
        # print("init centroids are:")
        # print(self.centroids)
        for i in range(self.epochs):
            # 计算每个点的责任
            responsibilities = self.calculate_responsibilities(X)
            # 更新质心
            new_centroids = self.update_centroids(X, responsibilities)

            # 检查质心是否变化很小
            if np.all(np.abs(new_centroids - self.centroids) <= self.tol):
                break

            self.centroids = new_centroids

        # 最终的责任用于确定每个点最有可能属于哪个簇
        self.labels_ = np.argmax(responsibilities, axis=1)
        return self

class EnhancedKMeans(KMeansPlus):
    def __init__(self, k=3, epochs=300, tol=1e-6, merge_threshold=0.5, split_threshold=2.0):
        super().__init__(k, epochs, tol)
        self.merge_threshold = merge_threshold
        self.split_threshold = split_threshold

    def merge_centroids(self):
        merged = False
        new_centroids = []
        skip = set()
        for i in range(len(self.centroids)):
            if i in skip:
                continue
            for j in range(i + 1, len(self.centroids)):
                if j in skip:
                    continue
                if np.linalg.norm(self.centroids[i] - self.centroids[j]) < self.merge_threshold:
                # if self.cal_dis(self.centroids[i], self.centroids[j]) < self.merge_threshold:
                    new_centroid = (self.centroids[i] + self.centroids[j]) / 2
                    new_centroids.append(new_centroid)
                    skip.add(j)
                    skip.add(i)
                    merged = True
                    break

            if i not in skip:
                new_centroids.append(self.centroids[i])
        if merged:
            self.centroids = np.array(new_centroids)

    def split_centroids(self, X, labels):
        split = False
        new_centroids = list(self.centroids)
        for i, centroid in enumerate(self.centroids):
            assigned_points = X[labels == i]
            # 确保有足够的点来进行分裂
            if assigned_points.size == 0 or len(assigned_points) < 2:
                continue
            avg_distance = np.mean(np.linalg.norm(assigned_points - centroid, axis=1))
            if avg_distance > self.split_threshold:
                # 由于我们已经检查了assigned_points的长度，可以安全地进行采样
                split_points = assigned_points[np.random.choice(len(assigned_points), 2, replace=False)]
                new_centroids.append(split_points[0])
                new_centroids.append(split_points[1])
                # 直接基于索引删除元素
                del new_centroids[i]
                split = True
                break  # 分裂后重新开始

        if split:
            self.centroids = np.array(new_centroids)

    def fit(self, X):
        super().fit(X)
        # 执行合并和分裂移动
        self.merge_centroids()
        self.split_centroids(X, self.labels_)
            # super().fit(X)
        # 重新进行拟合，以考虑合并和分裂后的变化
        if len(self.centroids) != self.k:
            self.k = len(self.centroids)
            self.fit(X)

    def randFit(self,X):
        super().fit(X)
        # 执行合并和分裂移动
        # for _ in range(5):
        self.merge_centroids()
        self.split_centroids(X, self.labels_)
        for _ in range(30):
            self.k = len(self.centroids)
            self.merge_centroids()
            self.split_centroids(X, self.labels_)
            super().fit(X)

class EnhancedSoftKMeans(SoftKMeans):
    def __init__(self, k=3, epochs=300, tol=1e-6, beta=1.0, merge_threshold=0.5, split_threshold=2.0):
        super().__init__(k, epochs, tol, beta)
        self.merge_threshold = merge_threshold
        self.split_threshold = split_threshold

    def merge_centroids(self):
        merged = False
        new_centroids = []
        skip = set()
        for i in range(len(self.centroids)):
            if i in skip:
                continue
            for j in range(i + 1, len(self.centroids)):
                if j in skip:
                    continue
                # if np.linalg.norm(self.centroids[i] - self.centroids[j]) < self.merge_threshold:
                if self.cal_dis(self.centroids[i], self.centroids[j]) < self.merge_threshold:
                    new_centroid = (self.centroids[i] + self.centroids[j]) / 2
                    new_centroids.append(new_centroid)
                    skip.add(j)
                    skip.add(i)
                    merged = True
                    break

            if i not in skip:
                new_centroids.append(self.centroids[i])
        if merged:
            self.centroids = np.array(new_centroids)

    def split_centroids(self, X, responsibilities):
        split = False
        new_centroids = list(self.centroids)
        for i, centroid in enumerate(self.centroids):
            # 确保有足够的点来进行分裂
            if np.sum(responsibilities[:, i]) == 0:
                continue
            avg_distance = np.mean(np.linalg.norm(X - centroid, axis=1))
            if avg_distance > self.split_threshold:
                # 由于已经检查了assigned_points的长度，可以安全地进行采样
                split_points = X[np.random.choice(len(X), 2, replace=False)]
                new_centroids.append(split_points[0])
                new_centroids.append(split_points[1])
                # 直接基于索引删除元素
                del new_centroids[i]
                split = True
                break

    def fit(self, X, n_init = 10):
        super().fit(X)
        # 执行合并和分裂移动
        self.merge_centroids()
        self.split_centroids(X, self.calculate_responsibilities(X))
        # 重新进行拟合，以考虑合并和分裂后的变化
        if len(self.centroids) != self.k:
            self.k = len(self.centroids)
            self.fit(X)

class SoftKMeansForAss(SoftKMeans):
    def __init__(self, k=3, epochs=300, tol=1e-6, beta=1.0, eta = 1.0):
        super().__init__(k, epochs, tol, beta)
        self.eta = eta

    def calculate_responsibilities(self, X):
        # 计算每个点到每个质心的距离
        distances = self.calculate_distances(X, self.centroids)**2
        # 计算软分配概率（责任）
        # 使用负的beta值乘以距离来计算指数
        exp_distances = np.exp(-self.beta * distances)
        # 计算归一化的责任
        # responsibilities = exp_distances / np.sum(exp_distances, axis=1, keepdims=True)

        responsibilities = 1/(1+(distances/self.eta)**(1/(self.beta-1)))
        return responsibilities

    def fit_possibilities(self, X, n_init = 10, init_centroids = None):
        return self.fit(X, n_init, init_centroids)

    def fit_fuzzy(self, X, n_init = 10, init_centroids = None):
        if init_centroids is not None:
            self.centroids = init_centroids
        else:
            self.init_centroids(X)
        # self.centroids = np.array([[-2,0],[2,0]])
        # print("init centroids are:")
        # print(self.centroids)
        responsibitilies = None

        for i in range(self.epochs):
            # 计算每个点的责任
            responsibilities = super().calculate_responsibilities(X)
            # 更新质心
            new_centroids = self.update_centroids(X, responsibilities)

            # 检查质心是否变化很小
            if np.all(np.abs(new_centroids - self.centroids) <= self.tol):
                break

            self.centroids = new_centroids
            print('centroids are:', self.centroids)

        # 最终的责任用于确定每个点最有可能属于哪个簇
        self.labels_ = np.argmax(responsibilities, axis=1)
        return self
    def fit(self, X, n_init = 10, init_centroids = None):
        # self.centroids = self.init_centroids(X)
        if init_centroids is not None:
            self.centroids = init_centroids
        else:
            self.init_centroids(X)
        # self.centroids = np.array([[-2,0],[2,0]])
        # print("init centroids are:")
        # print(self.centroids)
        responsibitilies = None

        for i in range(self.epochs):
            # 计算每个点的责任
            responsibilities = self.calculate_responsibilities(X)
            # 更新质心
            new_centroids = self.update_centroids(X, responsibilities)

            # 检查质心是否变化很小
            if np.all(np.abs(new_centroids - self.centroids) <= self.tol):
                break

            self.centroids = new_centroids
            print('centroids are:', self.centroids)

        # 最终的责任用于确定每个点最有可能属于哪个簇
        self.labels_ = np.argmax(responsibilities, axis=1)
        return self
