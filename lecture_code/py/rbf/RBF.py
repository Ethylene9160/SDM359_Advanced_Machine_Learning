
import numpy as np

# import py.model.Model as m

# from model import Model as m
import KMeans as km

class RBF():
    def __init__(self, lr, epochs, n_clusters):
        # super().__init__(lr, epochs)
        self.lr = lr
        self.epochs = epochs
        self.n_clusters = n_clusters
        self.clusters = None

    def train(self, input_data, target, lr=None, epochs=None):
        if lr is None:
            lr = self.lr
        if epochs is None:
            epochs = self.epochs
        # self.input_weights = np.random.randn(input_data.shape(0), n_clusters)
        
        self._get_clusters(input_data, self.n_clusters)
        self.output_weights = np.random.randn(1, self.n_clusters)
        self.output_biases = np.random.rand(1)
        
        for _ in range(epochs):
            # 正向传播
            hidden_outputs = np.array([self.gaussian(input_data, c) for c in self.clusters])
            output = np.dot(self.output_weights, hidden_outputs) + self.output_biases
            print('output: ', output)
            for j in range(len(target)):
                # 计算损失
                loss = target[j] - output[j]
                
                # 反向传播更新参数
                delta_output = -loss
                # 更新输出层权重
                self.output_weights -= lr * np.outer(delta_output, hidden_outputs)
                # 更新输出层偏置
                self.output_biases -= lr * delta_output
        
        # for i in range(self.num_hidden):
        #     self.hidden_weights[i] -= self.lr * delta_hidden[i] * hidden_outputs[i] * (input_data - self.hidden_weights[i]) / self.sigma**2
        #     self.hidden_biases[i] -= self.lr * delta_hidden[i]

    def predict(self, x):
        hidden_outputs = np.array([self.gaussian(x, c) for c in self.n_clusters])
        output = np.dot(self.output_weights, hidden_outputs) + self.output_biases
        return output

    def _get_clusters(self, x_train, n_clusters):
        kmeans = km.KMeans(n_clusters)
        kmeans.train(x_train)
        self.clusters = kmeans.clusters

        
    
    def gaussian(self, x, c):
        sigma = 1.0  # 设定高斯函数的 sigma 值
        return np.exp(-np.linalg.norm(x - c)**2 / (2 * sigma**2))
            

if __name__ == '__main__':
    # 帮我生成一条曲线，y=x^2+x-1，用来训练和测试RBF网络。
    import matplotlib.pyplot as plt

    # 生成数据
    np.random.seed(0)
    X_train = np.linspace(-2, 2, 100).reshape(-1, 1)
    y_train = X_train**2 + X_train - 1 + np.random.normal(0, 0.5, size=X_train.shape)
    X_test = np.linspace(-2, 2, 30).reshape(-1, 1)
    
    y_test = y_train**2 + y_train - 1 + np.random.normal(0, 0.5, size=y_train.shape)

    # 可视化数据
    plt.scatter(X_train, y_train, label='Data')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Data for RBF Network')
    plt.legend()
    plt.show()
    n_clusters = np.array([[0.2,0],[1.1,1.5],[-1.2,-1.4]])
    rbf = RBF(lr = 0.01, epochs = 1000, n_clusters=3)
    rbf.train(X_train, y_train)

    predictions = rbf.predict(x_test)

    plt.figure()
    plt.scatter(X_train, y_train, label='X Data')
    
    plt.scatter(X_test, y_test, label = 'y Data')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Data for RBF Network')
    plt.legend()
    plt.show()


