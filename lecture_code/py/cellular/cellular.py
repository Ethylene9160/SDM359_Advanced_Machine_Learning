import numpy as np
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

class CellularNeuronNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 初始化输入层到隐藏层的权重矩阵和偏置向量
        self.weights_input_hidden = np.random.randn(hidden_size, input_size)
        self.bias_hidden = np.random.randn(hidden_size, 1)

        # 初始化隐藏层到输出层的权重矩阵和偏置向量
        self.weights_hidden_output = np.random.randn(output_size, hidden_size)
        self.bias_output = np.random.randn(output_size, 1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        # 隐藏层的输入
        hidden_input = np.dot(self.weights_input_hidden, x) + self.bias_hidden
        # 隐藏层的输出
        hidden_output = self.sigmoid(hidden_input)

        # 输出层的输入
        output_input = np.dot(self.weights_hidden_output, hidden_output) + self.bias_output
        # 输出层的输出
        output = self.sigmoid(output_input)

        return output


    def train(self, x_train, y_train, learning_rate, epochs):
        for epoch in range(epochs):
            # Forward pass
            hidden_input = np.dot(self.weights_input_hidden, x_train.T) + self.bias_hidden
            hidden_output = self.sigmoid(hidden_input)
            output_input = np.dot(self.weights_hidden_output, hidden_output) + self.bias_output
            output = self.sigmoid(output_input)
            
            # Compute loss
            loss = mean_squared_error(y_train.T, output)
            
            # Backward pass
            output_error = (output - y_train.T) * output * (1 - output)
            hidden_error = np.dot(self.weights_hidden_output.T, output_error) * hidden_output * (1 - hidden_output)
            
            # Update weights and biases
            self.weights_hidden_output -= learning_rate * np.dot(output_error, hidden_output.T)
            self.bias_output -= learning_rate * np.sum(output_error, axis=1, keepdims=True)
            self.weights_input_hidden -= learning_rate * np.dot(hidden_error, x_train)
            self.bias_hidden -= learning_rate * np.sum(hidden_error, axis=1, keepdims=True)
            
            # Print loss every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss {loss}")

if __name__ == '__main__':
    # 定义网络的输入、隐藏层大小和输出大小
    input_size = 3
    hidden_size = 4
    output_size = 2

    # 创建一个 Cellular Neuron Network 实例
    cnn = CellularNeuronNetwork(input_size, hidden_size, output_size)

    # 定义训练数据和标签
    x_train = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    y_train = np.array([[0.4, 0.7], [0.3, 0.6], [0.1, 0.8]])

    # 设置学习率和训练轮数
    learning_rate = 0.1
    epochs = 1000

    # 创建网络实例
    cnn = CellularNeuronNetwork(input_size, hidden_size, output_size)

    # 训练网络
    cnn.train(x_train, y_train, learning_rate, epochs)

    # 定义输入数据
    input_data = np.array([0.5, 0.3, 0.8])

    # 进行前向传播计算
    output = cnn.forward(input_data)

    print("Output:", output)
