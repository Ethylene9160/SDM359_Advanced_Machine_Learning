import numpy as np

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

class ConvolutionalNeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_convolutional_layer(self, num_filters, filter_size, input_shape):
        # 初始化卷积核参数
        filter_height, filter_width = filter_size
        input_channels, input_height, input_width = input_shape
        filter_weights = np.random.randn(num_filters, input_channels, filter_height, filter_width)
        filter_biases = np.random.randn(num_filters, 1)
        
        # 将卷积层添加到网络中
        self.layers.append({'type': 'conv', 'weights': filter_weights, 'biases': filter_biases})

    def add_pooling_layer(self, pool_size):
        # 添加池化层到网络中
        self.layers.append({'type': 'pool', 'pool_size': pool_size})

    def add_activation_layer(self, activation_type):
        # 添加激活函数层到网络中
        self.layers.append({'type': 'activation', 'activation_type': activation_type})

    def add_fully_connected_layer(self, num_units):
        # 添加全连接层到网络中
        self.layers.append({'type': 'fc', 'num_units': num_units})

    def forward(self, input_data):
        output = input_data
        for layer in self.layers:
            if layer['type'] == 'conv':
                output = self.convolution_forward(output, layer['weights'], layer['biases'])
            elif layer['type'] == 'pool':
                output = self.pooling_forward(output, layer['pool_size'])
            elif layer['type'] == 'activation':
                output = self.activation_forward(output, layer['activation_type'])
            elif layer['type'] == 'fc':
                output = self.fully_connected_forward(output, layer['num_units'])
        return output

    def convolution_forward(self, input_data, weights, biases):
        num_filters, input_channels, filter_height, filter_width = weights.shape
        input_height, input_width = input_data.shape[-2:]

        output_height = input_height - filter_height + 1
        output_width = input_width - filter_width + 1

        output = np.zeros((num_filters, output_height, output_width))

        for f in range(num_filters):
            for i in range(output_height):
                for j in range(output_width):
                    output[f, i, j] = np.sum(input_data[:, i:i+filter_height, j:j+filter_width] * weights[f]) + biases[f]

        return output
    
    def pooling_forward(self, input_data, pool_size):
        pool_height, pool_width = pool_size
        input_channels, input_height, input_width = input_data.shape

        output_height = input_height // pool_height
        output_width = input_width // pool_width

        output = np.zeros((input_channels, output_height, output_width))

        for c in range(input_channels):
            for i in range(output_height):
                for j in range(output_width):
                    output[c, i, j] = np.max(input_data[c, i*pool_height:(i+1)*pool_height, j*pool_width:(j+1)*pool_width])

        return output


    def activation_forward(self, input_data, activation_type):
        if activation_type == 'relu':
            return np.maximum(0, input_data)
        elif activation_type == 'sigmoid':
            return 1 / (1 + np.exp(-input_data))
        elif activation_type == 'tanh':
            return np.tanh(input_data)
        else:
            raise ValueError("Unknown activation function type")
        
    def fully_connected_forward(self, input_data, num_units):
        input_size = np.prod(input_data.shape)
        input_flattened = input_data.flatten().reshape((input_size, 1))

        weights = np.random.randn(num_units, input_size)
        biases = np.random.randn(num_units, 1)

        output = np.dot(weights, input_flattened) + biases

        return output

    def convolution_backward(self, input_data, weights, biases, output_gradient):
        num_filters, input_channels, filter_height, filter_width = weights.shape
        input_height, input_width = input_data.shape[-2:]

        output_height, output_width = output_gradient.shape[-2:]

        weights_gradient = np.zeros_like(weights)
        biases_gradient = np.zeros_like(biases)
        input_gradient = np.zeros_like(input_data)

        for f in range(num_filters):
            for i in range(output_height):
                for j in range(output_width):
                    input_slice = input_data[:, i:i+filter_height, j:j+filter_width]
                    weights_gradient[f] += input_slice * output_gradient[f, i, j]
                    biases_gradient[f] += output_gradient[f, i, j]
                    input_gradient[:, i:i+filter_height, j:j+filter_width] += weights[f] * output_gradient[f, i, j]

        return input_gradient, weights_gradient, biases_gradient

    def pooling_backward(self, input_data, pool_size, output_gradient):
        pool_height, pool_width = pool_size
        input_channels, input_height, input_width = input_data.shape

        output_height, output_width = output_gradient.shape[-2:]

        input_gradient = np.zeros_like(input_data)

        for c in range(input_channels):
            for i in range(output_height):
                for j in range(output_width):
                    max_idx = np.argmax(input_data[c, i*pool_height:(i+1)*pool_height, j*pool_width:(j+1)*pool_width])
                    idx = np.unravel_index(max_idx, (pool_height, pool_width))
                    input_gradient[c, i*pool_height+idx[0], j*pool_width+idx[1]] = output_gradient[c, i, j]

        return input_gradient

    def activation_backward(self, input_data, activation_type, output_gradient):
        if activation_type == 'relu':
            return (input_data > 0) * output_gradient
        elif activation_type == 'sigmoid':
            return (1 - self.activation_forward(input_data, 'sigmoid')) * self.activation_forward(input_data, 'sigmoid') * output_gradient
        elif activation_type == 'tanh':
            return (1 - self.activation_forward(input_data, 'tanh') ** 2) * output_gradient
        else:
            raise ValueError("Unknown activation function type")
        
    def fully_connected_backward(self, input_data, weights, biases, output_gradient):
        weights_gradient = np.dot(output_gradient, input_data.T)
        biases_gradient = np.sum(output_gradient, axis=1, keepdims=True)
        input_gradient = np.dot(weights.T, output_gradient)

        return input_gradient, weights_gradient, biases_gradient

    def train(self, x_train, y_train, learning_rate=0.01, epochs=1000):
        for epoch in range(epochs):
            total_loss = 0

            for x, y in zip(x_train, y_train):
                # Forward pass
                output = self.forward(x)

                # Compute loss
                loss = mean_squared_error(y, output)
                total_loss += loss

                # Backward pass
                output_gradient = output - y
                for layer in reversed(self.layers):
                    if layer['type'] == 'conv':
                        input_gradient, weights_gradient, biases_gradient = self.convolution_backward(x, layer['weights'], layer['biases'], output_gradient)
                    elif layer['type'] == 'pool':
                        input_gradient = self.pooling_backward(x, layer['pool_size'], output_gradient)
                    elif layer['type'] == 'activation':
                        input_gradient = self.activation_backward(x, layer['activation_type'], output_gradient)
                    elif layer['type'] == 'fc':
                        input_gradient, weights_gradient, biases_gradient = self.fully_connected_backward(x, layer['weights'], layer['biases'], output_gradient)

                    # Update weights and biases
                    if layer['type'] in ['conv', 'fc']:
                        layer['weights'] -= learning_rate * weights_gradient
                        layer['biases'] -= learning_rate * biases_gradient

                # Print loss every 100 samples
                if i % 100 == 0:
                    print(f"Epoch {epoch}, Sample {i}: Loss {loss}")

            # Print average loss for the epoch
            average_loss = total_loss / len(x_train)
            print(f"Epoch {epoch}: Average Loss {average_loss}")


if __name__ == '__main__':
    cnn = ConvolutionalNeuralNetwork()

    # 添加卷积层、池化层、激活函数层和全连接层
    cnn.add_convolutional_layer(num_filters=32, filter_size=(3, 3), input_shape=(3, 32, 32))
    cnn.add_pooling_layer(pool_size=(2, 2))
    cnn.add_activation_layer(activation_type='relu')
    cnn.add_fully_connected_layer(num_units=10)

    # 定义输入数据
    input_data = np.random.randn(3, 32, 32)

    # 定义目标输出数据
    target_output = np.random.randint(0, 10, size=(1,))

    # 进行前向传播计算
    output = cnn.train(input_data, target_output)

    print("Output shape:", output.shape)

    # 计算损失
    loss = mean_squared_error(output, target_output)

    print("Loss:", loss)

    # 进行反向传播计算
    cnn.fully_connected_backward(target_output)

    # 定义测试数据
    test_data = np.random.randn(3, 32, 32)

    # 进行前向传播计算
    test_output = cnn.forward(test_data)

    print("Test output shape:", test_output.shape)
