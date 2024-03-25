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
