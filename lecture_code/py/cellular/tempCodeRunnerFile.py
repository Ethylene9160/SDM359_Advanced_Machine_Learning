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