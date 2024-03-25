
class Model:
    def __init__(self, lr, epochs):
        self.lr = lr
        self.epochs = epochs

    def train(self, x_train, y_train, lr = self.lr, epochs = self.epochs):
        pass

    def predict(self, x):
        pass