import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01):
        self.weights = np.random.rand(input_size + 1)  # +1 for bias
        self.learning_rate = learning_rate

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return 1 if summation > 0 else -1

    def train(self, training_data, labels, epochs):
        errors = []
        for epoch in range(epochs):
            error = 0
            for inputs, label in zip(training_data, labels):
                prediction = self.predict(inputs)
                if prediction != label:
                    error += 1
                    self.weights[1:] += self.learning_rate * label * inputs
                    self.weights[0] += self.learning_rate * label
            errors.append(error)
        return errors

def load_mat_file(file_path):
    data = loadmat(file_path)
    print("Keys in the .mat file:", data.keys())
    
    x = data['X']
    y = data['Y'].ravel()
    
    y[y == 0] = -1
    
    print(f"Loaded features with shape {x.shape} and labels with shape {y.shape}")
    return x, y

def split_data(x, y, test_size=0.2):
    np.random.seed(42)
    mask = np.random.rand(len(y)) < (1 - test_size)
    x_train, x_test = x[mask], x[~mask]
    y_train, y_test = y[mask], y[~mask]
    return x_train, y_train, x_test, y_test

def plot_combined_results(results):
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Perceptron Results for Different Learning Rates', fontsize=16)
    
    colors = ['r', 'g', 'b']
    
    for i, (lr, perceptron, x_test, y_test, errors) in enumerate(results):
        # Plot training error
        axs[0, i].plot(range(1, len(errors) + 1), errors, color=colors[i])
        axs[0, i].set_xlabel('Epochs')
        axs[0, i].set_ylabel('Training Error')
        axs[0, i].set_title(f'Training Error vs. Epochs (LR: {lr})')
        
        # Plot decision boundary
        axs[1, i].scatter(x_test[y_test == 1][:, 0], x_test[y_test == 1][:, 1], color='blue', label='Class 1')
        axs[1, i].scatter(x_test[y_test == -1][:, 0], x_test[y_test == -1][:, 1], color='red', label='Class -1')

        x_min, x_max = x_test[:, 0].min() - 1, x_test[:, 0].max() + 1
        y_min, y_max = x_test[:, 1].min() - 1, x_test[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        z = np.array([perceptron.predict(np.array([x, y])) for x, y in np.c_[xx.ravel(), yy.ravel()]])
        z = z.reshape(xx.shape)

        axs[1, i].contourf(xx, yy, z, alpha=0.4)
        axs[1, i].set_xlabel('Feature 1')
        axs[1, i].set_ylabel('Feature 2')
        axs[1, i].legend()
        axs[1, i].set_title(f'Decision Boundary (Test Data, LR: {lr})')
    
    plt.tight_layout()
    plt.show()

def main():
    file_path = 'dataset2.mat'  # Replace with your actual file path
    x, y = load_mat_file(file_path)
    x_train, y_train, x_test, y_test = split_data(x, y)

    learning_rates = [0.01, 0.1, 0.5]
    epochs = 20
    results = []

    for lr in learning_rates:
        print(f"\nTraining with learning rate: {lr}")
        perceptron = Perceptron(input_size=x.shape[1], learning_rate=lr)
        errors = perceptron.train(x_train, y_train, epochs)

        test_predictions = [perceptron.predict(x) for x in x_test]
        accuracy = np.mean(test_predictions == y_test)
        print(f"Test accuracy: {accuracy:.2f}")

        results.append((lr, perceptron, x_test, y_test, errors))

    plot_combined_results(results)

if __name__ == "__main__":
    main()