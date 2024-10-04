import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, threshold=0):
        self.weights = np.random.rand(input_size)
        self.learning_rate = learning_rate
        self.threshold = threshold

    def predict_single(self, inputs):
        summation = np.dot(inputs, self.weights)
        return 1 if summation > self.threshold else -1

    def predict(self, inputs):
        return np.array([self.predict_single(x) for x in inputs])

    def train(self, training_data, epochs):
        errors = []
        for epoch in range(epochs):
            error = 0
            for inputs in training_data:
                prediction = self.predict_single(inputs[:-1])  # Use all but last element as input
                actual = 1 if inputs[-1] > 0 else -1  # Use last element to determine actual class
                if prediction != actual:
                    error += 1
                    self.weights += self.learning_rate * (actual - prediction) * inputs[:-1]
            errors.append(error)
        return errors

def load_mat_file(file_path):
    data = loadmat(file_path)
    print("Keys in the .mat file:", data.keys())
    
    x = data['X']
    y = data['Y']
    
    # Combine X and Y into a single array
    combined_data = np.hstack((x, y))
    
    print(f"Loaded data with shape {combined_data.shape}")
    return combined_data

def split_data(data, test_size=0.2):
    np.random.seed(42)
    mask = np.random.rand(len(data)) < (1 - test_size)
    train_data = data[mask]
    test_data = data[~mask]
    return train_data, test_data

def determine_threshold(perceptron, val_data):
    outputs = np.array([np.dot(x[:-1], perceptron.weights) for x in val_data])
    sorted_outputs = np.sort(outputs)
    best_threshold = 0
    best_accuracy = 0
    
    for threshold in sorted_outputs:
        predictions = np.where(outputs > threshold, 1, -1)
        actual = np.where(val_data[:, -1] > 0, 1, -1)
        accuracy = np.mean(predictions == actual)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    return best_threshold

def plot_combined_results(results):
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Perceptron Results for Different Thresholds', fontsize=16)
    
    colors = ['r', 'g', 'b']
    
    for i, (threshold, perceptron, test_data, errors) in enumerate(results):
        # Plot training error
        axs[0, i].plot(range(1, len(errors) + 1), errors, color=colors[i])
        axs[0, i].set_xlabel('Epochs')
        axs[0, i].set_ylabel('Training Error')
        axs[0, i].set_title(f'Training Error vs. Epochs (Threshold: {threshold:.2f})')
        
        # Plot decision boundary
        x_test = test_data[:, :-1]
        y_test = np.where(test_data[:, -1] > 0, 1, -1)
        axs[1, i].scatter(x_test[y_test == 1][:, 0], x_test[y_test == 1][:, 1], color='blue', label='Class 1')
        axs[1, i].scatter(x_test[y_test == -1][:, 0], x_test[y_test == -1][:, 1], color='red', label='Class -1')

        x_min, x_max = x_test[:, 0].min() - 1, x_test[:, 0].max() + 1
        y_min, y_max = x_test[:, 1].min() - 1, x_test[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(xx.shape)

        axs[1, i].contourf(xx, yy, z, alpha=0.4)
        axs[1, i].set_xlabel('Feature 1')
        axs[1, i].set_ylabel('Feature 2')
        axs[1, i].legend()
        axs[1, i].set_title(f'Decision Boundary (Test Data, Threshold: {threshold:.2f})')
    
    plt.tight_layout()
    plt.show()

def main():
    file_path = 'dataset2.mat'  # Use the non-linearly separable dataset
    data = load_mat_file(file_path)
    train_data, test_data = split_data(data, test_size=0.3)
    train_data, val_data = split_data(train_data, test_size=0.2)

    learning_rate = 0.1
    epochs = 10
    results = []

    # Train with threshold = 0
    perceptron_0 = Perceptron(input_size=train_data.shape[1]-1, learning_rate=learning_rate, threshold=0)
    errors_0 = perceptron_0.train(train_data, epochs)
    
    # Determine optimal threshold
    optimal_threshold = determine_threshold(perceptron_0, val_data)
    
    thresholds = [0, optimal_threshold, optimal_threshold * 1.5]  # Test three thresholds

    for threshold in thresholds:
        print(f"\nTraining with threshold: {threshold}")
        perceptron = Perceptron(input_size=train_data.shape[1]-1, learning_rate=learning_rate, threshold=threshold)
        errors = perceptron.train(train_data, epochs)

        test_predictions = perceptron.predict(test_data[:, :-1])
        actual_test = np.where(test_data[:, -1] > 0, 1, -1)
        accuracy = np.mean(test_predictions == actual_test)
        print(f"Test accuracy: {accuracy:.2f}")

        results.append((threshold, perceptron, test_data, errors))

    plot_combined_results(results)

if __name__ == "__main__":
    main()