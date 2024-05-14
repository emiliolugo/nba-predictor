import numpy as np
import matplotlib.pyplot as plt

# Cost function
def cost(coords, w, b):
    mse = 0
    for i in range(len(coords)):
        x_value, y_value = coords[i]
        mse += (y_value - calculate_y_hat(x_value, w, b)) ** 2
    mse /= len(coords)
    return mse

# Gradient descent function
def gradient_descent(coords, w, b, alpha, threshold):
    while True:
        # Compute gradients
        dw = 0
        db = 0
        for i in range(len(coords)):
            x_value, y_value = coords[i]
            dw += -2/len(coords) * (y_value - calculate_y_hat(x_value, w, b)) * x_value
            db += -2/len(coords) * (y_value - calculate_y_hat(x_value, w, b))

        # Update parameters
        tmp_w = w
        tmp_b = b
        w -= alpha * dw
        b -= alpha * db

        # Check for convergence
        if abs(tmp_w - w) < threshold and abs(tmp_b - b) < threshold:
            return w, b

# Prediction function
def calculate_y_hat(x_value, w, b):
    return w * x_value + b

def plot_data_and_fit(coords, w, b, save_location):
    # Plot data points
    for i in range(len(coords)):
        x_value, y_value = coords[i]
        plt.scatter(x_value, y_value, label="Data")

    # Compute line of best fit
    opt_w, opt_b = gradient_descent(coords, w, b, alpha=0.3, threshold=0.001)
    x_values = np.linspace(0, 2, 100)
    y_values = opt_w * x_values + opt_b

    # Plot line of best fit
    plt.plot(x_values, y_values, color='red', label='Line of Best Fit')

    # Add labels and legend
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

    # Save plot
    plt.savefig(save_location)
    plt.show()

# Usage
def main():
    # Sample data
    np.random.seed(0)
    x = 2 * np.random.rand(100, 1)
    y = 4 + 3 * x + np.random.randn(100, 1)

    # Coordinates
    coords = list(zip(x, y))

    # Initial parameters
    w = 0
    b = 0

    # Specify save location
    save_location = '/Users/emiliolugo/nba-predictor/plot.png'
    

    plot_data_and_fit(coords, w, b, save_location)
    mse = cost(coords,w,b)
    print(f"MSE is {mse}")

if __name__ == "__main__":
    main()

    





