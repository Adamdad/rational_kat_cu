import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf

# Define the complex function model
def complex_function(x, a0, a1, a2, a3, a4, a5, b1, b2, b3, b4):
    numerator = a0 + a1 * x + a2 * (x**2) + a3 * (x**3) + a4 * (x**4) + a5 * (x**5)
    denominator = 1 + np.abs(b1 * x + b2 * (x**2) + b3 * (x**3) + b4 * (x**4))
    return (numerator / denominator)

# Define the activation functions
def relu(x):
    return np.maximum(0, x)

def gelu(x):
    return x * 0.5 * (1 + erf(x / np.sqrt(2)))

def silu(x):  # also known as Swish
    return x / (1 + np.exp(-x))

def mish(x):
    return x * np.tanh(np.log(1 + np.exp(x)))

def GEGLU(x):
    return gelu(x) * x

def ReGLU(x):
    return relu(x) * x 

def Swish(x):
    return x / (1 + np.exp(-x))

def SwishGLU(x):
    return Swish(x) * x

def  erfc_Softplus_2(x):
    # erfc(Softplus(x))2
    return (1 - erf(np.log(1 + np.exp(x))))**2

# Plotting enhancements
def plot_results(x_data, y_data, y_fitted, function_name):
    label_size = 24
    legend_size = 18
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, 'r-', label=f'{function_name} Function', linewidth=4)
    plt.plot(x_data, y_fitted, 'b--', label='Fitted Rational', linewidth=3)

    # Add details for better readability and presentation
    # plt.title(f'Fitting Complex Function to {function_name}', fontsize=16)
    plt.xlabel('x', fontsize=label_size)
    plt.ylabel('y', fontsize=label_size)
    plt.legend(fontsize=label_size)
    plt.grid(True)  # Turn on grid
    plt.tight_layout()  # Adjust layout to not cut off elements

    # Increase tick font size
    plt.xticks(fontsize=legend_size)
    plt.yticks(fontsize=legend_size)

    # plt.show()
    plt.savefig(f'{function_name}_fit.pdf')


# Function to fit and plot
def fit_and_plot_activation(function_name):
    # Select the activation function
    activation_functions = {
        'ReLU': relu,
        'GELU': gelu,
        'SiLU': silu,
        'Mish': mish,
        'GEGLU': GEGLU,
        'ReGLU': ReGLU,
        'Swish': Swish,
        'SwishGLU': SwishGLU,
        'erfc_Softplus_2': erfc_Softplus_2
    }
    
    if function_name not in activation_functions:
        print("Invalid function name. Choose 'ReLU', 'GELU', or 'SiLU'.")
        return

    activation_func = activation_functions[function_name]

    # Generate sample data
    x_data = np.linspace(-3, 3, 1000)
    y_data = activation_func(x_data)

    # Initial parameter guesses
    initial_guesses = [0, 1, 0, 0, 0, 1, 0, 0, 0, 0]

    # Fit the complex function to the activation function data
    try:
        popt, pcov = curve_fit(complex_function, x_data, y_data, p0=initial_guesses)
        print(f"Fitted coefficients for {function_name}: {popt}")
    except Exception as e:
        print(f"An error occurred during fitting: {e}")
        return

    # Generate y values from the fitted model
    y_fitted = complex_function(x_data, *popt)

    # Enhanced plotting function
    plot_results(x_data, y_data, y_fitted, function_name)
    
# Example usage
fit_and_plot_activation('ReLU')
fit_and_plot_activation('GELU')
# fit_and_plot_activation('SiLU')
fit_and_plot_activation('Swish')
# fit_and_plot_activation('Mish')
fit_and_plot_activation('GEGLU')
fit_and_plot_activation('ReGLU')
fit_and_plot_activation('SwishGLU')
fit_and_plot_activation('erfc_Softplus_2')

