import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from tabulate import tabulate
from mpl_toolkits.mplot3d import Axes3D
import pdfkit

# L values
L_values = [0.03, 0.1, 0.3, 1.0]
# epsilon values
epsilon_values = [0.1, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 
            2.25, 2.5, 2.75, 3.0, 3.25, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0]
# T values
T_values = [0.000, 0.025, 0.050, 0.075, 0.100, 0.125,
            0.150, 0.175, 0.200, 0.225, 0.250, 0.275,
            0.300, 0.325, 0.350, 0.375, 0.400, 0.425,
            0.450, 0.475, 0.500, 0.525, 0.550, 0.575,
            0.600]

# Initialize empty arrays to store the values of x1, x2, x3, and y
X = []
y = []

# Initialize empty 2d array to store the values of x1, x2, x3, and y
data_array = []

# Loop over each folder (each epsilon value)
for i, eps in enumerate(epsilon_values):
    # Find folder for epsilon value
    folder_path = fr"C:\Users\paige\OneDrive\Desktop\Iowa State\Research\Results\MNIST_results_only\mnist_L1_e={eps}"
    if not os.path.exists(folder_path):
        print("Folder Not Found")

    # Read data from text file
    file_path = fr"{folder_path}\mnist_robust_accuracy_argmax.txt"
    if not os.path.exists(file_path):
        print("File Not Found")
    data = np.genfromtxt(file_path, skip_header=2,
                        delimiter=', ', usecols=range(1, len(L_values) + 1))  
    
    # Loop over each T value
    for j, T in enumerate(T_values):
    
        # Loop over each L value
        for k, L in enumerate(L_values):
            y = data[j, k]

            # Add the y value at the L value and T value and epsilon value
            data_array.append([eps, T, L, y])
            
def multiple_polynomial_regression(x, y, degree):
    """
    Performs multiple polynomial regression on the given input features (x) and output values (y) 
    with the specified degree.
    
    Parameters:
    x (array-like): Input features
    y (array-like): Output values
    degree (int): Degree of the polynomial
    
    Returns:
    (tuple): Returns a tuple containing the trained polynomial regression model, the polynomial features,
             and the mean squared error.
    """
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Create polynomial features
    poly = PolynomialFeatures(degree=degree)
    x_poly = poly.fit_transform(x_train)
    
    # Create a polynomial regression model and fit it to the training data
    regressor = LinearRegression()
    regressor.fit(x_poly, y_train)

    # Predict y values for the test data
    x_poly_test = poly.transform(x_test)
    y_pred = regressor.predict(x_poly_test)

    # Calculate mean squared error
    mse = mean_squared_error(y_test, y_pred)

    return regressor, poly, mse


# Split the data into input features (x) and output values (y)
# x = data_array[:, :3]
x = [[row[0], row[1], row[2]] for row in data_array]
# y = data_array[:, 3]
y = [row[3] for row in data_array]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Perform multiple polynomial regression with degree 9
regressor, poly, mse = multiple_polynomial_regression(x_train, y_train, degree=9)

# Evaluate the model using the coefficient of determination (R^2) score
r_squared = regressor.score(poly.transform(x_test), y_test)

# Print the results
print("Results to Deg 9")
print('Mean Squared Error:', mse)
print("R-squared value:", r_squared)

# Print the coefficients for degree 9
degree = 9
for i in range(0, degree+1):
    print(f"coefficients for x^{i}: {regressor.coef_[i]}")
    
# initialize empty list to store table data and average error for each T value
table_data = []
avg_errors = []

# define the degree of polynomial features
degree = 9

# create polynomial features object
poly = PolynomialFeatures(degree)

# create linear regression object
regressor = LinearRegression()

# fit the regressor on data
regressor.fit(x_train, y_train)

# loop over each combination of epsilon, T, and L
for T in T_values:
    for eps in epsilon_values:
        for L in L_values:
            # create polynomial features
            X = poly.fit_transform(np.array([[eps, T, L]]))
            # make prediction
            pred = regressor.predict([[eps, T, L]])[0]
            # get actual y value from your data array
            actual_y = data_array[(T_values.index(T) * len(L_values)) + L_values.index(L)][3]
            # calculate error using mean squared error
            error = mean_squared_error([actual_y], [pred])
            # append to table data
            table_data.append({'epsilon': eps, 'T': T, 'L': L, 'predicted': pred, 'actual': actual_y, 'error': error})

    # calculate and store average error for this T value
    t_df = pd.DataFrame(table_data)
    t_avg_error = t_df['error'].mean()
    avg_errors.append({'T': T, 'Average Error': t_avg_error})

    # format dataframe as plain text table
    table_str = tabulate(t_df, headers='keys', tablefmt='plain')

    # write table to file
    with open(f'T={T}_error.txt_poly', 'w') as f:
        f.write(table_str)

    # clear table data for next T value
    table_data.clear()

# create pandas dataframe from average error data
avg_error_df = pd.DataFrame(avg_errors)

# format average error dataframe as plain text table
avg_error_table_str = tabulate(avg_error_df, headers='keys', tablefmt='plain')

# write average error table to file
with open('average_errors_T_poly.txt', 'w') as f:
    f.write(avg_error_table_str)

# plot average errors for each T value
plt.plot(avg_error_df['T'], avg_error_df['Average Error'])
plt.title('Average Error vs. T')
plt.xlabel('T')
plt.ylabel('Average Error')
plt.show()

# initialize empty list to store table data and average error for each Epsilon value
table_data = []
avg_errors = []

# define the degree of polynomial features
degree = 9

# create polynomial features object
poly = PolynomialFeatures(degree)

# create linear regression object
regressor = LinearRegression()

# fit the regressor on data
regressor.fit(x_train, y_train)

# loop over each combination of epsilon, T, and L
for eps in epsilon_values:
    for T in T_values:
        for L in L_values:
            # create polynomial features
            X = poly.fit_transform(np.array([[eps, T, L]]))
            # make prediction
            pred = regressor.predict([[eps, T, L]])[0]
            # get actual y value from your data array
            actual_y = data_array[(T_values.index(T) * len(L_values)) + L_values.index(L)][3]
            # calculate error using mean squared error
            error = mean_squared_error([actual_y], [pred])
            # append to table data
            table_data.append({'epsilon': eps, 'T': T, 'L': L, 'predicted': pred, 'actual': actual_y, 'error': error})

    # calculate and store average error for this eps value
    eps_df = pd.DataFrame(table_data)
    eps_avg_error = t_df['error'].mean()
    avg_errors.append({'Epsilon': eps, 'Average Error': eps_avg_error})

    # format dataframe as plain text table
    table_str = tabulate(eps_df, headers='keys', tablefmt='plain')

    # write table to file
    # with open(f'Epsilon={eps}_error.txt_poly', 'w') as f:
    #     f.write(table_str)

    # clear table data for next T value
    table_data.clear()

# create pandas dataframe from average error data
avg_error_df = pd.DataFrame(avg_errors)

# format average error dataframe as plain text table
avg_error_table_str = tabulate(avg_error_df, headers='keys', tablefmt='plain')

# write average error table to file
# with open('average_errors_epsilon_poly.txt', 'w') as f:
#     f.write(avg_error_table_str)

# plot average errors for each eps value
plt.plot(avg_error_df['Epsilon'], avg_error_df['Average Error'])
plt.title('Average Error vs. Epsilon')
plt.xlabel('Epsilon')
plt.ylabel('Average Error')
plt.show()

# initialize empty list to store table data and average error for each T value
table_data = []
avg_errors = []

# define the degree of polynomial features
degree = 9

# create polynomial features object
poly = PolynomialFeatures(degree)

# create linear regression object
regressor = LinearRegression()

# fit the regressor on data
regressor.fit(x_train, y_train)

# loop over each combination of epsilon, T, and L
for L in L_values:
    for eps in epsilon_values:
        for T in T_values:
            # create polynomial features
            X = poly.fit_transform(np.array([[eps, T, L]]))
            # make prediction
            pred = regressor.predict([[eps, T, L]])[0]
            # get actual y value from your data array
            actual_y = data_array[(T_values.index(T) * len(L_values)) + L_values.index(L)][3]
            # calculate error using mean squared error
            error = mean_squared_error([actual_y], [pred])
            # append to table data
            table_data.append({'epsilon': eps, 'T': T, 'L': L, 'predicted': pred, 'actual': actual_y, 'error': error})

    # calculate and store average error for this L value
    l_df = pd.DataFrame(table_data)
    l_avg_error = l_df['error'].mean()
    avg_errors.append({'L': L, 'Average Error': l_avg_error})

    # format dataframe as plain text table
    table_str = tabulate(l_df, headers='keys', tablefmt='plain')

    # write table to file
    with open(f'L={L}_error.txt_poly', 'w') as f:
        f.write(table_str)

    # clear table data for next L value
    table_data.clear()

# create pandas dataframe from average error data
avg_error_df = pd.DataFrame(avg_errors)

# format average error dataframe as plain text table
avg_error_table_str = tabulate(avg_error_df, headers='keys', tablefmt='plain')

# write average error table to file
with open('average_errors_L_poly.txt', 'w') as f:
    f.write(avg_error_table_str)

# plot average errors for each L value
plt.plot(avg_error_df['L'], avg_error_df['Average Error'])
plt.title('Average Error vs. L')
plt.xlabel('L')
plt.ylabel('Average Error')
plt.show()