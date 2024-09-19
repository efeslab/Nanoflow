import pandas as pd

def f(x):
    return x**2 +10 # Example function, replace with your actual function

def binary_search_f(threshold):
    low = 1
    high = 50
    f0 = f(0)
    intermediate_points = []

    while low <= high:
        mid = (low + high) // 2
        f_mid = f(mid)
        intermediate_points.append((mid, f_mid))
        
        if f_mid - f0 > threshold:
            high = mid - 1
        else:
            low = mid + 1

    return low, intermediate_points

def evaluate_larger_x(starting_x, step_size, threshold):
    x = starting_x
    f0 = f(0)
    larger_x_points = []

    while f(x) <= 4 * f0:
        f_x = f(x)
        larger_x_points.append((x, f_x))
        x += step_size

    return larger_x_points

threshold = 10
result, intermediate_points = binary_search_f(threshold)

# Calculate the step size as the range of intermediate points divided by 5
step_size = (max(intermediate_points, key=lambda point: point[0])[0] - min(intermediate_points, key=lambda point: point[0])[0]) // 5

# Evaluate larger x values until f(x) > 4 * f(0)
larger_x_points = evaluate_larger_x(result, step_size, 4 * f(0))

# Save intermediate points and larger x points to dataframes
df_intermediate = pd.DataFrame(intermediate_points, columns=['x', 'f(x)'])
df_larger_x = pd.DataFrame(larger_x_points, columns=['x', 'f(x)'])

# Save the dataframes to CSV files
df_intermediate.to_csv('intermediate_points.csv', index=False)
df_larger_x.to_csv('larger_x_points.csv', index=False)

# Print the results
print(f"The first point where f(x) - f(0) > {threshold} is at x = {result}")
print("Intermediate points:")
print(df_intermediate)
print("Larger x points:")
print(df_larger_x)
