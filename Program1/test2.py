from math import sqrt
import csv
import os
# hard-coded data

os.chdir('/home/william0friend/fall2023/machine-learning/Program1')

countries = [
    # Format: Country, Life Expectancy, CPI, Democracy, Freedom, Generosity, GDP per capita
    ["Afghanistan", 59.61, 23.21, 74.30, 4.44, 0.40, 1.5171],
    ["Haiti", 45.00, 47.67, 73.10, 0.09, 3.40, 1.7999],
    ["Nigeria", 51.30, 38.23, 82.60, 1.07, 4.10, 2.4493],
    ["Egypt", 70.48, 26.58, 19.60, 1.86, 5.30, 2.8622],
    ["Argentina", 75.77, 32.30, 13.30, 0.76, 10.10, 2.9961],
    ["China", 74.87, 29.98, 13.70, 1.95, 6.40, 3.6356],
    ["Brazil", 73.12, 42.93, 14.50, 1.43, 7.20, 3.7741],
    ["Israel", 81.30, 28.80, 3.60, 6.77, 12.50, 5.8069],
    ["U.S.A", 78.51, 29.85, 6.30, 4.72, 13.70, 7.1357],
    ["Ireland", 80.15, 27.23, 3.50, 0.60, 11.50, 7.5360],
    ["U.K.", 80.09, 28.49, 4.40, 2.59, 13.00, 7.7751],
    ["Germany", 80.24, 22.07, 3.50, 1.31, 12.00, 8.0461],
    ["Canada", 80.99, 24.79, 4.90, 1.42, 14.20, 8.6725],
    ["Australia", 82.09, 25.40, 4.20, 1.86, 11.50, 8.8442],
    ["Sweden", 81.43, 22.18, 2.40, 1.27, 12.80, 9.2985],
    ["NewZealand", 80.67, 27.81, 4.90, 1.13, 12.30, 9.4627]
]

def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)):
        distance += ((row1[i]) - (row2[i])) ** 2
    return sqrt(distance)









# 0. My Normalize # correct for this assignment
def range_normalize(dataset):#_mine(dataset):
    min_vals = [min([row[i] for row in dataset]) for i in range(1, 6)]
    max_vals = [max([row[i] for row in dataset]) for i in range(1, 6)]
    
    normalized_dataset = []
    for row in dataset:
        normalized_row = [row[0]]
        for i in range(1, 6):
            normalized_val = (row[i] - min_vals[i-1]) / (max_vals[i-1] - min_vals[i-1])
            normalized_row.append(normalized_val)
        normalized_row.append(row[6])  # Append CPI without normalization
        normalized_dataset.append(normalized_row)
    
    return normalized_dataset

# # 1. min-max Normalize #  correct for this assignment
# from sklearn.preprocessing import MinMaxScaler

# def range_normalize(dataset):#_minmax(dataset):
#     scaler = MinMaxScaler()
#     # Exclude the country names and CPI from the scaling process
#     scaled_values = scaler.fit_transform([row[1:6] for row in dataset])
#     # Combine country names, scaled values, and CPI back together
#     normalized_dataset = [[dataset[i][0]] + list(scaled_values[i]) + [dataset[i][6]] for i in range(len(dataset))]
#     return normalized_dataset

# # 2. Standard Normalize # not correct for this assignment
# from sklearn.preprocessing import StandardScaler

# def range_normalize(dataset):#_standard(dataset):
#     scaler = StandardScaler()
#     # Exclude the country names and CPI from the scaling process
#     scaled_values = scaler.fit_transform([row[1:6] for row in dataset])
#     # Combine country names, scaled values, and CPI back together
#     normalized_dataset = [[dataset[i][0]] + list(scaled_values[i]) + [dataset[i][6]] for i in range(len(dataset))]
#     return normalized_dataset

# # 3. Robust Normalize # Not correct for this assignment
# from sklearn.preprocessing import RobustScaler

# def range_normalize(dataset):#_robust(dataset):
#     scaler = RobustScaler()
#     # Exclude the country names and CPI from the scaling process
#     scaled_values = scaler.fit_transform([row[1:6] for row in dataset])
#     # Combine country names, scaled values, and CPI back together
#     normalized_dataset = [[dataset[i][0]] + list(scaled_values[i]) + [dataset[i][6]] for i in range(len(dataset))]
#     return normalized_dataset
















# Print the table
def print_table(data_with_metrics, headers):
    # Print the header
    print(" ".join(headers))
    
    for row in data_with_metrics:
        print(" ".join(["{:.4f}".format(val) if isinstance(val, float) else val for val in row]))

# 3-nearest neighbor prediction
def kNN_3(countries, target_country):
    distances = [(country[0], euclidean_distance(country[1:6], target_country[1:6]), country[6]) for country in countries]
    distances.sort(key=lambda x: x[1])
    
    print_table(distances, ["Country", "Euclid", "CPI"])
    
    # Compute mean of 3 nearest neighbors
    return sum([row[2] for row in distances[:3]]) / 3

# Weighted k-NN prediction for k=16
def weighted_kNN_16(countries, target_country):
    data_with_weights = []
    for country in countries:
        dist = euclidean_distance(country[1:6], target_country[1:6])
        weight = 1 / (dist ** 2) if dist != 0 else 0
        weighted_cpi = weight * country[6]
        data_with_weights.append((country[0], dist, country[6], weight, weighted_cpi))

    data_with_weights.sort(key=lambda x: x[1])

    print_table(data_with_weights, ["Country", "Euclid", "CPI", "Weight", "W*CPI"])
    
    weighted_sum = sum([row[4] for row in data_with_weights])
    weight_accumulator = sum([row[3] for row in data_with_weights])
    
    return weighted_sum / weight_accumulator if weight_accumulator != 0 else 0

# # Russia data
russia = ["Russia", 67.62, 31.68, 10.00, 3.87, 12.90]

print("#############################################")
print("Program output")
print("#############################################")

# 1. What value would a 3-nearest neighbor prediction model using Euclidean distance return for the CPI of Russia?
cpi_3nn = kNN_3(countries, russia)
print(f"\nCPI for 3-NN: {cpi_3nn:.4f}\n")

# 2. What value would a weighted k-NN prediction model return for the CPI of Russia? 
# Use k=16 (i.e., the full dataset) and a weighting scheme of the reciprocal of the 
# squared Euclidean distance between the neighbor and the query.
cpi_weighted_16nn = weighted_kNN_16(countries, russia)
print(f"\nCPI for weighted 16-NN: {cpi_weighted_16nn:.4f}\n")

# 3. What value would a 3-nearest neighbor prediction model using Euclidean 
# distance return for the CPI of Russia when the descriptive features 
# have been normalized using range normalization?
normalized_countries = range_normalize(countries)
russia_normalized = ["Russia", 0.6099, 0.3754, 0.0948, 0.5658, 0.9058]
cpi_3nn_normalized = kNN_3(normalized_countries, russia_normalized)
print(f"\nCPI for 3-NN with normalized data: {cpi_3nn_normalized:.4f}\n")

# 4. What value would a weighted k-NN prediction model—with k=16 
# (i.e., the full dataset) and using a weighting scheme of the 
# reciprocal of the squared Euclidean distance between the neighbor 
# and the query—return for the CPI of Russia when it is applied 
# to the range-normalized data?
cpi_weighted_16nn_normalized = weighted_kNN_16(normalized_countries, russia_normalized)
print(f"\nCPI for weighted 16-NN with normalized data: {cpi_weighted_16nn_normalized:.4f}\n")
