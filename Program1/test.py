from math import sqrt  # Importing the sqrt function for computing the square root.
import csv  # Importing the csv module to read/write data from/to CSV files.
import os  # Importing the os module for operating system dependent functionality.


os.chdir('/home/william0friend/fall2023/machine-learning/Program1')

# Function to read country data from a given CSV file.
def read_data_from_file(filename):
    countries = []  # Initialize an empty list to store country data.
    # Open the file in read mode.
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)  # Create a CSV reader object.
        for row in csv_reader:
            # Convert the data into the expected format and append to the list.
            country_data = [row[0]] + [float(value) for value in row[1:]]
            countries.append(country_data)
    return countries  # Return the list of countries.

# Use the function to read country data from 'CPI.txt'.
countries = read_data_from_file('CPI.txt')

# Function to compute the Euclidean distance between two rows of data.
def euclidean_distance(row1, row2):
    distance = 0.0  # Initialize distance to 0.
    # Iterate over each element and compute the sum of squared differences.
    for i in range(len(row1)):
        distance += ((row1[i]) - (row2[i])) ** 2
    return sqrt(distance)  # Return the square root of the summed squared differences.

# Import MinMaxScaler for range normalization.
from sklearn.preprocessing import MinMaxScaler

# Function to normalize the data using Min-Max normalization.
def range_normalize(dataset):
    scaler = MinMaxScaler()  # Create a MinMaxScaler object.
    # Normalize the data excluding country names and CPI.
    scaled_values = scaler.fit_transform([row[1:6] for row in dataset])
    # Combine normalized data with country names and CPI.
    normalized_dataset = [[dataset[i][0]] + list(scaled_values[i]) + [dataset[i][6]] for i in range(len(dataset))]
    return normalized_dataset  # Return the normalized dataset.

# Function to print a table with specified headers.
def print_table(data_with_metrics, headers):
    print(" ".join(headers))  # Print the headers.
    # Print each row in the specified format.
    for row in data_with_metrics:
        print(" ".join(["{:.4f}".format(val) if isinstance(val, float) else val for val in row]))

# Function for 3-nearest neighbor prediction using Euclidean distance.
def kNN_3(countries, target_country):
    # Compute distances between the target country and all other countries.
    distances = [(country[0], euclidean_distance(country[1:6], target_country[1:6]), country[6]) for country in countries]
    distances.sort(key=lambda x: x[1])  # Sort the distances in ascending order.
    
    print_table(distances, ["Country", "Euclid", "CPI"])  # Print the distances.
    
    # Return the mean CPI of the 3 nearest neighbors.
    return sum([row[2] for row in distances[:3]]) / 3

# Function for weighted k-NN prediction using k=16.
def weighted_kNN_16(countries, target_country):
    data_with_weights = []
    for country in countries:
        dist = euclidean_distance(country[1:6], target_country[1:6])  # Compute the Euclidean distance.
        # Compute the weight as the inverse of squared distance.
        weight = 1 / (dist ** 2) if dist != 0 else 0
        # Compute the weighted CPI.
        weighted_cpi = weight * country[6]
        data_with_weights.append((country[0], dist, country[6], weight, weighted_cpi))

    # Sort the data based on distances.
    data_with_weights.sort(key=lambda x: x[1])

    print_table(data_with_weights, ["Country", "Euclid", "CPI", "Weight", "W*CPI"])  # Print the data with weights.
    
    # Compute the weighted sum and total weight.
    weighted_sum = sum([row[4] for row in data_with_weights])
    weight_accumulator = sum([row[3] for row in data_with_weights])
    
    # Return the weighted average CPI.
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
print(f"\nAfter normalization: CPI for 3-NN: {cpi_3nn_normalized:.4f}\n")

# 4. What value would a weighted k-NN prediction model—with k=16 
# (i.e., the full dataset) and using a weighting scheme of the 
# reciprocal of the squared Euclidean distance between the neighbor 
# and the query—return for the CPI of Russia when it is applied 
# to the range-normalized data?
cpi_weighted_16nn_normalized = weighted_kNN_16(normalized_countries, russia_normalized)
print(f"\nAfter normalization: CPI for weighted 16-NN: {cpi_weighted_16nn_normalized:.4f}\n")
