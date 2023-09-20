# from csv import reader
# from math import sqrt
# from sklearn.preprocessing import MinMaxScaler
# import os  # Importing the os module for operating system dependent functionality.


# os.chdir('/home/william0friend/fall2023/machine-learning/Program1')

# # Load a CSV file
# def load_csv(filename):
#     dataset = []
#     with open(filename, 'r') as file:
#         csv_reader = reader(file)
#         for row in csv_reader:
#             if not row:
#                 continue
#             dataset.append(row)
#     return dataset

# # Convert string columns to float
# def str_columns_to_float(dataset, columns):
#     for row in dataset:
#         for column in columns:
#             row[column] = float(row[column])

# # Normalize the dataset
# def normalize_dataset(dataset):
#     scaler = MinMaxScaler()
#     scaled_values = scaler.fit_transform(dataset)
#     return scaled_values.tolist()

# # Calculate the Euclidean distance between two vectors
# def euclidean_distance(row1, row2):
#     distance = 0.0
#     # Only compute distance for numerical columns; skip the country name.
#     for i in range(1, len(row1) - 1):  
#         distance += (row1[i] - row2[i]) ** 2
#     return sqrt(distance)

# # Locate the most similar neighbors
# def get_neighbors(train, test_row, num_neighbors):
#     distances = []
#     for train_row in train:
#         dist = euclidean_distance(test_row, train_row)
#         distances.append((train_row, dist))
#     distances.sort(key=lambda tup: tup[1])
#     neighbors = [distances[i][0] for i in range(num_neighbors)]
#     return neighbors

# # Compute the mean CPI of the nearest neighbors
# def predict_cpi_3nn(train, test_row):
#     neighbors = get_neighbors(train, test_row, 3)
#     return sum([row[6] for row in neighbors]) / 3

# # Compute the weighted average CPI
# def predict_weighted_cpi_16nn(train, test_row):
#     data_with_weights = []
#     for train_row in train:
#         dist = euclidean_distance(test_row, train_row)
#         weight = 1 / (dist ** 2) if dist != 0 else 0
#         weighted_cpi = weight * train_row[6]
#         data_with_weights.append((train_row, dist, weight, weighted_cpi))
    
#     weighted_sum = sum([row[3] for row in data_with_weights])
#     weight_accumulator = sum([row[2] for row in data_with_weights])
#     return weighted_sum / weight_accumulator if weight_accumulator != 0 else 0

# # Execution starts here
# filename = 'CPI.txt'
# dataset = load_csv(filename)
# str_columns_to_float(dataset, range(1, 7))
# normalized_dataset = normalize_dataset([row[1:6] for row in dataset])
# russia_normalized = [0.6099, 0.3754, 0.0948, 0.5658, 0.9058]

# print("#############################################")
# print("Program output")
# print("#############################################")

# cpi_3nn = predict_cpi_3nn(dataset, russia_normalized + [0])
# print(f"\nCPI for 3-NN: {cpi_3nn:.4f}\n")

# cpi_weighted_16nn = predict_weighted_cpi_16nn(dataset, russia_normalized + [0])
# print(f"\nCPI for weighted 16-NN: {cpi_weighted_16nn:.4f}\n")
from csv import reader
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
import os

os.chdir('/home/william0friend/fall2023/machine-learning/Program1')

def load_csv(filename):
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

def str_columns_to_float(dataset, columns):
    for row in dataset:
        for column in columns:
            row[column] = float(row[column])

def normalize_dataset(dataset):
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(dataset)
    return scaled_values.tolist()

def euclidean_distance(row1, row2):
    distance = 0.0
    # Calculate the minimum length of the two rows
    min_length = min(len(row1), len(row2))
    for i in range(1, min_length - 1):
        distance += (row1[i] - row2[i]) ** 2
    return sqrt(distance)


def get_neighbors_with_distances(train, test_row, num_neighbors):
    distances = []
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    return distances[:num_neighbors]

def print_neighbors(neighbors):
    for row in neighbors:
        print(len(row[0]))  # Check length of the list
        print(row[0])       # Print the actual list for inspection
        print(f"{row[0][0]} {row[1]:.4f} {row[0][6]:.4f}")

def predict_cpi_3nn(train, test_row):
    neighbors = get_neighbors_with_distances(train, test_row, 3)
    print_neighbors(neighbors)
    return sum([row[0][6] for row in neighbors]) / 3

def predict_weighted_cpi_16nn(train, test_row):
    data_with_weights = get_neighbors_with_distances(train, test_row, 16)
    
    # Display country, distance, CPI, weight, weighted CPI
    print("Country Euclid CPI Weight W*CPI")
    weighted_sum = 0
    weight_accumulator = 0
    for row in data_with_weights:
        weight = 1 / (row[1] ** 2) if row[1] != 0 else 0
        weighted_cpi = weight * row[0][6]
        weighted_sum += weighted_cpi
        weight_accumulator += weight
        
        print(f"{row[0][0]} {row[1]:.4f} {row[0][6]:.4f} {weight:.4f} {weighted_cpi:.4f}")
    
    return weighted_sum / weight_accumulator if weight_accumulator != 0 else 0

filename = 'CPI.txt'
dataset = load_csv(filename)
str_columns_to_float(dataset, range(1, 7))
normalized_dataset = normalize_dataset([row[1:6] for row in dataset])
russia_normalized = ["Russia",0.6099, 0.3754, 0.0948, 0.5658, 0.9058]

print("#############################################")
print("Program output")
print("#############################################")

cpi_3nn = predict_cpi_3nn(dataset, russia_normalized + [0])
print(f"\nCPI for 3-NN: {cpi_3nn:.4f}\n")

cpi_weighted_16nn = predict_weighted_cpi_16nn(dataset, russia_normalized + [0])
print(f"\nCPI for weighted 16-NN: {cpi_weighted_16nn:.4f}\n")

# For normalized data
cpi_3nn_normalized = predict_cpi_3nn(normalized_dataset, russia_normalized + [0])
print(f"\nAfter normalization: CPI for 3-NN: {cpi_3nn_normalized:.4f}\n")

cpi_weighted_16nn_normalized = predict_weighted_cpi_16nn(normalized_dataset, russia_normalized + [0])
print(f"\nAfter normalization: CPI for weighted 16-NN: {cpi_weighted_16nn_normalized:.4f}\n")
