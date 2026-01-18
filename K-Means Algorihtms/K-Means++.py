import random
import math

def distance(p, c):
    return math.sqrt((p[0] - c[0])**2 + (p[1] - c[1])**2)

n_samples = int(input("Enter the number of samples: "))
n_clusters = int(input("Enter the number of clusters: "))
x_range = int(input("Enter the range of x: "))
y_range = int(input("Enter the range of y: "))
er = 0.0001

X = []
for _ in range(n_samples):
    X.append((random.uniform(0, x_range), random.uniform(0, y_range)))

centroids = []
centroids.append(random.choice(X))  #C1

print("\nInitial Centroid:")
print(centroids)

#finds the distance from each point to C1
distances = []
for point in X:
    min_dist = float('inf')
    for c in centroids:
        d = distance(point, c)
        if d < min_dist:
            min_dist = d
    distances.append(min_dist)

#Probability Distribution
distances_squared = []
for d in distances:
    distances_squared.append(d**2)
print("\nSquared Distances:")
print(distances_squared)

total = sum(distances_squared)
print("\nTotal of Squared Distances:")
print(total)

probabilities = []
for d2 in distances_squared:
    probabilities.append(d2 / total)

print("\nProbabilities being the new Centroid:")
print(probabilities)


r = random.uniform(0, 1)
cumulative_probability = 0.0

#Select the second centroid based on the probability distribution
for i in range(len(X)):
    cumulative_probability += probabilities[i]
    if r <= cumulative_probability:
        centroids.append(X[i])  #C2
        break

print("\nSelected Centroid:")
print(centroids)

clusters=[[] for _ in range(n_clusters)]

#Assign points to clusters
for point in X:
    closest_index = 0
    min_dist = float('inf')
    for i in range(len(centroids)):
        d = distance(point, centroids[i])
        if d < min_dist:
            min_dist = d
            closest_index = i
    clusters[closest_index].append(point)

print("\nClusters:")
for i, cluster in enumerate(clusters):
    print(f"Cluster {i}: {cluster}")