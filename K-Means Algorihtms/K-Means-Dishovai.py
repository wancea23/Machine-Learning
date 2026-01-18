import matplotlib.pyplot as plt
import random
import math

def euclidean_distances(point, centroid):
    return math.sqrt((point[0] - centroid[0])**2 + (point[1] - centroid[1])**2)

def kmeans(X, n_clusters, er):
    centroids = []
    for _ in range(n_clusters):
        centroids.append([random.uniform(0, x_range), random.uniform(0, y_range)])

    max_diff = x_range * y_range

    while max_diff > er:
        clusters = [[] for _ in range(n_clusters)]
        for i in X:
            distances = [euclidean_distances(i, centroid) for centroid in centroids]
            closest_centroid = distances.index(min(distances)) #.index() function helps choosing the correct index from the condition
            clusters[closest_centroid].append(i)

        new_centroids = []
        for cluster in clusters:
            if cluster:
                new_centroid = [sum(coord) / len(cluster) for coord in zip(*cluster)]
                new_centroids.append(new_centroid)
            else:
                new_centroids.append([random.uniform(0, x_range), random.uniform(0, y_range)])

        max_diff = max(euclidean_distances(old, new) for old, new in zip(centroids, new_centroids))
        centroids = new_centroids  # Update centroids to the new values

    return centroids, clusters

n_samples = int(input("Enter the number of samples: "))
n_clusters = int(input("Enter the number of clusters: "))
x_range = int(input("Enter the range of x: "))
y_range = int(input("Enter the range of y: "))
er = 0.0001

X = []
for _ in range(n_samples):
    X.append((random.uniform(0, x_range), random.uniform(0, y_range)))

final_centroids, final_clusters = kmeans(X, n_clusters, 0.0001)

for i, cluster in enumerate(final_clusters):
    if cluster:  # Check if the cluster is not empty
        plt.scatter(*zip(*cluster), marker='^', label=f'Cluster {i}')
plt.scatter(*zip(*final_centroids), c='red', marker='x', s=100, label='Centroids')
plt.title("K-means Clustering")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()

