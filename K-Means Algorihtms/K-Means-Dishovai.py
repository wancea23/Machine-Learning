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

n_samples = 21
n_clusters = 5
x_range = 55
y_range = 55
er = 0.0001

X = [
    # Cluster 1 – zona (10–20, 10–25)
    (9,12), (11,18), (14,15), (16,22), (12,20),
    (18,17), (15,24), (10,19), (17,14), (13,16),
    (19,21), (11,13), (14,19), (16,11), (12,23),
    (18,20), (9,17), (20,16), (15,18),

    # Cluster 2 – zona (35–50, 5–20)
    (36,10), (42,18), (48,12), (45,15), (39,8),
    (41,14), (46,9), (44,19), (50,11), (37,16),
    (43,7), (49,17), (40,13), (38,19), (47,14),
    (52,10), (35,15), (51,18), (45,8),

    # Cluster 3 – zona (20–35, 30–50)
    (22,35), (28,42), (31,38), (25,47), (34,44),
    (29,36), (27,49), (33,41), (24,39), (30,46),
    (21,43), (26,34), (35,48), (23,37), (32,45),
    (28,50), (36,40), (20,41), (34,33),

    # Cluster 4 – zona (45–65, 30–50)
    (47,34), (52,41), (60,38), (55,47), (49,44),
    (58,36), (63,42), (46,39), (61,48), (53,33),
    (50,45), (57,40), (64,35), (48,50), (59,46),
    (66,41), (54,49), (62,37), (51,43),

    # Cluster 5 – zona (25–45, 60–80)
    (28,62), (34,71), (40,68), (37,75), (31,66),
    (43,73), (29,79), (45,64), (35,70), (27,74),
    (42,77), (33,63), (38,80), (44,69), (30,72),
    (41,65), (36,78), (26,67), (39,76),
]

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

