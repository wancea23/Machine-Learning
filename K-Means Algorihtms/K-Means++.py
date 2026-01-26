import random
import math
import matplotlib.pyplot as plt

def distance(p, c):
    return math.sqrt((p[0] - c[0])**2 + (p[1] - c[1])**2)

def assign_clusters(X, centroids):
    clusters = [[] for _ in range(n_clusters)]

    for point in X:
        closest_index = 0
        min_dist = float('inf')
        for i in range(len(centroids)):
            d = distance(point, centroids[i])
            if d < min_dist:
                min_dist = d
                closest_index = i
        clusters[closest_index].append(point)
    return clusters

def update_centroids(clusters):
    new_centroids = []
    for cluster in clusters:
        if len(cluster) == 0:
            new_centroids.append((0, 0))
        else:
            x_mean = sum(p[0] for p in cluster) / len(cluster)
            y_mean = sum(p[1] for p in cluster) / len(cluster)
            new_centroids.append((x_mean, y_mean))
    return new_centroids


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

centroids = []
centroids.append(random.choice(X))  # C1

print("\nInitial Centroid:")
print(centroids)

# KMeans++ initialization
while len(centroids) < n_clusters:
    distances = []
    for point in X:
        min_dist = float('inf')
        for c in centroids:
            d = distance(point, c)
            if d < min_dist:
                min_dist = d
        distances.append(min_dist)

    distances_squared = [d**2 for d in distances]
    total = sum(distances_squared)
    probabilities = [d2 / total for d2 in distances_squared]

    r = random.uniform(0, 1)
    cumulative_probability = 0.0

    for i in range(len(X)):
        cumulative_probability += probabilities[i]
        if r <= cumulative_probability:
            centroids.append(X[i])
            break

print("\nInitial Centroids:")
print(centroids)

# KMeans main loop
while True:
    clusters = assign_clusters(X, centroids)
    new_centroids = update_centroids(clusters)

    movement = 0
    for i in range(len(centroids)):
        movement += distance(centroids[i], new_centroids[i])

    if movement < er:
        break

    centroids = new_centroids

print("\nFinal Centroids:")
print(centroids)

print("\nClusters:")
for i, cluster in enumerate(clusters):
    print(f"Cluster {i}: {cluster}")

# -----------------------------
# PLOT THE RESULT WITH COLORS
# -----------------------------
colors = []
for _ in range(n_clusters):
    colors.append((random.random(), random.random(), random.random()))

for i, cluster in enumerate(clusters):
    if cluster:
        plt.scatter([p[0] for p in cluster], [p[1] for p in cluster], color=colors[i], label=f"Cluster {i}")

# Plot centroids
plt.scatter([c[0] for c in centroids], [c[1] for c in centroids],
            color='red', marker='x', s=100, label="Centroids")

plt.title("KMeans++ Clustering")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
