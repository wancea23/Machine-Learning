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

distances = []
for point in X:
    min_dist = float('inf')
    for c in centroids:
        d = distance(point, c)
        if d < min_dist:
            min_dist = d
    distances.append(min_dist)

#print("Initial Centroid:")
#print(centroids)