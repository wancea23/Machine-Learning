import random

n_samples = int(input("Enter the number of samples: "))
n_clusters = int(input("Enter the number of clusters: "))
x_range = int(input("Enter the range of x: "))
y_range = int(input("Enter the range of y: "))
er = 0.0001

X = []
for _ in range(n_samples):
    X.append((random.uniform(0, x_range), random.uniform(0, y_range)))

centroids = []
centroids.append(random.choice(X))

print("Initial Centroid:")
print(centroids)