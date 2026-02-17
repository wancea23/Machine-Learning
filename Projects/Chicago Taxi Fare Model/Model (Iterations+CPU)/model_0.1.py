import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv(r'C:\Users\wance\OneDrive\Code\Python\ML\Projects\Chicago Taxi Fare Model\Dataset\chicago_taxi_train.csv')

x=df[["TRIP_MILES"]].to_numpy()
y=df["FARE"].to_numpy()

w, b = 0, 0
alpha = 0.001
iterations = 100
mse = 0
mse_history = []  # <--- ADD THIS LINE

for i in range(iterations):
    y_pred = w*x+b
    mse = np.mean((y_pred - y)**2)
    mse_history.append(mse)  # <--- ADD THIS LINE
    dw = np.mean(2*(y_pred - y) * x)
    db = np.mean(2*(y_pred - y))
    w -= alpha * dw
    b -= alpha * db


    print(f"Iteration {i+1}: dw={dw:.4f}, w={w:.4f},db={db:.4f}, b={b:.4f}, MSE={mse:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.scatter(x, y, color='blue', label='Actual')
x_line = np.array([x.min(), x.max()])
y_line = w * x_line + b
ax1.plot(x_line, y_line, 'r-', linewidth=2, label=f'Fitted line: y = {w:.2f}x + {b:.2f}')

ax1.set_xlabel('Trip Miles')
ax1.set_ylabel('Fare ($)')
ax1.set_title('Taxi Fare vs Trip Miles (MSE Regression)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: MSE over iterations
ax2.plot(range(1, iterations + 1), mse_history, 'b-', linewidth=2)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('MSE')
ax2.set_title('MSE Convergence')
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')  # Log scale to see improvement better

plt.tight_layout()
plt.show()
