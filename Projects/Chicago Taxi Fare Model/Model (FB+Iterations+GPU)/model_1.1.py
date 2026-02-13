import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

df = pd.read_csv(r'C:\Users\johns\OneDrive\Code\Python\ML\Projects\Chicago Taxi Fare Model\Dataset\chicago_taxi_train.csv')

# 1. Copying the data from the NumPy array to that memory
# 2. Converting to the specified dtype (float32)
# 3. Torch creates new tensors for w and b, allocating the memory on the specified device
# if on gpu - vram if on cpu - ram
x = torch.tensor(df["TRIP_MILES"].values, dtype=torch.float32, device=device)
y = torch.tensor(df["FARE"].values, dtype=torch.float32, device=device)

# 1. Creates tensors with a single value 0.0
# 2. Allocating the memory on the specified device if on gpu - vram if on cpu - ram
# 3. Setting requires_grad=True to track operations on these tensors for automatic differentiation
# allocating additional memory to store gradients
w = torch.tensor(0.0, device=device, requires_grad=True)
b = torch.tensor(0.0, device=device, requires_grad=True)

alpha = 0.007 # Learning rate
err = 0.000001
i = 0
mse_history = []

print(f"Training on {torch.cuda.get_device_name(0)} with PyTorch (YOUR logic):")
print("-" * 60)

while True:
    # After every iterations clears the gradients values stored previously
    if w.grad is not None:
        w.grad.zero_()
        b.grad.zero_()

    y_pred = w * x + b
    errors = y_pred - y
    mse = torch.mean(errors ** 2)
    # When call .item() on a scalar tensor, pytorch stores the values to 
    # CPU RAM, even if no .cpu() was called, this is what happens under the hood:
    # mse_cpu = mse.cpu()        # 1. Transfers explicit in RAM
    # mse_float = mse_cpu.item() # 2. converts to float Python
    # mse_history.append(mse_float)
    # mse_history.append(mse.cpu().item())  # Același lucru
    mse_history.append(mse.item())
    
    if i > 0:  # Need at least 2 iterations to compare
        mse_change = abs(mse_history[-2] - mse_history[-1])
        if mse_change < err:
            print(f"\nConverged at iteration {i + 1}!")
            print(f"MSE change: {mse_change:.10f} < {err}")
            break

    mse.backward()  # Compute gradients for w and b auttomatically

    # temporarily disables autograd tracking
    with torch.no_grad():
        w -= alpha * w.grad
        b -= alpha * b.grad
    
    i += 1
    if (i) % 10 == 0:
        print(f"Iteration {i:3d}: w={w.item():.4f}, b={b.item():.4f}, MSE={mse.item():.4f}")

print("-" * 60)
print(f"Final model: FARE = {w.item():.4f} * TRIP_MILES + {b.item():.4f}")
print(f"Final MSE: {mse_history[-1]:.4f}")

print("\nModel trained successfully!")
print(f"Used device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Plotting
x_cpu = x.cpu().numpy()
y_cpu = y.cpu().numpy()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(x_cpu, y_cpu, color='blue', label='Actual', alpha=0.5, s=10)
x_line = np.array([x_cpu.min(), x_cpu.max()])
y_line = w.item() * x_line + b.item()
ax1.plot(x_line, y_line, 'r-', linewidth=2, label=f'Fitted line: y = {w.item():.2f}x + {b.item():.2f}')
ax1.set_xlabel('Trip Miles')
ax1.set_ylabel('Fare ($)')
ax1.set_title('Taxi Fare vs Trip Miles')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(range(1, len(mse_history) + 1), mse_history, 'b-', linewidth=2)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('MSE')
ax2.set_title('MSE Convergence')
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

plt.tight_layout()
plt.show()