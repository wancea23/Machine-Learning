import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

df = pd.read_csv(r'C:\Users\johns\OneDrive\Code\Python\ML\Projects\Chicago Taxi Fare Model\Dataset\chicago_taxi_train.csv')
df['TRIP_MINUTES'] = df['TRIP_SECONDS'] / 60

# 1. Copying the data from the NumPy array to that memory
# 2. Converting to the specified dtype (float32)
# 3. Torch creates new tensors for w and b, allocating the memory on the specified device
# if on gpu - vram if on cpu - ram
x = torch.tensor(df[["TRIP_MILES", "TRIP_MINUTES"]].values, dtype=torch.float32, device=device)
y = torch.tensor(df["FARE"].values, dtype=torch.float32, device=device)

# 1. Creates tensors with a single value 0.0
# 2. Allocating the memory on the specified device if on gpu - vram if on cpu - ram
# 3. Setting requires_grad=True to track operations on these tensors for automatic differentiation
# allocating additional memory to store gradients
w = torch.tensor([0.0, 0.0], device=device, requires_grad=True)
b = torch.tensor([0.0], device=device, requires_grad=True)

alpha = 0.00001 # Learning rate
err = 0.0000001
i = 0
mse_history = []

if device.type == "cuda":
    print(f"Training on {torch.cuda.get_device_name(0)} with PyTorch:")
else:
    print("Training on CPU with PyTorch:")
print("-" * 60)

while True:
    # After every iterations clears the gradients values stored previously
    if w.grad is not None:
        w.grad.zero_()
        b.grad.zero_()

    # x shape: [n, 5], w shape: [5], result: [n]
    y_pred = torch.mv(x, w) + b # Matrix-vector multiplication
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
    if (i) % 1000 == 0:
        # Convert weights to list for easy formatting
        w_list = w.detach().cpu().numpy()
        print(f"Iteration {i:3d}: w={w_list}, b={b.item():.4f}, MSE={mse.item():.4f}")

print("-" * 60)
print(f"Final model: FARE = {w_list[0]:.4f} * TRIP_MILES + {w_list[1]:.4f} * TRIP_MINUTES + {b.item():.4f}")
print(f"Final MSE: {mse_history[-1]:.4f}")

print("\nModel trained successfully!")
print(f"Used device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Plotting
y_cpu = y.cpu().numpy()
y_pred_cpu = y_pred.cpu().detach().numpy()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot actual vs predicted
ax1.scatter(y_cpu, y_pred_cpu, alpha=0.5, s=10)
ax1.plot([y_cpu.min(), y_cpu.max()], [y_cpu.min(), y_cpu.max()], 'r--', linewidth=2)
ax1.set_xlabel('Actual Fare ($)')
ax1.set_ylabel('Predicted Fare ($)')
ax1.set_title('Actual vs Predicted Fare')
ax1.grid(True, alpha=0.3)

# Plot MSE convergence
ax2.plot(range(1, len(mse_history) + 1), mse_history, 'b-', linewidth=2)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('MSE')
ax2.set_title('MSE Convergence')
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')
plt.tight_layout()
plt.show()

# Check Random 10 predictions
print("\n")
print("-" * 60)
print("Random 10 predictions:")
print("-" * 60)

sumerr = 0.0

with torch.no_grad():
    random_indices = torch.randint(0, len(x), (10,))
    for idx in random_indices:
        features = x[idx].cpu().numpy()
        y_true = y[idx].item()
        y_pred = (torch.dot(w, x[idx]) + b).item()
        sumerr += abs(y_true - y_pred)
        
        print(f"Features: Miles={features[0]:.2f}, Min={features[1]:.0f}")
        print(f"  Actual: {y_true:.2f}$, Predicted: {y_pred:.2f}$, Error: {abs(y_true - y_pred):.2f}$")
        print()

print("\nAverage Error on 10 random samples: {:.2f}$".format(sumerr / 10))
print("RMSE: {:.2f}$".format(np.sqrt(mse_history[-1])))