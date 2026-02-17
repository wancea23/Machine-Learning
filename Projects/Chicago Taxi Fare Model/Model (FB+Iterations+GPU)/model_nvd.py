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

day_trips = df[df['TRIP_START_HOUR'].between(6, 20)]
night_trips = df[~df['TRIP_START_HOUR'].between(6, 20)]

# 1. Copying the data from the NumPy array to that memory
# 2. Converting to the specified dtype (float32)
# 3. Torch creates new tensors for w and b, allocating the memory on the specified device
# if on gpu - vram if on cpu - ram
x_day = torch.tensor(day_trips[["TRIP_MILES", "TRIP_MINUTES"]].values, dtype=torch.float32, device=device)
y_day = torch.tensor(day_trips["FARE"].values, dtype=torch.float32, device=device)

x_night = torch.tensor(night_trips[["TRIP_MILES", "TRIP_MINUTES"]].values, dtype=torch.float32, device=device)
y_night = torch.tensor(night_trips["FARE"].values, dtype=torch.float32, device=device)

# 1. Creates tensors with a single value 0.0
# 2. Allocating the memory on the specified device if on gpu - vram if on cpu - ram
# 3. Setting requires_grad=True to track operations on these tensors for automatic differentiation
# allocating additional memory to store gradients
w_day = torch.tensor([0.0, 0.0], device=device, requires_grad=True)
b_day = torch.tensor([0.0], device=device, requires_grad=True)

w_night = torch.tensor([0.0, 0.0], device=device, requires_grad=True)
b_night = torch.tensor([0.0], device=device, requires_grad=True)

alpha = 0.00001 # Learning rate
err = 0.0000001
i = 0
mse_day_history = []
mse_night_history = []
day_converged = False
night_converged = False

if device.type == "cuda":
    print(f"Training on {torch.cuda.get_device_name(0)} with PyTorch:")
else:
    print("Training on CPU with PyTorch:")
print("-" * 60)

while True:
    # After every iterations clears the gradients values stored previously
    if w_day.grad is not None:
        w_day.grad.zero_()
        b_day.grad.zero_()
    if w_night.grad is not None:
        w_night.grad.zero_()
        b_night.grad.zero_()

    # x shape: [n, 5], w shape: [5], result: [n]
    y_pred_day = torch.mv(x_day, w_day) + b_day # Matrix-vector multiplication
    errors_day = y_pred_day - y_day
    mse_day = torch.mean(errors_day ** 2)

    y_pred_night = torch.mv(x_night, w_night) + b_night # Matrix-vector multiplication
    errors_night = y_pred_night - y_night
    mse_night = torch.mean(errors_night ** 2)

    # When call .item() on a scalar tensor, pytorch stores the values to 
    # CPU RAM, even if no .cpu() was called, this is what happens under the hood:
    # mse_cpu = mse.cpu()        # 1. Transfers explicit in RAM
    # mse_float = mse_cpu.item() # 2. converts to float Python
    # mse_history.append(mse_float)
    # mse_history.append(mse.cpu().item())  # Același lucru
    mse_day_history.append(mse_day.item())
    mse_night_history.append(mse_night.item())

    total_loss = mse_day + mse_night
    
    if i > 0:  # Need at least 2 iterations to compare
        if not day_converged:
            mse_day_change = abs(mse_day_history[-2] - mse_day_history[-1])
            if mse_day_change < err:
                print(f"\nDay model converged at iteration {i + 1}!")
                print(f"Day MSE change: {mse_day_change:.10f} < {err}")
                day_converged = True
        
        if not night_converged:
            mse_night_change = abs(mse_night_history[-2] - mse_night_history[-1])
            if mse_night_change < err:
                print(f"\nNight model converged at iteration {i + 1}!")
                print(f"Night MSE change: {mse_night_change:.10f} < {err}")
                night_converged = True
        
        if day_converged and night_converged:
            print("\nBoth models have converged!")
            break


    total_loss.backward()  # Compute gradients for w and b auttomatically

    # temporarily disables autograd tracking
    with torch.no_grad():
        w_day -= alpha * w_day.grad
        b_day -= alpha * b_day.grad
        w_night -= alpha * w_night.grad
        b_night -= alpha * b_night.grad
    
    i += 1
    if i % 5000 == 0:
        # Use the correct variables
        w_day_list = w_day.detach().cpu().numpy()
        w_night_list = w_night.detach().cpu().numpy()
        print(f"Iteration {i:5d}:")
        print(f"  Day: w_miles={w_day_list[0]:.4f}, w_min={w_day_list[1]:.4f}, b={b_day.item():.4f}, MSE={mse_day.item():.4f}")
        print(f"  Night: w_miles={w_night_list[0]:.4f}, w_min={w_night_list[1]:.4f}, b={b_night.item():.4f}, MSE={mse_night.item():.4f}")
        print()

# Get final parameters
w_day_final = w_day.detach().cpu().numpy()
b_day_final = b_day.item()
w_night_final = w_night.detach().cpu().numpy()
b_night_final = b_night.item()

print("\n" + "="*60)
print("FINAL MODELS")
print("="*60)
print(f"DAY MODEL: FARE = {w_day_final[0]:.4f} * MILES + {w_day_final[1]:.4f} * MINUTES + {b_day_final:.4f}")
print(f"NIGHT MODEL: FARE = {w_night_final[0]:.4f} * MILES + {w_night_final[1]:.4f} * MINUTES + {b_night_final:.4f}")
print(f"Final Day MSE: {mse_day_history[-1]:.4f}")
print(f"Final Night MSE: {mse_night_history[-1]:.4f}")

# VISUALIZATION
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Day vs Night Taxi Fare Models', fontsize=16)

# Get predictions for plotting
with torch.no_grad():
    y_pred_day_all = (x_day @ w_day + b_day).cpu().numpy()
    y_pred_night_all = (x_night @ w_night + b_night).cpu().numpy()

y_day_cpu = y_day.cpu().numpy()
y_night_cpu = y_night.cpu().numpy()

# PLOT 1: Day Model - Actual vs Predicted
ax1 = axes[0]
ax1.scatter(y_day_cpu, y_pred_day_all, alpha=0.3, s=5, c='blue', label='Day predictions')
ax1.plot([y_day_cpu.min(), y_day_cpu.max()], 
         [y_day_cpu.min(), y_day_cpu.max()], 'r--', linewidth=2, label='Perfect fit')
ax1.set_xlabel('Actual Fare ($)')
ax1.set_ylabel('Predicted Fare ($)')
ax1.set_title(f'Day Model (MSE: {mse_day_history[-1]:.2f})')
ax1.grid(True, alpha=0.3)
ax1.legend()

# PLOT 2: Night Model - Actual vs Predicted
ax2 = axes[1]
ax2.scatter(y_night_cpu, y_pred_night_all, alpha=0.3, s=5, c='orange', label='Night predictions')
ax2.plot([y_night_cpu.min(), y_night_cpu.max()], 
         [y_night_cpu.min(), y_night_cpu.max()], 'r--', linewidth=2, label='Perfect fit')
ax2.set_xlabel('Actual Fare ($)')
ax2.set_ylabel('Predicted Fare ($)')
ax2.set_title(f'Night Model (MSE: {mse_night_history[-1]:.2f})')
ax2.grid(True, alpha=0.3)
ax2.legend()

# PLOT 3: MSE Convergence (both on same plot)
ax3 = axes[2]
ax3.plot(mse_day_history, 'b-', alpha=0.7, label='Day MSE')
ax3.plot(mse_night_history, 'orange', alpha=0.7, label='Night MSE')
ax3.set_xlabel('Iteration')
ax3.set_ylabel('MSE')
ax3.set_title('MSE Convergence Comparison')
ax3.set_yscale('log')
ax3.grid(True, alpha=0.3)
ax3.legend()

plt.tight_layout()
plt.show()

# RANDOM SAMPLE PREDICTIONS
print("\n" + "-"*60)
print("RANDOM SAMPLE PREDICTIONS")
print("-"*60)

with torch.no_grad():
    # Random day samples
    day_indices = torch.randint(0, len(x_day), (5,))
    print("\nDAY TRIPS:")
    for idx in day_indices:
        features = x_day[idx].cpu().numpy()
        true_fare = y_day[idx].item()
        pred_fare = (x_day[idx] @ w_day + b_day).item()
        error = abs(true_fare - pred_fare)
        print(f"  Miles={features[0]:.2f}, Min={features[1]:.1f} → "
              f"Actual=${true_fare:.2f}, Pred=${pred_fare:.2f}, Error=${error:.2f}")
    
    # Random night samples
    night_indices = torch.randint(0, len(x_night), (5,))
    print("\nNIGHT TRIPS:")
    for idx in night_indices:
        features = x_night[idx].cpu().numpy()
        true_fare = y_night[idx].item()
        pred_fare = (x_night[idx] @ w_night + b_night).item()
        error = abs(true_fare - pred_fare)
        print(f"  Miles={features[0]:.2f}, Min={features[1]:.1f} → "
              f"Actual=${true_fare:.2f}, Pred=${pred_fare:.2f}, Error=${error:.2f}")

# MODEL COMPARISON TABLE
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)
print(f"{'Metric':<20} {'DAY':<15} {'NIGHT':<15} {'DIFF':<10}")
print("-" * 60)
print(f"{'Miles weight':<20} {w_day_final[0]:<15.4f} {w_night_final[0]:<15.4f} {w_night_final[0]-w_day_final[0]:<+10.4f}")
print(f"{'Minutes weight':<20} {w_day_final[1]:<15.4f} {w_night_final[1]:<15.4f} {w_night_final[1]-w_day_final[1]:<+10.4f}")
print(f"{'Bias':<20} {b_day_final:<15.4f} {b_night_final:<15.4f} {b_night_final-b_day_final:<+10.4f}")
print(f"{'Final MSE':<20} {mse_day_history[-1]:<15.4f} {mse_night_history[-1]:<15.4f} {mse_night_history[-1]-mse_day_history[-1]:<+10.4f}")
print(f"{'RMSE':<20} ${np.sqrt(mse_day_history[-1]):<14.2f} ${np.sqrt(mse_night_history[-1]):<14.2f} ${np.sqrt(mse_night_history[-1])-np.sqrt(mse_day_history[-1]):<+9.2f}")