import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('chicago_taxi_train.csv')

# x will have shape: (number_of_samples, 3)
# Example: x[0] = [seconds_value, miles_value, speed_value] for first taxi ride
x = df[['TRIP_SECONDS', 'TRIP_MILES', "TRIP_SPEED", "TIPS", "TIP_RATE"]].to_numpy()
# Selects the 'FARE' column and converts it to a 1D NumPy array
# y contains the fare amounts for each taxi ride
y = df['FARE'].to_numpy()

# fig: The entire figure/window that contains everything
# axes: List of 3 individual plot areas (since 1 row × 3 columns = 3 plots)
fig, axes = plt.subplots(1, 5, figsize=(18, 5))

for i in range(x.shape[1]):  #x.shape[1] loops through number of columns [0] through rows
    # Get the current subplot (axis) to work with
    ax=axes[i]
     # x[:, i]: All rows, column i → gets one feature column (e.g., all TRIP_SECONDS values)
    ax.scatter(x[:,i], y, color='blue', marker='o', s=10, alpha=0.5, edgecolors='none')
    ax.set_xlabel(["Trip_Seconds", "Trip_Miles", "Trip_Speed", "TIPS", "TIP_RATE"][i])
    ax.set_ylabel("Fare")
    ax.set_title(f"{['Trip_Seconds', 'Trip_Miles', 'Trip_Speed', 'TIPS', 'TIP_RATE'][i]} vs Fare")

# Automatically adjust spacing between subplots to prevent overlapping
plt.tight_layout()
plt.show()
