import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv(r'C:\Users\johns\OneDrive\Code\Python\ML\Projects\Chicago Taxi Fare Model\Dataset\chicago_taxi_train.csv')

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

# Quick check: Are there negative fares? Zero fares? Outliers?
print(df['FARE'].describe())
print(f"\nNumber of zero fares: {(df['FARE'] == 0).sum()}")
print(f"Number of negative fares: {(df['FARE'] < 0).sum()}")

# Day vs Night trip comparison
day_trips = df[df['TRIP_START_HOUR'].between(6, 20)]
night_trips = df[~df['TRIP_START_HOUR'].between(6, 20)]

print("="*60)
print("DAY TRIPS (6am-8pm) vs NIGHT TRIPS (8pm-6am)")
print("="*60)

print(f"\n{'Metric':<20} {'DAY':<15} {'NIGHT':<15} {'DIFF':<10}")
print("-"*60)

# Average miles
day_miles = day_trips['TRIP_MILES'].mean()
night_miles = night_trips['TRIP_MILES'].mean()
print(f"{'Avg Miles':<20} {day_miles:<15.2f} {night_miles:<15.2f} {night_miles-day_miles:<+10.2f}")

# Average minutes
day_min = (day_trips['TRIP_SECONDS']/60).mean()
night_min = (night_trips['TRIP_SECONDS']/60).mean()
print(f"{'Avg Minutes':<20} {day_min:<15.2f} {night_min:<15.2f} {night_min-day_min:<+10.2f}")

# Average fare
day_fare = day_trips['FARE'].mean()
night_fare = night_trips['FARE'].mean()
print(f"{'Avg Fare':<20} ${day_fare:<14.2f} ${night_fare:<14.2f} ${night_fare-day_fare:<+9.2f}")

# Fare per mile
day_rate = (day_trips['FARE'] / day_trips['TRIP_MILES']).mean()
night_rate = (night_trips['FARE'] / night_trips['TRIP_MILES']).mean()
print(f"{'Avg $/mile':<20} ${day_rate:<14.2f} ${night_rate:<14.2f} ${night_rate-day_rate:<+9.2f}")

# Fare per minute
day_rate_min = (day_trips['FARE'] / (day_trips['TRIP_SECONDS']/60)).mean()
night_rate_min = (night_trips['FARE'] / (night_trips['TRIP_SECONDS']/60)).mean()
print(f"{'Avg $/min':<20} ${day_rate_min:<14.2f} ${night_rate_min:<14.2f} ${night_rate_min-day_rate_min:<+9.2f}")
