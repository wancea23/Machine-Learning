import matplotlib.pyplot as plt
import numpy as np

#1 
# x = [1, 2, 3, 4, 5]
# y = [2, 3, 5, 7, 11]
# plt.plot(x, y)
# plt.title("Prime Numbers")
# plt.xlabel("Index")
# plt.ylabel("Value")
# plt.show()

# x = np.array([1, 2, 3, 4, 5])
# y = np.array([2, 3, 5, 7, 11])
# plt.plot(x, y)
# plt.title("Prime Numbers")
# plt.xlabel("Index")
# plt.ylabel("Value")
# plt.show()

#2 Customizing the plot
# line_style1 = dict(marker='H', markersize=20, markerfacecolor="r", markeredgecolor="purple", linestyle='--', linewidth=3, color='g')

# x = np.array([1, 2, 3, 4, 5])
# y1 = np.array([2, 3, 5, 7, 11])
# y2 = np.array([1, 4, 9, 16, 25])
# plt.plot(x, y1, **line_style1)
# plt.plot(x, y2, marker='o', markersize=10, markerfacecolor="b", markeredgecolor="orange", linestyle='-', linewidth=2, color='m')
# plt.title("Prime Numbers", fontsize=20,  family='serif', fontweight='bold', color='darkblue')
# plt.xlabel("Index")
# plt.ylabel("Value")
# plt.grid(axis='both', color='gray', linestyle='--', linewidth=4) # grid
# plt.tick_params(axis='both', direction='inout', length=10, width=2, colors='red', grid_color='blue', grid_alpha=1) # ticks and grid
# plt.xticks(x)
# plt.show()

#3 bar chart
# categories = ['A', 'B', 'C', 'D', 'E']
# values = [10, 15, 7, 12, 20]
# plt.bar(categories, values, color=['red', 'blue', 'green', 'orange', 'purple'])
# plt.title("Category Values")
# plt.xlabel("Categories")
# plt.ylabel("Values")
# plt.yticks(values)
# plt.show()

#4 pie chart
# categories = ['A', 'B', 'C', 'D', 'E']
# values = [10, 15, 7, 12, 20]
# plt.pie(values, labels=categories, autopct='%1.1f%%', startangle=140, colors=['red', 'blue', 'green', 'orange', 'purple'], explode=[0.1, 0, 0, 0, 0]) # explode to highlight the first slice
# plt.title("Category Distribution")
# plt.axis('equal')  # Equal aspect ratio ensures that pie chart is circular.
# plt.show()

#5 scatter plot
# x = np.array([0,1,2,3,4,5,6,7,7,7,8])
# y = np.array([55,60,65,62,70,68,75,80,85,90,95])
# plt.scatter(x, y, color='red', marker='o', s=100) # s is the size of the markers
# plt.title("Scatter Plot of Prime Numbers")
# plt.xlabel("Index")
# plt.ylabel("Value")
# plt.grid(True)
# plt.show()

#6 histogram
# data = np.random.normal(loc=0, scale=10, size=1000)
# scores = np.clip(data, -1, 1) # clip the data to be between -3 and 3
# plt.hist(data, bins=30, color='blue', edgecolor='black') # bins is the number of bars in the histogram
# plt.title("Histogram of Normally Distributed Data")
# plt.xlabel("Value")
# plt.ylabel("Frequency")
# plt.grid(axis='y', alpha=0.75) # grid only on y-axis
# plt.show()

