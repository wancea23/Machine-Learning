import pandas as pd

# print(pd.__version__)

#1 Creating a Series from a list
# data = [12, 233, 34, 411, 523]
# index = [i for i in range(len(data))]
# series = pd.Series(data, index=index)
# series.loc[3] = 10000
# print(series)
# print("\n")
# print(series.iloc[0])
# print("\n")
# print(series[series > 1000])

#2 Creating a DataFrame from a dictionary (2d list)
# calories = {
#     'Day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'],
#     'Calories': [2200, 2100, 2500, 2300, 2400]
# }
# series = pd.Series(calories['Calories'], index=calories['Day'])
# series.loc['Thu'] += 2600000
# print(series)
# print(series[series > 2500])
# print("\n")
# df = pd.DataFrame(calories)
# df.loc[3, 'Calories'] += 2600000
# print(df)
# print(df[df['Calories'] > 2500])
# print("\n")
# print(df.loc[3])
# print("\n")
# #Adding a new column to the DataFrame
# df['Proteins'] = [50, 60, 55, 65, 70]
# print(df)
# print("\n")
# #adding a new row to the DataFrame
# new_row = pd.DataFrame([{'Day': 'Sat', 'Calories': 2600, 'Proteins': 75}])
# df = pd.concat([df, new_row], ignore_index=True)
# print(df)

#3 Reading a CSV file into a DataFrame
# df = pd.read_csv('chicago_taxi_train.csv')

# #everything
# # print(df.to_string())

# print(df)

#4 Reading a CSV file with a specific separator
# df=pd.read_csv('chicago_taxi_train.csv')

# #selecting specific columns
# print(df[['TRIP_END_TIMESTAMP', 'TRIP_START_HOUR']])

#5 Filtering rows based on a condition
# df = pd.read_csv('chicago_taxi_train.csv')

# fare = df[(df['FARE'] > 150) | (df['FARE'] % 5 == 0)]
# print(fare[['TRIP_START_TIMESTAMP', 'FARE']])

#6 Grouping data and calculating aggregate statistics
# df = pd.read_csv('chicago_taxi_train.csv')
# grouped = df.groupby('TRIP_START_HOUR')['FARE'].mean() #grouping the DataFrame by 'TRIP_START_HOUR' and calculating the mean of the 'FARE' column for each group
# print(grouped)
# sum_fare = df.groupby('TRIP_START_HOUR')['FARE'].sum()
# print(sum_fare)
# print(df.mean(numeric_only=True)) #calculating the mean of all numeric columns in the DataFrame
# print(df.count()) #counting the number of non-null values in each column of the DataFrame
# print(df.max())
# print("\n")
# print(df["FARE"].count()) #counting the number of non-null values in the 'FARE' column of the DataFrame
# print(df["FARE"].sum())
# print("\n")
# print(df.max())

#7 Cleaning data by handling missing values
# df = pd.read_csv('chicago_taxi_train.csv')
# df = df.dropna() #removing rows with missing values
# print(df)