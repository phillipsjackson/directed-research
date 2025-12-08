import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('output9.csv')

# Filter data for carId 0 and 1
data_car1 = df[df['carId'] == 0]
data_car2 = df[df['carId'] == 1]

# List of columns to compare
columns_to_compare = [' speed', ' brake', ' track position']
car_labels = ['Car 0', 'Car 1']

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

for i, col in enumerate(columns_to_compare):
    axs[i].boxplot(
        [data_car1[col].dropna(), data_car2[col].dropna()],
        labels=car_labels
    )
    axs[i].set_xlabel('Car')
    axs[i].set_ylabel(col.strip().capitalize())
    axs[i].set_title(f'Boxplot of {col.strip().capitalize()}')
    axs[i].grid(True)
    # Set more specific y-axis ticks for track position
    if col == ' track position':
        min_pos = min(data_car1[col].min(), data_car2[col].min())
        max_pos = max(data_car1[col].max(), data_car2[col].max())
        axs[i].set_yticks(np.arange(
            np.floor(min_pos * 20) / 20,  # round down to nearest 0.05
            np.ceil(max_pos * 20) / 20 + 0.05,  # round up to nearest 0.05
            0.05
        ))

plt.tight_layout()
plt.show()