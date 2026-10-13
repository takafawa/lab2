import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('brainsize.txt', sep='\t', na_values='NA')
plt.figure(figsize=(8, 6))
corr_matrix = df[['FSIQ', 'VIQ', 'PIQ', 'Weight', 'Height', 'MRI_Count']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Тепловая карта корреляций (Brain Size)")
plt.show()
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PIQ', y='MRI_Count', hue='Gender', style='Gender', s=100)
plt.title("Зависимость MRI_Count от PIQ (с разделением по полу)")
plt.xlabel("PIQ")
plt.ylabel("MRI_Count")
plt.legend(title="Пол")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
