import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('../dataset/heart.csv')

print(data.info())
print(data.describe())
print(data.isnull().sum())

sns.countplot(x='target', data=data)
plt.title('Distribusi Target')
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap Korelasi')
plt.show()
