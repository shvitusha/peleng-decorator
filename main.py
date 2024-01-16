import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

data = pd.read_excel("D:\\Peleng\\data_processing\\data_1.xlsx")
data = data.iloc[:, 1:]

#нормализация данных
scaler = MinMaxScaler(feature_range=(-1, 1))
normalized_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

#обработка пропусков и выбросов
for column in normalized_data.columns:
    middle_value = normalized_data[column].median()
    normalized_data[column].fillna(middle_value, inplace=True)

    Q1 = normalized_data[column].quantile(0.25)
    Q3 = normalized_data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    normalized_data[column] = np.where(
        (normalized_data[column] < lower) | (normalized_data[column] > upper),
        middle_value,
        normalized_data[column]
    )

#матрица корреляции
correlation_matrix = normalized_data.corr()

#гистограммы распределения
#plt.figure(1)
for column in normalized_data.columns:
    plt.figure()
    sns.histplot(normalized_data[column], kde=True, bins=20)
    plt.title(column)
    plt.xlabel(column)
    #sns.histplot(normalized_data[column], kde=True, bins=20, label=column)
#plt.legend()

#взаимозависимые данные
plt.figure(8)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")

#нормальность
for column in normalized_data.columns:
    statistic, p_value = stats.normaltest(normalized_data[column])
    if p_value < 0.05:
        print(f"{column} не является нормально распределенной. p-value: {p_value}")
        print(statistic)
    else:
        print(f"{column} является нормально распределенной. p-value: {p_value}")


plt.show()
