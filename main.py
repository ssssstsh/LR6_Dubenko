# # 1. Переключаем бэкенд на 'Agg' — неинтерактивный, умеет только сохранять в файлы
# import matplotlib
# matplotlib.use('Agg')   # ЭТА СТРОКА ДОЛЖНА БЫТЬ ДО ИМПОРТА pyplot

# # 2. Теперь можно импортировать pyplot
# import matplotlib.pyplot as plt

# # 3. Создаём фигуру и оси
# fig, ax = plt.subplots()

# # 4. Строим график
# ax.plot([1, 2, 3, 4], [1, 4, 2, 5])

# # 5. Подписываем ось Y
# plt.ylabel('some numbers')

# # 6. Вместо plt.show() сохраняем график в файл
# plt.savefig('my_plot.png')

# 7. (Опционально) Выведем сообщение пользователю
# print("График сохранён как my_plot.png")
# 2 задание


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import pandas as pd

dataset, classes = make_blobs(n_samples=200, n_features=2,
                              centers=4, cluster_std=0.5, random_state=0)
df = pd.DataFrame(dataset, columns=['var1', 'var2'])
print(df.head(2))

inertias = []
k_range = range(1, 12)
for k in k_range:
    model = KMeans(n_clusters=k, random_state=0)
    model.fit(df)
    inertias.append(model.inertia_)

plt.figure()
plt.plot(list(k_range), inertias, 'bo-')
plt.xlabel('k')
plt.ylabel('Distortion score')
plt.title('Elbow Method')
plt.savefig('elbow_plot.png')
plt.close() 
print("Elbow plot saved as elbow_plot.png")

kmeans = KMeans(n_clusters=4, init='k-means++',
random_state=0).fit(df)
# print (kmeans.labels_)
# print (kmeans.cluster_centers_)
# print(kmeans.inertia_)
# print(kmeans.n_iter_)

from collections import Counter
Counter(kmeans.labels_)
print(Counter(kmeans.labels_))

import seaborn as sns

sns.scatterplot(data=df, x='var1', y='var2', hue=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker="X", c="r", s=80, label="centroids")
plt.grid(True)
plt.savefig('scatter_plot.png')
plt.close() 
print("Scatter plot saved as scatter_plot.png")