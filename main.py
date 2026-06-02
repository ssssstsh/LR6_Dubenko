
# Лабораторная работа №6
# Общая часть
# Микросервис кластеризации данных методом KMeans
# на базе паттерна Python Data Science в Replit



# Блок 1. Подключение библиотек


# Подключаем matplotlib и переводим его в неинтерактивный режим Agg.
# Это нужно для Replit, чтобы графики не открывались в отдельном окне,
# а сохранялись в виде PNG-файлов.
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# make_blobs используется для генерации синтетического набора данных.
from sklearn.datasets import make_blobs

# KMeans используется для кластеризации методом K-средних.
from sklearn.cluster import KMeans

# pandas используется для представления данных в табличном виде.
import pandas as pd

# Counter используется для подсчета количества объектов в каждом кластере.
from collections import Counter

# json используется для сохранения результатов работы микросервиса в файл.
import json



# Блок 2. Генерация исходного набора данных


# Генерируем двумерный набор данных из 200 объектов.
# Параметр centers=4 задает четыре исходных центра распределения данных.
# Параметр cluster_std=0.5 задает разброс точек вокруг центров.
dataset, classes = make_blobs(
    n_samples=200,
    n_features=2,
    centers=4,
    cluster_std=0.5,
    random_state=0
)

# Преобразуем массив данных в DataFrame для удобной обработки и вывода.
df = pd.DataFrame(dataset, columns=["var1", "var2"])

print("Исходный набор данных, первые 5 строк:")
print(df.head(5))
print()



# Блок 3. Поиск оптимального числа кластеров методом локтя


# Список для хранения значений внутрикластерной суммы квадратов.
inertias = []

# Проверяем разные значения количества кластеров от 1 до 11.
k_range = range(1, 12)

# Для каждого значения k обучаем модель KMeans и сохраняем inertia.
# inertia показывает сумму квадратов расстояний объектов до центров кластеров.
for k in k_range:
    model = KMeans(
        n_clusters=k,
        random_state=0,
        n_init=10
    )
    model.fit(df)
    inertias.append(model.inertia_)

print("Значения внутрикластерной суммы квадратов для метода локтя:")
for k, inertia in zip(k_range, inertias):
    print(f"k = {k}: inertia = {inertia:.4f}")
print()



# Блок 4. Построение графика метода локтя


plt.figure()
plt.plot(list(k_range), inertias, "bo-")
plt.xlabel("Количество кластеров k")
plt.ylabel("Внутрикластерная сумма квадратов")
plt.title("Метод локтя для выбора числа кластеров")
plt.grid(True)
plt.tight_layout()
plt.savefig("elbow_plot.png")
plt.close()

print("График метода локтя сохранен в файл elbow_plot.png")
print()



# Блок 5. Обучение итоговой модели KMeans


# По графику метода локтя для сгенерированных данных выбираем 4 кластера,
# так как исходные данные были сгенерированы вокруг четырех центров.
kmeans = KMeans(
    n_clusters=4,
    init="k-means++",
    random_state=0,
    n_init=10
)

# Метод fit обучает модель на подготовленном наборе данных.
kmeans.fit(df)

# Получаем прогнозируемые метки кластеров для каждой точки.
labels = kmeans.labels_

# Получаем координаты центроидов кластеров.
centroids = kmeans.cluster_centers_

# Получаем значение внутрикластерной суммы квадратов итоговой модели.
final_inertia = kmeans.inertia_

# Получаем количество итераций, потребовавшихся алгоритму для сходимости.
iterations = kmeans.n_iter_

# Считаем размер каждого кластера.
cluster_sizes = Counter(labels)



# Блок 6. Вывод результатов кластеризации в консоль


print("Прогнозируемые кластеры для каждой точки данных:")
print(labels)
print()

print("Координаты центроидов кластеров:")
print(centroids)
print()

print(f"Внутрикластерная сумма квадратов итоговой модели: {final_inertia:.4f}")
print(f"Количество итераций KMeans: {iterations}")
print()

print("Размер каждого кластера:")
print(cluster_sizes)
print()



# Блок 7. Сохранение результатов работы микросервиса в JSON


results = {
    "dataset_head": df.head(5).to_dict(orient="records"),
    "selected_clusters": 4,
    "elbow_inertias": {
        str(k): inertia for k, inertia in zip(k_range, inertias)
    },
    "labels": labels.tolist(),
    "centroids": centroids.tolist(),
    "final_inertia": final_inertia,
    "iterations": iterations,
    "cluster_sizes": {
        str(cluster): int(size) for cluster, size in cluster_sizes.items()
    }
}

with open("results.json", "w", encoding="utf-8") as file:
    json.dump(results, file, ensure_ascii=False, indent=4)

print("Результаты работы микросервиса сохранены в файл results.json")
print()



# Блок 8. Построение диаграммы рассеяния кластеров


plt.figure()

# Строим точки данных. Цвет каждой точки соответствует предсказанному кластеру.
scatter = plt.scatter(
    df["var1"],
    df["var2"],
    c=labels,
    s=40
)

# Отдельно отмечаем центроиды кластеров маркером X.
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="X",
    s=120,
    label="Центроиды"
)

plt.xlabel("var1")
plt.ylabel("var2")
plt.title("Диаграмма рассеяния кластеров KMeans")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("scatter_plot.png")
plt.close()

print("Диаграмма рассеяния сохранена в файл scatter_plot.png")