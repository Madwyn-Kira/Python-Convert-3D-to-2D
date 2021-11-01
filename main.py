from math import sqrt

from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler
from PIL import Image as img, ImageColor
import numpy as np
import matplotlib.pyplot as plt
import re


# Считаем кол-во вершин
def CountLinesOfVertices():
    ite = 0
    _word = 'f '
    with open("index.txt", "r", encoding="utf-8") as file:
        for line in file:
            if _word in line:
                continue
            else:
                ite += 1
    return ite


# Считаем кол-во ребер
def CountLinesOfRebra():
    ite = 0
    _word = 'v '
    with open("index.txt", "r", encoding="utf-8") as file:
        for line in file:
            if _word in line:
                continue
            else:
                ite += 1
    return ite


vertices_array = np.zeros((CountLinesOfVertices(), 3))
triagles_array = np.zeros((CountLinesOfRebra(), 3), dtype=int)
word = 'f '
it = 0
it2 = 0


# Записываем в массив
def writeInArray(_line, itr):
    p = _line.split()
    vertices_array[itr] = p


# Записываем в массив ребра
def writeInArrayRebra(_line, itr):
    p = _line.split()
    triagles_array[itr] = p


# Достаем всё из файла и убираем лишнее
with open("index.txt", "r", encoding="utf-8") as file:
    for line in file:
        if word in line:
            _lineReb = re.sub(r"f ", "", line)
            writeInArrayRebra(_lineReb, it2)
            it2 = it2 + 1
        else:
            _line = re.sub(r"v ", "", line)
            writeInArray(_line, it)
            it = it + 1

# Нормализация для изображения 512x512
scaler = MinMaxScaler(feature_range=(0, 512))
scaler.fit(vertices_array)
scaled_array = scaler.transform(vertices_array)
# print(scaled_array)


# Для отрисовки графика чайника вида сбоку.
'''
# Генерируем график
for i in range(CountLinesOfVertices()):
    # ax.scatter3D(scaled[i][0], scaled[i][1], scaled[i][2], c=scaled[0][2], cmap='hsv'); # Для 3Д (Пока не нужно)
    plt.plot(scaled[i][0], scaled[i][1], 'bx')


# Рисуем график
plt.show()
print(CountLinesOfVertices())
'''


# рисуем пиксель 1
def set_pixel_first(_img, x, y):
    total_row, total_col, linear = _img.shape
    cen_x, cen_y = total_row / 2, total_col / 2
    d1 = (x - cen_x) ** 2
    d2 = (y - cen_y) ** 2
    D = euclidean((x, y), (cen_x, cen_y))
    print(D)
    d = sqrt(d1 + d2) + 160
    base_color = [255, 0, 0]
    N = 512
    color = [0, 0, 0]
    color[0] = int(base_color[0] * (1 - d / N))
    color[1] = int(base_color[1] * (1 - d / N))
    color[2] = int(base_color[2] * (1 - d / N))
    _img[512 - x - 1][512 - y - 1] = color


# рисуем пиксель 2
def set_pixel_second(_img, x, y):
    total_row, total_col, linear = _img.shape
    cen_x, cen_y = total_row / 2, total_col / 2
    d1 = (x - cen_x) ** 2
    d2 = (y - cen_y) ** 2
    d = sqrt(d1 + d2) + 160
    base_color = [255, 0, 0]
    N = 512
    color = [0, 0, 0]
    color[0] = int(base_color[0] * (1 - d / N))
    color[1] = int(base_color[1] * (1 - d / N))
    color[2] = int(base_color[2] * (1 - d / N))
    _img[512 - y - 1][512 - x - 1] = color


# Алгоритм Брезенхема
def line(_img, x0, y0, x1, y1):
    steep = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        steep = True
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    dy = abs(y1 - y0)
    signy = -1 if y0 > y1 else 1
    error = 0
    y = y0
    for x in range(x0, x1):
        if steep:
            if x > 0 and y > 0:
                set_pixel_first(_img, x, y)
        else:
            if x > 0 and y > 0:
                set_pixel_second(_img, x, y)
        error += dy
        if abs(x1 - x0) <= 2 * error:
            y += signy
            error -= abs(x1 - x0)


# NumPy массив изображения
_img = np.zeros((512, 512, 3), dtype=np.uint8)
_img[:, :, :3] = 100  # Задали цвет фона

# Отрисовка ребер треугольников
for i in range(CountLinesOfRebra()):
    v1 = triagles_array[i][0] - 1
    v2 = triagles_array[i][1] - 1
    v3 = triagles_array[i][2] - 1

    line(_img, int(scaled_array[v1][0]), int(scaled_array[v1][1]), int(scaled_array[v2][0]), int(scaled_array[v2][1]))
    line(_img, int(scaled_array[v2][0]), int(scaled_array[v2][1]), int(scaled_array[v3][0]), int(scaled_array[v3][1]))
    line(_img, int(scaled_array[v3][0]), int(scaled_array[v3][1]), int(scaled_array[v1][0]), int(scaled_array[v1][1]))


# Конвертирование и вывод изображения
_img = img.fromarray(_img)
_img.save("result.png")