import math
import scipy.stats as stats
import numpy
import numpy.linalg as l
import random


# Функції---------------------------------------------------------------------------------------------------------------


def Student_check(dises1):  # Функція оцінювання значимості коефіцієнтів регресії згідно критерію Стьюдента
    disBi = sum(dises1) / N
    disb = disBi / N * m
    disb2 = math.sqrt(disb)
    b = numpy.insert(norm_matrix_of_x, 0, 1, axis=1).transpose()
    betas = [sum([x * y for x, y in zip(middles_y, b[i])]) / N
             for i in range(len(b))]
    ts = [i / disb2 for i in betas]
    counti = len(ts)
    for k, i in enumerate(ts):
        if i < stats.t.ppf(1.95 / 2, f3):
            koef[k] = 0
            counti -= 1
    y0i = [y_function(koef, el) for el in matrix_of_x]
    print("----------------------------------------------------------------------")
    print("Student koef significance Checking:")
    print("Koefs:")
    print('| '.join('b{} = {:.2f} '.format(n + 1, k) for n, k in enumerate(koef)))
    print("Y only with significant koefs:")
    print('| '.join('Y{} = {:.2f} '.format(n + 1, k) for n, k in enumerate(y0i)))
    return disBi, counti, y0i


def Kokhren_check(matrix_y, m):  # Функія для перевірки методом Кохрена
    disesi = [find_dispersion(matrix_y[i]) for i in range(len(matrix_y))]
    gp = max(disesi) / sum(disesi)
    fisher = table_fisher(0.95, 1, f1 * m)
    Gt = fisher / (fisher + f1)

    print("----------------------------------------------------------------------")
    print("Kochren Checking:")

    if gp <= Gt:
        print("Dispersion is homogeneous")
        return 0
    else:
        print("Dispersion is not homogeneous")
        return 1


def fill_second_norm_matrix(matrix, lis1, lis2):  # Функція для знаходження нормовної матриці х з еф. взаємодії
    new = numpy.column_stack((numpy.array(matrix),
                              numpy.array([abs(x * y) if x * y == 0 else x * y for x, y in zip(lis1, lis2)])))
    return new


def fill_norm_matrix_square(matrix, lis1):  # Функція для знаходження нормовної матриці з квадратами
    new = numpy.column_stack((numpy.array(matrix),
                              numpy.array([x * x for x in lis1])))
    return new


def find_dispersion(yg):  # Функція для знаходження дисперсій
    yj = sum(yg) / m
    n = [(yj - yg[i]) ** 2 for i in range(m)]
    return sum(n) / m


def table_fisher(prob, d, f3k):  # Функція для перевірки методом Фішера
    x_vec = [i * 0.001 for i in range(int(10 / 0.001))]
    for i in x_vec:
        if abs(stats.f.cdf(i, 4 - d, f3k) - prob) < 0.0001:
            return i


def find_first_a(x, y):
    lis = []
    for i in numpy.array(x).transpose():
        a = [i[n] * y[n] for n in range(len(i))]
        lis.append(sum(a) / len(i))
    return lis


def matrix_generator(max_y, min_y, len_matrix, mg):  # Генератор випадкових матриць для у
    matrix = []
    for n in range(len_matrix):
        lis = [random.randrange(min_y, max_y) for k in range(mg)]
        matrix.append(lis)
    return matrix


def fill_matrix_of_x(matrix, lis_max, lis_min):  # Заміщення нормованих коефіцієнтів
    x = numpy.array(matrix).transpose()
    new_matrix = []
    for k, i in enumerate(x):
        n = [lis_max[k] * j if j >= 0 else lis_min[k] * abs(j) for j in i]
        new_matrix.append(n)
    n = numpy.array(new_matrix).transpose()
    return n


def find_koefs(number, matrix, lis, div):
    mx = matrix.copy()
    for i, el in enumerate(mx):
        el[number] = lis[i]
    return l.det(mx) / div


def find_second_a(x):
    lis = []
    for i in numpy.array(x).transpose():
        a = [k * k for k in i]
        lis.append(sum(a) / len(a))
    return lis


def find_third_a(lis1, lis2):
    lis = [a * b for a, b in zip(lis1, lis2)]
    return sum(lis) / len(lis)


def y_function(kof, x):  # Лінійне рівняння регресії
    return kof[0] + kof[1] * x[0] + kof[2] * x[1] + kof[3] * x[2] + kof[4] * x[3] + kof[5] * x[4] + kof[6] * x[5] + \
           kof[7] * x[6] + kof[8] * x[7] + kof[9] * x[8]


N = 15
x1min, x1max = -4, 6  # Значення за варіантом
x2min, x2max = -1, 2
x3min, x3max = -4, 2
li_max = [x1max, x2max, x3max]
li_min = [x1min, x2min, x3min]
x_mid_max = (x1max + x2max + x3max) / 3
x_mid_min = (x1min + x2min + x3min) / 3
y_max = 200 + x_mid_max
y_min = 200 + x_mid_min
m = 3
while True:
    f1 = m - 1
    f2 = N
    f3 = f1 * f2
    norm_matrix_of_x = [[-1, -1, -1],  # Матриця планування, частина з х
                        [-1, -1, 1],
                        [-1, 1, -1],
                        [-1, 1, 1],
                        [1, -1, -1],
                        [1, -1, 1],
                        [1, 1, -1],
                        [1, 1, 1],
                        [-1.215, 0, 0],
                        [1.215, 0, 0],
                        [0, -1.215, 0],
                        [0, 1.215, 0],
                        [0, 0, -1.215],
                        [0, 0, 1.215],
                        [0, 0, 0]]
    matrix_of_x = fill_matrix_of_x(norm_matrix_of_x, li_max, li_min)  # Заміщення нормованих значень
    matrix_of_x_transpose = numpy.array(matrix_of_x).transpose()  # Транспоную матрицю для зручнішого використання
    matrix_of_x_2 = fill_second_norm_matrix(matrix_of_x, matrix_of_x_transpose[0],
                                            matrix_of_x_transpose[1])
    matrix_of_x_2 = fill_second_norm_matrix(matrix_of_x_2, matrix_of_x_transpose[0],
                                            matrix_of_x_transpose[2])
    matrix_of_x_2 = fill_second_norm_matrix(matrix_of_x_2, matrix_of_x_transpose[1],
                                            matrix_of_x_transpose[2])
    matrix_of_x_2 = fill_second_norm_matrix(matrix_of_x_2, matrix_of_x_transpose[0],
                                            matrix_of_x_2.transpose()[5])
    final_matrix = fill_norm_matrix_square(matrix_of_x_2, matrix_of_x_transpose[0])
    final_matrix = fill_norm_matrix_square(final_matrix, matrix_of_x_transpose[1])
    matrix_of_x = fill_norm_matrix_square(final_matrix, matrix_of_x_transpose[2])
    matrix_of_y = matrix_generator(int(y_max), int(y_min), f2, m)  # Створення матриці планування у
    middles_y = [sum(i) / len(i) for i in matrix_of_y]  # Список середніх значень функції відгуку
    list_of_mx = [sum(i) / len(i) for i in numpy.array(matrix_of_x).transpose()]  # Список чередніх х
    my = sum(middles_y) / len(middles_y)  # Середнє значення середніх у
    list_of_a = find_first_a(matrix_of_x, middles_y)  # Список з а1, а2, а3 (вільні члени)
    list_of_a1 = find_second_a(matrix_of_x)  # Список з a11, a22, a33 i.т.д
    matrix_of_a = [[find_third_a(i, column) for column in matrix_of_x.transpose()] for i in
                   matrix_of_x.transpose()]  # список з а
    list_of_a.insert(0, my)  # Роблю матрицю з коефіціентів а та мх
    matrix_of_a.insert(0, list_of_mx)  # Роблю матрицю з коефіціентів а та мх
    list_of_mx_copy = list_of_mx.copy()  # Роблю матрицю з коефіціентів а та мх
    list_of_mx_copy.insert(0, 1)  # Роблю матрицю з коефіціентів а та мх
    for i, line in enumerate(matrix_of_a):  # Роблю матрицю з коефіціентів а та мх
        line.insert(0, list_of_mx_copy[i])  # Роблю матрицю з коефіціентів а та мх
    divider = l.det(numpy.array(matrix_of_a))  # Знайшов детермінант головної матриці
    koef = [find_koefs(i, matrix_of_a, list_of_a, divider) for i in
            range(len(list_of_a))]  # Знаходжу коефіціенти (у функції)
    print(koef)
    y = []
    dises = [find_dispersion(matrix_of_y[i]) for i in range(len(matrix_of_y))]  # Знаходжу дисперсії
    for i in range(len(matrix_of_x)):
        y.append(y_function(koef, matrix_of_x[i]))  # Знаходжу значення функії
    if Kokhren_check(matrix_of_y, m) == 0:  # Перевірка методом Кохрена (відбувається у функції)
        break
    else:
        m += 1
disB, count, y0 = Student_check(
    dises)  # Оцінка значимості коефіцієнтів регресії згідно критерієм Стьюдента (відбувається у функціїї)
print("----------------------------------------------------------------------")
print("Fisher Сriterion:")
sad = sum([(x - y) ** 2 for x, y in zip(y0, middles_y)])
kof = sad / disB
print("Fp = " + str(kof))
fp = stats.f.ppf(0.95, N - count, f3)
if kof <= fp:  # Перевірка адекватності за критерієм Фішера
    print("Уравнение регрессии адекватно оригиналу на уровне значимости 0.05")
    print("Ответ:")
    print("y = {:.2f} + {:.2f}*x1 + {:.2f}*x2 + {:.2f}*x3 + {:.2f}*x1x2 + {:.2f}*x1x3 + {:.2f}*x2x3 + "
          "{:.2f}*x1x2x3 + {:.2f}*x1^2 + {:.2f}*x2^2 + {:.2f}*x3^2".format(*koef))
    print("----------------------------------------------------------------------")

else:
    print("уравнение регрессии не соответствует оригиналу на уровне значимости 0.05")
