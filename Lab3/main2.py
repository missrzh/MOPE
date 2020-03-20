import math

import numpy as m
import numpy.linalg as l
import random


def matrix_generator(max_y, min_y, len_matrix):  # Генератор випадкових матриць для у
    matrix = []
    for n in range(4):
        lis = [random.randrange(min_y, max_y) for k in range(len_matrix)]
        matrix.append(lis)
    return matrix


def find_first_a(x, y):
    lis = []
    for i in m.array(x).transpose():
        a = [i[n] * y[n] for n in range(len(i))]
        lis.append(sum(a) / len(i))
    return lis


def find_second_a(x):
    lis = []
    for i in m.array(x).transpose():
        a = [k * k for k in i]
        lis.append(sum(a) / len(a))
    return lis


def find_third_a(lis1, lis2):
    lis = [a * b for a, b in zip(lis1, lis2)]
    return sum(lis) / len(lis)


matrix_of_x = [[-20, -15, -15],  # Матриця планування, частина з х
               [-20, 35, -10],
               [15, -15, -10],
               [15, 35, -15]]

matrix_of_y = matrix_generator(213, 183, 4)  # Заповнюю матрицю планування у
print(" Matrix of Y:")
print(" ---------------")
print('\n'.join([''.join(['{:4}'.format(item) for item in row])
      for row in matrix_of_y]))
print(" ---------------")
middles_y = [sum(i) / len(i) for i in matrix_of_y]  # Список середніх значень функції відгуку
print('| '.join('Ymid{} = {} '.format(n+1, k) for n, k in enumerate(middles_y)))
list_of_mx = [sum(i) / len(i) for i in m.array(matrix_of_x).transpose()]  # Список чередніх х
print('| '.join('Mx{} = {} '.format(n+1, k) for n, k in enumerate(list_of_mx)))
my = sum(middles_y) / len(middles_y)  # Середнє значення середніх у
print("My = " + str(my))
list_of_a = find_first_a(matrix_of_x, middles_y)  # Список з а1, а2, а3
print('| '.join('a{} = {} '.format(n+1, k) for n, k in enumerate(list_of_a)))
list_of_a1 = find_second_a(matrix_of_x)  # Список з a11, a22, a33
print('| '.join('a{}{} = {} '.format(n+1, n+1, k) for n, k in enumerate(list_of_mx)))
a12 = a21 = find_third_a(m.array(matrix_of_x).transpose()[0], m.array(matrix_of_x).transpose()[1])
print("a12 = a21 = " + str(a12))
a13 = a31 = find_third_a(m.array(matrix_of_x).transpose()[0], m.array(matrix_of_x).transpose()[2])
print("a13 = a31 = " + str(a13))
a23 = a32 = find_third_a(m.array(matrix_of_x).transpose()[1], m.array(matrix_of_x).transpose()[2])
print("a23 = a32 = " + str(a23))
print("------------------")
divider_matrix = m.array([[1, list_of_mx[0], list_of_mx[1], list_of_mx[2]],
                          [list_of_mx[0], list_of_a1[0], a12, a13],
                          [list_of_mx[1], a12, list_of_a1[1], a32],
                          [list_of_mx[2], a13, a23, list_of_a1[2]]])

first_matrix = m.array([[my, list_of_mx[0], list_of_mx[1], list_of_mx[2]],
                        [list_of_a[0], list_of_a1[0], a12, a13],
                        [list_of_a[1], a12, list_of_a1[1], a32],
                        [list_of_a[2], a13, a23, list_of_a1[2]]])

second_matrix = m.array([[1, my, list_of_mx[1], list_of_mx[2]],
                         [list_of_mx[0], list_of_a[0], a12, a13],
                         [list_of_mx[1], list_of_a[1], list_of_a1[1], a32],
                         [list_of_mx[2], list_of_a[2], a23, list_of_a1[2]]])

third_matrix = m.array([[1, list_of_mx[0], my, list_of_mx[2]],
                        [list_of_mx[0], list_of_a1[0], list_of_a[0], a13],
                        [list_of_mx[1], a12, list_of_a[1], a32],
                        [list_of_mx[2], a13, list_of_a[2], list_of_a1[2]]])

fourth_matrix = m.array([[1, list_of_mx[0], list_of_mx[1], my],
                         [list_of_mx[0], list_of_a1[0], a12, list_of_a[0]],
                         [list_of_mx[1], a12, list_of_a1[1], list_of_a[1]],
                         [list_of_mx[2], a13, a23, list_of_a[2]]])

print("Finding koef")
print("------------------")

b0 = l.det(first_matrix) / l.det(divider_matrix)  # Знаходжу коефіціенти
b1 = l.det(second_matrix) / l.det(divider_matrix)
b2 = l.det(third_matrix) / l.det(divider_matrix)
b3 = l.det(fourth_matrix) / l.det(divider_matrix)
koef = [b0, b1, b2, b3]
print('| '.join('b{} = {:.2f} '.format(n, k) for n, k in enumerate(koef)))
print("Check: ")

y1 = b0 + b1 * (-20) + b2 * (-15) + b3 * (-15)  # Перевірка коефіціентів
y2 = b0 + b1 * (-20) + b2 * 35 + b3 * (-10)
y3 = b0 + (b1 * 15) + (b2 * -15) + (b3 * -10)
y4 = b0 + b1 * 15 + b2 * 35 + b3 * (-15)
y = [y1, y2, y3, y4]
print('| '.join('Y{} = {:.2f} '.format(n+1, k) for n, k in enumerate(y)))
print('| '.join('Ymid{} = {} '.format(n+1, k) for n, k in enumerate(middles_y)))
# Перевірка за критерієм Кохрена
print("------------------")
print("Kohren Checking:")
print("------------------")
dis1 = ((matrix_of_y[0][0] - middles_y[0]) ** 2 + (matrix_of_y[0][1] - middles_y[0]) ** 2 +  # Знайду дисперсії
        (matrix_of_y[0][2] - middles_y[0]) ** 2) / 3
dis2 = ((matrix_of_y[1][0] - middles_y[1]) ** 2 + (matrix_of_y[1][1] - middles_y[1]) ** 2 +
        (matrix_of_y[1][2] - middles_y[1]) ** 2) / 3
dis3 = ((matrix_of_y[2][0] - middles_y[2]) ** 2 + (matrix_of_y[2][1] - middles_y[2]) ** 2 +
        (matrix_of_y[2][2] - middles_y[2]) ** 2) / 3
dis4 = ((matrix_of_y[3][0] - middles_y[3]) ** 2 + (matrix_of_y[3][1] - middles_y[3]) ** 2 +
        (matrix_of_y[3][2] - middles_y[3]) ** 2) / 3
dises = [dis1, dis2, dis3, dis4]
print('| '.join('Dispersion{} = {:.2f} '.format(n+1, k) for n, k in enumerate(dises)))
gp = max(dises) / sum(dises)
print("Gp = " + str(gp))
if gp <= 0.7679:
    print("Dispersion is homogeneous")
else:
    print("Dispersion is patchy")

# Оціню значимість коефіцієнтів регресії згідно критерію Стьюдента
print("------------------")
print("Student koef significance Checking:")
print("------------------")
disB = sum(dises) / 4
disb = disB / 12
disb2 = math.sqrt(disb)
beta0 = (middles_y[0] * 1 + middles_y[1] * 1 + middles_y[2] * 1 + middles_y[3] * 1) / 4
beta1 = (middles_y[0] * (-1) + middles_y[1] * (-1) + middles_y[2] * 1 + middles_y[3] * 1) / 4
beta2 = (middles_y[0] * (-1) + middles_y[1] * 1 + middles_y[2] * (-1) + middles_y[3] * 1) / 4
beta3 = (middles_y[0] * (-1) + middles_y[1] * 1 + middles_y[2] * 1 + middles_y[3] * (-1)) / 4
betas = [beta0, beta1, beta2, beta3]
print('| '.join('Beta{} = {:.2f} '.format(n+1, k) for n, k in enumerate(betas)))
t0 = beta0 / disb2
t1 = beta1 / disb2
t2 = beta2 / disb2
t3 = beta3 / disb2
count = len(betas)
for k, i in enumerate(betas):
    if i < 2.306:
        koef[k] = 0
        count -= 1
print("Koefs:")
print('| '.join('b{} = {:.2f} '.format(n+1, k) for n, k in enumerate(koef)))
y01 = koef[0] + koef[1]*matrix_of_x[0][0]+koef[2]*matrix_of_x[0][1]+koef[3]*matrix_of_x[0][2]
y02 = koef[0] + koef[1]*matrix_of_x[1][0]+koef[2]*matrix_of_x[1][1]+koef[3]*matrix_of_x[1][2]
y03 = koef[0] + koef[1]*matrix_of_x[2][0]+koef[2]*matrix_of_x[2][1]+koef[3]*matrix_of_x[2][2]
y04 = koef[0] + koef[1]*matrix_of_x[3][0]+koef[2]*matrix_of_x[3][1]+koef[3]*matrix_of_x[3][2]
y0 = [y01, y02, y03, y04]
print("Y only with significant koefs:")
print('| '.join('Y{} = {:.2f} '.format(n+1, k) for n, k in enumerate(y0)))
# Критерій Фішера
print("------------------")
print("Fisher Сriterion:")
print("------------------")
sad = (math.pow(y01 - middles_y[0], 2) + math.pow(y02 - middles_y[1], 2) +
       math.pow(y03 - middles_y[2], 2) + math.pow(y04 - middles_y[3], 2))
kof = sad/disB
print("Fp = " + str(kof))
if count == 2:
    if kof <= 4.5:
        print("The regression equation is adequate to the original at a significance level of 0.05")
    else:
        print("The regression equation is inadequate to the original at a significance level of 0.05")
elif count == 1:
    if kof <= 4.1:
        print("The regression equation is adequate to the original at a significance level of 0.05")
    else:
        print("The regression equation is inadequate to the original at a significance level of 0.05")
elif count == 3:
    if kof <= 5.3:
        print("The regression equation is adequate to the original at a significance level of 0.05")
    else:
        print("The regression equation is inadequate to the original at a significance level of 0.05")