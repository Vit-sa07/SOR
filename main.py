import numpy as np
import matplotlib.pyplot as plt


# Функции для правой части и точного решения уравнения Пуассона
def source_function(x, y):
    return -2 * np.cos(x) * np.sin(y)


def true_solution(x, y):
    return np.cos(x) * np.sin(y)


def calculate_error(u, u_true):
    return np.linalg.norm(u - u_true, ord=np.inf)


# Функция инициализации задает начальные условия
def initialize_grid(N, h):
    # Генерация равномерной сетки
    x = np.linspace(0, 1, N + 2)
    y = np.linspace(0, 1, N + 2)
    u_initial = np.zeros((N + 2, N + 2))  # Начальное приближение
    source_term = np.zeros((N + 2, N + 2))  # Правая часть уравнения
    # Вычисление правой части для каждой внутренней точки сетки
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            source_term[i, j] = h ** 2 * source_function(x[j], y[i])
    return x, y, u_initial, source_term


# Функция реализует метод SOR с регистрацией ошибок на каждой итерации
def solve_SOR(u, source_term, omega, N, h, tolerance, max_iterations, u_true):
    errors = []  # Список для хранения ошибок на каждой итерации
    for iteration in range(max_iterations):
        previous_u = u.copy()
        # Итерационный процесс метода SOR
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                u[i, j] = ((1 - omega) * u[i, j] +
                           omega * 0.25 * (u[i + 1, j] + u[i - 1, j] +
                                           u[i, j + 1] + u[i, j - 1] -
                                           source_term[i, j]))
        # Вычисление ошибки на текущей итерации
        error = np.linalg.norm(u - u_true, ord=np.inf)
        errors.append(error)
        # Проверка условия остановки
        if error < tolerance:
            break
    return u, iteration + 1, errors


# Функция для решения методом конъюгированных градиентов с регистрацией ошибок на каждой итерации
def solve_conjugate_gradient(u, source_term, N, h, tolerance, max_iterations, u_true):
    errors = []  # Список для хранения ошибок на каждой итерации
    residual = source_term - laplace_operator(u, h)  # Вычисление остатка
    direction = residual.copy()
    for iteration in range(max_iterations):
        Adirection = laplace_operator(direction, h)
        alpha = np.sum(residual ** 2) / np.sum(direction * Adirection)
        u += alpha * direction  # Обновление приближенного решения
        new_residual = residual - alpha * Adirection
        # Вычисление ошибки на текущей итерации
        error = np.linalg.norm(u - u_true, ord=np.inf)
        errors.append(error)
        # Проверка условия остановки
        if error < tolerance:
            break
        denom = np.sum(residual ** 2) + 1e-10  # Избежание деления на ноль
        beta = np.sum(new_residual ** 2) / denom
        direction = new_residual + beta * direction
        residual = new_residual
    return u, iteration + 1, errors


# Функция для вычисления оператора Лапласа
def laplace_operator(u, h):
    laplacian = np.zeros_like(u)
    for i in range(1, len(u) - 1):
        for j in range(1, len(u[i]) - 1):
            laplacian[i, j] = ((u[i + 1, j] + u[i - 1, j] + u[i, j + 1] + u[i, j - 1] -
                                4 * u[i, j]) / h ** 2)
    return laplacian


# Основной блок кода
h_values = [0.1, 0.05, 0.025, 0.01]  # Значения шага сетки
tolerance = 1e-6  # Порог для остановки итерационного процесса
max_iterations = 10000  # Максимальное количество итераций

sor_final_errors = []  # Финальные ошибки метода SOR
cg_final_errors = []  # Финальные ошибки метода конъюгированных градиентов

# Цикл по различным значениям шага сетки
for h in h_values:
    N = int(1 / h) - 1  # Размер сетки
    omega = 2 / (1 + np.pi * h)  # Параметр релаксации для метода SOR

    x, y, u, source_term = initialize_grid(N, h)  # Инициализация сетки и правой части
    u_true = true_solution(x[:, None], y[None, :])  # Точное решение

    # Решение уравнения методом SOR
    u_sor, sor_iterations, sor_errors = solve_SOR(u.copy(), source_term, omega, N, h, tolerance, max_iterations,
                                                  u_true)
    sor_final_error = calculate_error(u_sor, u_true)
    sor_final_errors.append(sor_final_error)

    # Решение уравнения методом конъюгированных градиентов
    u_cg, cg_iterations, cg_errors = solve_conjugate_gradient(u.copy(), source_term, N, h, tolerance, max_iterations,
                                                              u_true)
    cg_final_error = calculate_error(u_cg, u_true)
    cg_final_errors.append(cg_final_error)

    # Вывод результатов
    print(f'h={h}: SOR iterations={sor_iterations}, Final Error={sor_final_error}')
    print(f'h={h}: CG iterations={cg_iterations}, Final Error={cg_final_error}')

    # Построение графиков ошибок от количества итераций для текущего значения h
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, sor_iterations + 1), sor_errors, label=f'SOR (h={h})')
    plt.plot(range(1, cg_iterations + 1), cg_errors, label=f'CG (h={h})')
    plt.xlabel('Iterations')
    plt.ylabel('Error (L∞ norm)')
    plt.yscale('log')
    plt.title(f'Error vs Iterations for h={h}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Построение графика зависимости финальной ошибки от шага сетки h
plt.figure(figsize=(10, 5))
plt.plot(h_values, sor_final_errors, 'o-', label='SOR Final Error')
plt.plot(h_values, cg_final_errors, 's-', label='CG Final Error')
plt.xlabel('Grid spacing h')
plt.ylabel('Final Error (L∞ norm)')
plt.xscale('log')
plt.yscale('log')
plt.gca().invert_xaxis()  # Для того чтобы меньшие значения h были справа
plt.legend()
plt.title('Final Error vs Grid spacing h')
plt.grid(True)
plt.show()
