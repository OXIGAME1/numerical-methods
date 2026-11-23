import time
import numpy as np

# ── Данные: 200 000 строк по 5 чисел ───────────────────────
rng = np.random.default_rng(42)
m = rng.normal(size=(200_000, 5))


# ── Способ 1: циклы ────────────────────────────────────────
def loop_version(mat):
    n, k = mat.shape

    # сумма каждой строки
    row_sums = [0.0] * n
    for i in range(n):                 # идём по каждой строке
        s = 0.0
        for j in range(k):             # складываем 5 чисел строки
            s += mat[i, j]
        row_sums[i] = s

    # максимум и его индекс по каждому столбцу
    col_max = [0.0] * k
    col_arg = [0] * k
    for j in range(k):                 # идём по каждому столбцу
        best = mat[0, j]               # считаем, что максимум — первый элемент
        best_i = 0
        for i in range(1, n):          # проверяем остальные строки
            if mat[i, j] > best:       # строго ">" — как argmax, берём первое вхождение
                best = mat[i, j]
                best_i = i
        col_max[j] = best
        col_arg[j] = best_i

    return row_sums, col_max, col_arg


# ── Способ 2: NumPy ────────────────────────────────────────
def numpy_version(mat):
    row_sums = mat.sum(axis=1)         # axis=1: схлопываем столбцы -> сумма каждой строки
    col_max = mat.max(axis=0)          # axis=0: схлопываем строки -> максимум каждого столбца
    col_arg = mat.argmax(axis=0)       # номер строки, где этот максимум
    return row_sums, col_max, col_arg


# ── Замер: 3 раза, берём лучший ────────────────────────────
def measure(fn, x, repeat=3):
    best = float("inf")
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn(x)
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return result, best


(r_loop,  t_loop)  = measure(loop_version,  m)
(r_numpy, t_numpy) = measure(numpy_version, m)

print(f"цикл:   {t_loop*1000:10.2f} мс")
print(f"NumPy:  {t_numpy*1000:10.2f} мс")
print(f"ускорение: {t_loop / t_numpy:.1f}×")
print()
print("суммы строк совпадают:  ", np.allclose(r_loop[0], r_numpy[0]))
print("максимумы совпадают:    ", np.allclose(r_loop[1], r_numpy[1]))
print("индексы совпадают:      ", np.array_equal(r_loop[2], r_numpy[2]))
print()
print("NumPy, максимумы по столбцам:", np.round(r_numpy[1], 4))
print("NumPy, их индексы:           ", r_numpy[2])