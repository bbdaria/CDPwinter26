import numpy as np
import timeit
from scipy.signal import convolve2d
from filters import correlation_numba

# Определяем ядра (как в твоем файле)
edge_kernel = np.array([[-3 / 18, -2 / 13, -1 / 10, 0, 1 / 10, 2 / 13, 3 / 18],
                        [-3 / 13, -2 / 8, -1 / 5, 0, 1 / 5, 2 / 8, 3 / 13],
                        [-3 / 10, -2 / 5, -1 / 2, 0, 1 / 2, 2 / 5, 3 / 10],
                        [-3 / 9, -2 / 4, -1 / 1, 0, 1 / 1, 2 / 4, 3 / 9],
                        [-3 / 10, -2 / 5, -1 / 2, 0, 1 / 2, 2 / 5, 3 / 10],
                        [-3 / 13, -2 / 8, -1 / 5, 0, 1 / 5, 2 / 8, 3 / 13],
                        [-3 / 18, -2 / 13, -1 / 10, 0, 1 / 10, 2 / 13, 3 / 18]])

# Для Scipy мы должны использовать перевернутое ядро (flipped)
flipped_edge_kernel = np.flip(edge_kernel)


def test_numba_speed_and_accuracy():
    print("Generating random image (500x500)...")
    # Создаем случайную картинку вместо загрузки файла
    image = np.random.rand(500, 500)

    print("--- Accuracy Check ---")
    # 1. Считаем через Scipy (эталон)
    # Важно: Scipy делает свертку, поэтому ему нужно перевернутое ядро,
    # чтобы получить тот же результат, что наша корреляция с прямым ядром.
    scipy_res = convolve2d(image, flipped_edge_kernel, mode='same', boundary='fill', fillvalue=0)

    # 2. Считаем через твою Numba
    numba_res = correlation_numba(edge_kernel, image)

    # Сравниваем
    if np.allclose(scipy_res, numba_res, atol=1e-5):
        print("✅ SUCCESS: Numba result matches Scipy result!")
    else:
        print("❌ FAIL: Results do not match.")
        print("Scipy sample:", scipy_res[0, 0])
        print("Numba sample:", numba_res[0, 0])

    print("\n--- Speed Check (Average of 10 runs) ---")

    def run_scipy():
        convolve2d(image, flipped_edge_kernel, mode='same', boundary='fill', fillvalue=0)

    def run_numba():
        correlation_numba(edge_kernel, image)

    # Первый запуск Numba всегда медленный (компиляция), прогреваем его
    run_numba()

    scipy_time = timeit.timeit(run_scipy, number=10) / 10
    numba_time = timeit.timeit(run_numba, number=10) / 10

    print(f"Scipy time: {scipy_time:.6f} sec")
    print(f"Numba time: {numba_time:.6f} sec")
    print(f"Speedup: {scipy_time / numba_time:.2f}x faster")


if __name__ == '__main__':
    test_numba_speed_and_accuracy()