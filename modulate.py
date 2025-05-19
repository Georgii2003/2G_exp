#!/usr/bin/env python3
"""
Реализация многоступенчатой GMSK‑модуляции по методу Laurent,
вдохновлённой исходным кодом sigProcLib (C++).
Эта реализация рассчитана на использование в FakeTRX.

Ключевые этапы:
  1. Преобразование входного битового потока в набор амплитудных значений (−1/+1).
  2. Расширение (upsampling): каждому биту соответствует sps отсчетов.
  3. Применение фазовой ротации с использованием предварительно рассчитанных таблиц.
  4. Реализация Laurent‑метода: разделение на две ветви (C0 и C1) с дифференциальным кодированием.
  5. Пульсовое формирование путём свёртки с фильтрами (имитирующими GSMPulse4: c0 и c1).
  6. Суммирование двух компонент для получения итогового комплексного сигнала.
  
Замечания:
  – Для упрощения предполагается, что входной битовый массив (numpy.array с 0 и 1)
    имеет не более 156 бит.
  – Длина итогового burst-а фиксирована (625 отсчетов для sps==4).
  – Все свёртки выполняются с помощью scipy.signal.convolve (в режиме ‘same’),
    что позволяет сохранить ту же длину.
  – Таблицы ротаций генерируются по схеме из C++: для 4 SPS используется шаг π/8,
    для 1 SPS – шаг π/2.
  – Векторные операции NumPy являются векторизованными и при наличии оптимизированных BLAS/MKL могут использовать SIMD.
"""

import numpy as np
from scipy.signal import convolve

# --- Константы и предварительно рассчитанные параметры ---
TABLESIZE = 1024
M_PI = np.pi

# Для данной реализации мы будем работать с 4 samples per symbol (sps)
DEFAULT_SPS = 4
BURST_LEN = 625   # Длина итогового burst-а для sps == 4
MAX_BITS   = 156  # Максимальное число бит для нормального burst-а

# Констелляция для EDGE (8-PSK) – приведена для примера (но в Laurent-методе GMSK её не используем)
psk8_table = np.array([
    -0.70710678+0.70710678j,
     0.0      -1.0j,
     0.0      +1.0j,
     0.70710678-0.70710678j,
    -1.0      +0.0j,
    -0.70710678-0.70710678j,
     0.70710678+0.70710678j,
     1.0      +0.0j,
], dtype=np.complex64)

# --- Таблицы ротаций (эмуляция работы GMSKRotation) ---

def init_gmsk_rotation_tables():
    """
    Генерирует таблицы поворотов для GMSK‑модуляции.
      * Для 4 SPS: длина = BURST_LEN, шаг фазы = (π/2)/4 = π/8.
      * Для 1 SPS: длина = 157, шаг фазы = π/2.
    Возвращает словарь с ключами: 'rot4', 'rev_rot4', 'rot1', 'rev_rot1'.
    """
    # Для 4 SPS
    rot4 = np.exp(1j * np.arange(BURST_LEN) * (M_PI/8))
    rev_rot4 = np.exp(-1j * np.arange(BURST_LEN) * (M_PI/8))
    # Для 1 SPS – длина 157 (как в C++ коде)
    len1 = 157
    rot1 = np.exp(1j * np.arange(len1) * (M_PI/2))
    rev_rot1 = np.exp(-1j * np.arange(len1) * (M_PI/2))
    return {
        'rot4': rot4,
        'rev_rot4': rev_rot4,
        'rot1': rot1,
        'rev_rot1': rev_rot1
    }

# Инициализируем таблицы (будут использованы при модуляции)
rotation_tables = init_gmsk_rotation_tables()

# --- Генерация pulse shaping фильтров (эмуляция GSMPulse4) ---
def generate_gsm_pulse(sps=4):
    """
    Для sps==4 генерирует фильтр pulse shaping для основной (C0) и вторичной (C1)
    компонент Laurent‑модуляции. Значения выбраны согласно исходному коду:
      c0 (16 отсчетов):
         0.0,
         4.46348606e-03,
         2.84385729e-02,
         1.03184855e-01,
         2.56065552e-01,
         4.76375085e-01,
         7.05961177e-01,
         8.71291644e-01,
         9.29453645e-01,
         8.71291644e-01,
         7.05961177e-01,
         4.76375085e-01,
         2.56065552e-01,
         1.03184855e-01,
         2.84385729e-02,
         4.46348606e-03
      c1 (8 отсчетов):
         0.0,
         8.16373112e-03,
         2.84385729e-02,
         5.64158904e-02,
         7.05463553e-02,
         5.64158904e-02,
         2.84385729e-02,
         8.16373112e-03
    Возвращает словарь с ключами 'c0' и 'c1' – оба как numpy-массивы типа float32.
    """
    if sps != 4:
        raise ValueError("В данной реализации поддерживается только sps==4")
    c0 = np.array([
         0.0,
         4.46348606e-03,
         2.84385729e-02,
         1.03184855e-01,
         2.56065552e-01,
         4.76375085e-01,
         7.05961177e-01,
         8.71291644e-01,
         9.29453645e-01,
         8.71291644e-01,
         7.05961177e-01,
         4.76375085e-01,
         2.56065552e-01,
         1.03184855e-01,
         2.84385729e-02,
         4.46348606e-03
    ], dtype=np.float32)
    c1 = np.array([
         0.0,
         8.16373112e-03,
         2.84385729e-02,
         5.64158904e-02,
         7.05463553e-02,
         5.64158904e-02,
         2.84385729e-02,
         8.16373112e-03
    ], dtype=np.float32)
    return {'c0': c0, 'c1': c1}

gsm_pulse = generate_gsm_pulse(sps=DEFAULT_SPS)

# --- Реализация Laurent‑модуляции GMSK ---
def modulate_burst_laurent(bits, sps=4):
    """
    Реализует Laurent‑модуляцию на основе входного битового потока.
    :param bits: numpy-массив, содержащий 0/1. Рекомендуется, чтобы len(bits) <= 156.
    :param sps: samples per symbol, должны быть 4.
    :return: numpy-массив комплексных отсчетов размером BURST_LEN (625).
    Алгоритм:
      а) Формирование вектора C0:
         - Паддинг: первый отсчет = -1, затем для каждого бита вставляем значение (2*bit - 1) через интервал sps,
           затем ещё один паддинг.
      б) Применение GMSK ротации к C0 с использованием таблицы rot4.
      в) Формирование вектора C1:
         - Начальная «магия»: на позиции 2*sps выставляется: C0 * (j * phase0), где phase0 = -1.
         - Для i = 2...len(bits)-1 вычисляется phase = 2*((bits[i-1] XOR bits[i-2])) - 1,
           и в соответствующей позиции C1 выставляется: C0 * (j * phase).
         - Последний отсчет C1 заполняется аналогично.
      г) Пульсовое формирование: свёртка C0 с фильтром c0 и C1 с фильтром c1 (из gsm_pulse).
      д) Суммирование полученных результатов даёт итоговый burst.
    """
    if len(bits) > MAX_BITS:
        raise ValueError("Слишком много бит для нормального burst-а (max %d)" % MAX_BITS)
    burst_len = BURST_LEN
    # Инициализируем C0 – создаём вектор длины burst_len.
    # Заполняем его значениями -1 (т.е. паддинг)
    c0_burst = -np.ones(burst_len, dtype=np.float32)
    # Определяем индексы для основных битов: начиная с индекса sps, далее через sps.
    main_indices = np.arange(sps, sps * (len(bits) + 1), sps)
    c0_burst[main_indices] = 2 * bits - 1.0  # мапинг 0-> -1, 1->+1
    # Паддинг в конце уже установлен (-1)

    # Приводим C0 к комплексному виду и применяем фазовую ротацию:
    # Используем предварительно рассчитанную таблицу для 4 SPS.
    c0_burst_complex = c0_burst.astype(np.complex64) * rotation_tables['rot4']

    # Инициализируем C1 как нулевой комплексный вектор
    c1_burst = np.zeros(burst_len, dtype=np.complex64)

    # Определяем указатели: в Python будем использовать индексацию
    # Начинаем с позиции = 2*sps (как в C++ коде)
    ptr = 2 * sps
    # «Начальная магия»: используем фиксированное значение, т.к. 0x01 XOR 0x01 = 0 → phase = 2*0-1 = -1.
    if ptr < burst_len:
        c1_burst[ptr] = c0_burst_complex[ptr] * (1j * -1)
    ptr += sps
    # Для i от 2 до len(bits)-1:
    for i in range(2, len(bits)):
        # Вычисляем phase по формуле: 2*((bits[i-1] XOR bits[i-2])) - 1.
        # XOR для 0/1 в Python: np.bitwise_xor.
        phase = 2 * (np.bitwise_xor(bits[i - 1], bits[i - 2])) - 1.0
        if ptr < burst_len:
            c1_burst[ptr] = c0_burst_complex[ptr] * (1j * phase)
        ptr += sps
    # «Завершающая магия»: для последнего бита (если возможно)
    if len(bits) >= 2 and ptr < burst_len:
        phase = 2 * (np.bitwise_xor(bits[-1], bits[-2])) - 1.0
        c1_burst[ptr] = c0_burst_complex[ptr] * (1j * phase)
    # Следующий шаг – пульсовое формирование (pulse shaping)
    # Выполняем свёртку C0 с фильтром c0, а C1 – с фильтром c1.
    # Для свёртки используем scipy.signal.convolve, mode='same' чтобы сохранить длину.
    c0_shaped = convolve(c0_burst_complex, gsm_pulse['c0'], mode='same')
    c1_shaped = convolve(c1_burst, gsm_pulse['c1'], mode='same')
    # Итоговый burst – сумма двух компонент:
    modulated_signal = c0_shaped + c1_shaped
    return modulated_signal

# --- Пример использования модулятора в FakeTRX ---
if __name__ == '__main__':
    # Для тестирования: сгенерируем случайный битовый поток длиной, например, 148 бит (типичная длина GSM-бурста)
    np.random.seed(42)  # для повторяемости
    bits = np.random.randint(0, 2, size=148).astype(np.int32)
    # Обрежем до MAX_BITS (максимум 156)
    bits = bits[:MAX_BITS]
    
    # Выполним модуляцию Laurent-методом
    modulated = modulate_burst_laurent(bits, sps=DEFAULT_SPS)
    
    # Сохраним результат в текстовый файл, чтобы FakeTRX мог его открыть
    # Для наглядности сохраняем две колонки: вещественная и мнимая части
    np.savetxt("osmocom-bb/src/target/trx_toolkit/soft_bits_modulated.txt",
               np.column_stack((modulated.real, modulated.imag)),
               fmt="%0.6f", header="Real   Imag")
    
    print("Модуляция завершена. Итоговый burst имеет %d отсчетов." % len(modulated))
    print("Данные сохранены в 'osmocom-bb/src/target/trx_toolkit/soft_bits_modulated.txt'")
