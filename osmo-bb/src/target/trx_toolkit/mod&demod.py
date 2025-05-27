# import signal
# import argparse
# import random
# import select
# import sys
# import re
# import os
# import numpy as np
# from scipy.signal import convolve
# import time

# # --- Laurent-модуляция GMSK ---
# def modulate_burst_laurent(bits, sps=4):
# 	burst_len = 625
# 	MAX_BITS = 156

# 	# --- Подготовка битовой последовательности с tail bits ---
# 	bits = np.array(bits).astype(np.uint8)
# 	if len(bits) > MAX_BITS:
# 		raise ValueError(f"Слишком много бит (max {MAX_BITS})")
# 	# Добавляем два tail bits: 0 в начале и конце
# 	bits_with_tail = np.concatenate(([0, 0], bits, [0]))
# 	N = len(bits_with_tail)

# 	# --- C0 burst ---
# 	c0_burst = np.zeros(burst_len, dtype=np.float32)
# 	for i in range(N):
# 		c0_burst[i * sps] = 2 * (bits_with_tail[i] & 0x01) - 1.0

# 	# --- Вращение ---
# 	rot = np.exp(1j * np.arange(burst_len) * (np.pi/8))
# 	c0_burst_complex = c0_burst.astype(np.complex64) * rot

# 	# --- C1 burst ---
# 	c1_burst = np.zeros(burst_len, dtype=np.complex64)
# 	for i in range(2, N):
# 		idx = i * sps
# 		phase = 2 * ((bits_with_tail[i-1] & 0x01) ^ (bits_with_tail[i-2] & 0x01)) - 1.0
# 		c1_burst[idx] = c0_burst_complex[idx] * (1j * phase)

# 	# --- Импульсные отклики ---
# 	c0_pulse = np.array([
# 		0.0, 4.46348606e-03, 2.84385729e-02, 1.03184855e-01,
# 		2.56065552e-01, 4.76375085e-01, 7.05961177e-01, 8.71291644e-01,
# 		9.29453645e-01, 8.71291644e-01, 7.05961177e-01, 4.76375085e-01,
# 		2.56065552e-01, 1.03184855e-01, 2.84385729e-02, 4.46348606e-03
# 	], dtype=np.float32)
# 	c1_pulse = np.array([
# 		0.0, 8.16373112e-03, 2.84385729e-02, 5.64158904e-02,
# 		7.05463553e-02, 5.64158904e-02, 2.84385729e-02, 8.16373112e-03
# 	], dtype=np.float32)

# 	# --- Фильтрация (convolve) ---
# 	c0_shaped = convolve(c0_burst_complex, c0_pulse, mode='full')[:burst_len]
# 	c1_shaped = convolve(c1_burst, c1_pulse, mode='full')[:burst_len]
# 	return c0_shaped + c1_shaped

# def demodulate_burst_laurent(received_signal, sps=4):
# 	burst_len = 625

# 	# --- Откат вращения ---
# 	rev_rot = np.exp(-1j * np.arange(burst_len) * (np.pi/8))
# 	de_rotated = received_signal * rev_rot

# 	# --- Коррекция group delay (фильтра c0) ---
# 	# Фильтр c0 длиной 16, group delay ≈ (16 - 1) / 2 = 7.5 сэмплов
# 	# Обычно берут отсчёты с индекса 8 (или 7/9), потом через 4 сэмпла
# 	group_delay = 7

# 	# --- Сэмплирование символов ---
# 	symbol_samples = []
# 	for i in range(group_delay, burst_len, sps):
# 		symbol_samples.append(de_rotated[i].real)
# 	symbol_samples = np.array(symbol_samples)

# 	# --- Хард-декодер (decision) ---
# 	bits = (symbol_samples > 0).astype(int)
# 	# --- Софт-декодер (softbits) ---
# 	softbits = symbol_samples  # можно нормировать, если нужно
# 	return bits, softbits


# # data = [1] * 148
# # print("Исходные данные :", data)
# # modulated = modulate_burst_laurent(data)
# # demodulated = demodulate_burst_laurent(modulated)
# # print("Демодулированные данные :", demodulated)


# bits = np.zeros(148, dtype=np.uint8)
# bits[10] = 1  # только один бит "1", чтобы видеть, где он появляется после демодуляции

# modulated_signal = modulate_burst_laurent(bits)

# bits_demod, soft = demodulate_burst_laurent(modulated_signal, sps=4)

# print("Demodulated bits:", bits_demod[:30])
# print("Soft samples:", soft[:30])

##################################################################

import numpy as np
from scipy.signal import convolve

# --- Laurent-модуляция GMSK ---
def modulate_burst_laurent(bits, sps=4):
    burst_len = 625
    MAX_BITS = 156

    bits = np.array(bits).astype(np.uint8)
    if len(bits) > MAX_BITS:
        raise ValueError(f"Слишком много бит (max {MAX_BITS})")
    bits_with_tail = np.concatenate(([0, 0], bits, [0]))
    N = len(bits_with_tail)

    # C0 burst
    c0_burst = np.zeros(burst_len, dtype=np.float32)
    for i in range(N):
        c0_burst[i * sps] = 2 * (bits_with_tail[i] & 0x01) - 1.0

    # Вращение
    rot = np.exp(1j * np.arange(burst_len) * (np.pi/8))
    c0_burst_complex = c0_burst.astype(np.complex64) * rot

    # C1 burst
    c1_burst = np.zeros(burst_len, dtype=np.complex64)
    for i in range(2, N):
        idx = i * sps
        phase = 2 * ((bits_with_tail[i-1] & 0x01) ^ (bits_with_tail[i-2] & 0x01)) - 1.0
        c1_burst[idx] = c0_burst_complex[idx] * (1j * phase)

    # Импульсные отклики
    c0_pulse = np.array([
        0.0, 4.46348606e-03, 2.84385729e-02, 1.03184855e-01,
        2.56065552e-01, 4.76375085e-01, 7.05961177e-01, 8.71291644e-01,
        9.29453645e-01, 8.71291644e-01, 7.05961177e-01, 4.76375085e-01,
        2.56065552e-01, 1.03184855e-01, 2.84385729e-02, 4.46348606e-03
    ], dtype=np.float32)
    c1_pulse = np.array([
        0.0, 8.16373112e-03, 2.84385729e-02, 5.64158904e-02,
        7.05463553e-02, 5.64158904e-02, 2.84385729e-02, 8.16373112e-03
    ], dtype=np.float32)

    c0_shaped = convolve(c0_burst_complex, c0_pulse, mode='full')[:burst_len]
    c1_shaped = convolve(c1_burst, c1_pulse, mode='full')[:burst_len]
    return c0_shaped + c1_shaped

# --- Демодуляция ---
def demodulate_burst_laurent(received_signal, sps=4, group_delay=8, phase_offset=0.0, num_bits=148):
    burst_len = 625
    rev_rot = np.exp(-1j * (np.arange(burst_len) * (np.pi/8) + phase_offset))
    de_rotated = received_signal * rev_rot

    symbol_samples = [de_rotated[group_delay + i * sps].real for i in range(num_bits)]
    bits = (np.array(symbol_samples) > 0).astype(int)
    return bits, symbol_samples

# --- Тест ---
def test_phase_and_gd():
    bits = np.zeros(148, dtype=np.uint8)
    bits[10] = 1  # Ставим одну "1" для диагностики смещения

    modulated_signal = modulate_burst_laurent(bits)
    print("Исходные биты: ", bits[:30])

    print("Пробуем разные group_delay и phase_offset:")
    for group_delay in range(6, 12):
        for phase_offset in [0, np.pi/8, np.pi/4, np.pi/2, -np.pi/8, -np.pi/4, -np.pi/2]:
            bits_demod, soft = demodulate_burst_laurent(
                modulated_signal,
                sps=4,
                group_delay=group_delay,
                phase_offset=phase_offset,
                num_bits=30  # Для вывода
            )
            print(f"group_delay={group_delay}, phase_offset={phase_offset:.3f}")
            print("bits: ", bits_demod)
            print("soft:", np.round(soft, 3))
            print("-"*60)

if __name__ == "__main__":
    test_phase_and_gd()


group_delay=7, phase_offset=-1.571
bits:  [0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
soft: [-0.429 -0.01  -0.001 -0.001 -0.001 -0.001 -0.001 -0.001 -0.001 -0.001
 -0.001  0.233  0.694 -0.853 -0.019 -0.001 -0.001 -0.001 -0.001 -0.001
 -0.001 -0.001 -0.001 -0.001 -0.001 -0.001 -0.001 -0.001 -0.001 -0.001]

group_delay=9, phase_offset=1.571
bits:  [1 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
soft: [ 0.116 -0.001 -0.001 -0.001 -0.001 -0.001 -0.001 -0.001 -0.001 -0.001
 -0.019 -0.853  0.694  0.233 -0.001 -0.001 -0.001 -0.001 -0.001 -0.001
 -0.001 -0.001 -0.001 -0.001 -0.001 -0.001 -0.001 -0.001 -0.001 -0.001]

group_delay=11, phase_offset=-1.571
bits:  [0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
soft: [-0.01  -0.001 -0.001 -0.001 -0.001 -0.001 -0.001 -0.001 -0.001 -0.001
  0.233  0.694 -0.853 -0.019 -0.001 -0.001 -0.001 -0.001 -0.001 -0.001
 -0.001 -0.001 -0.001 -0.001 -0.001 -0.001 -0.001 -0.001 -0.001 -0.001]