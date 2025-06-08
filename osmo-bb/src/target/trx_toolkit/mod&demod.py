import numpy as np
from scipy.signal import lfilter
import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass

@dataclass
class Ray:
    PropagationDistance: float  # расстояние в метрах

def gaussian_filter(bt, sps, ntaps=4):
    n = ntaps * sps
    t = np.linspace(-ntaps/2, ntaps/2, n)
    alpha = np.sqrt(np.log(2)) / (2 * np.pi * bt)
    h = np.exp(-0.5 * (t / alpha)**2)
    h /= np.sum(h)
    return h

from scipy.signal import convolve

def modulate_burst_laurent(bits, sps=4):
    burst_len = 625
    MAX_BITS = 156

    bits = np.array(bits).astype(np.uint8)
    if len(bits) > MAX_BITS:
        raise ValueError(f"Слишком много бит (max {MAX_BITS})")
    # Добавляем два tail bits: 0 в начале и один в конец
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

def demodulate_burst_laurent(received_signal, sps=4, group_delay=7, phase_offset=-np.pi/2, num_bits=148):
    burst_len = 625
    rev_rot = np.exp(-1j * (np.arange(burst_len) * (np.pi/8) + phase_offset))
    de_rotated = received_signal * rev_rot
    symbol_samples = [de_rotated[group_delay + i * sps].real for i in range(num_bits)]
    bits = (np.array(symbol_samples) > 0).astype(int)
    return bits, np.array(symbol_samples)


def multipath(msg_complex, frequency, rays):
		rays_quantity = len(rays)
		CH_bandwidth = 10e6  # 10 МГц
		duration_sample = 1 / CH_bandwidth
		speed_light = 3e8

		ray_distances = np.array([ray.PropagationDistance / 1000 for ray in rays])  # в км

		idx_min = np.argmin(ray_distances)
		ray_distances = np.concatenate(([ray_distances[idx_min]], np.delete(ray_distances, idx_min)))
		rays_sorted = [rays[idx_min]] + [rays[i] for i in range(len(rays)) if i != idx_min]
		
		delays = np.zeros(rays_quantity, dtype=int)
		for i in range(rays_quantity):
			delays[i] = int(round((ray_distances[i] - ray_distances[0]) * 1000 / (speed_light * duration_sample)))
		max_delay = delays.max()
		data_with_delays = np.zeros((rays_quantity, len(msg_complex) + max_delay), dtype=complex)
		for i in range(rays_quantity):
			if delays[i] == 0:
				data_with_delays[i, :len(msg_complex)] = msg_complex
			else:
				data_with_delays[i, delays[i]:delays[i]+len(msg_complex)] = msg_complex
		return data_with_delays, ray_distances, rays_quantity

def cost_hata(data_with_delays, tx_height, rx_height, ray_distances, frequency, rays_quantity):
		data_PL = np.zeros_like(data_with_delays, dtype=complex)
		for i in range(rays_quantity):
			a_height_rx = (1.1 * np.log10(frequency/1e6) - 0.7) * rx_height - (1.56 * np.log10(frequency/1e6) - 0.8)
			PL = 46.3 + 33.9 * np.log10(frequency/1e6) - 13.82 * np.log10(tx_height) - a_height_rx + \
				(44.9 - 6.55 * np.log10(tx_height)) * np.log10(ray_distances[i])
			attenuation = 10 ** (-PL / 10)
			data_PL[i, :] = data_with_delays[i, :] * attenuation
		return data_PL

def add_awgn(signal, snr_db):
    signal_power = np.mean(np.abs(signal)**2)
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*signal.shape) + 1j*np.random.randn(*signal.shape))
    return signal + noise


# --- Получаем исходные биты ---
# 148 бит для normal burst
num_bits = 148
bit_array = np.zeros(num_bits, dtype=int)
bit_array[0] = 1
print(f"[RAW] Burst data: {bit_array}")
# --- GMSK модуляция ---
modulated_signal = modulate_burst_laurent(bit_array, sps=4)
# =========================

# === МНОГОЛУЧЕВОСТЬ И ЗАТУХАНИЕ КАНАЛА  ===
# channel_signal = mod.channel() - длина, высота, паслосы, многолучёвость costhata
# Пример параметров лучей, частоты и антенн:
rays = [
    Ray(PropagationDistance=1000.0),   # 1000 м
    Ray(PropagationDistance=1200.0),   # 1200 м
    Ray(PropagationDistance=1500.0)    # 1500 м
]
frequency = 900e6  # 900 МГц, для GSM
tx_height = 30.0   # высота антенны передатчика, м
rx_height = 1.5    # высота антенны приемника, м

# # 1. Эффект многолучёвости
# data_with_delays, ray_distances, rays_quantity = multipath(modulated_signal, frequency, rays)

# # 2. Эффект затухания Cost Hata
# data_with_pathloss = cost_hata(data_with_delays, tx_height, rx_height, ray_distances, frequency, rays_quantity)

# # 3. Итоговый принимаемый сигнал выглядит как сумма по лучам
# # received_signal = np.sum(data_with_pathloss, axis=0)[:625]
# # === AWGN ===
# SNR_DB = 10
# noisy_signal = add_awgn(received_signal, snr_db=SNR_DB)
# demodulated_bits = mod.demodulate()
# bits, softbits = demodulate_burst_laurent(received_signal, sps=4)
# print("[DEMOD] bits:", bits)
# print("[DEMOD] softbits:", softbits[:10])

# --- GMSK демодуляция ---
# offset=2 для sps=4 и стандартного фильтра
# for gd in range(6, 12):
#     for po in [0, np.pi/8, np.pi/4, np.pi/2, -np.pi/8, -np.pi/4, -np.pi/2]:
#         demod_bits, _ = demodulate_burst_laurent(modulated_signal, sps=4, group_delay=gd, phase_offset=po, num_bits=148)
#         ber = np.sum(np.array(bit_array[:148]) != demod_bits) / 148
#         print(f"group_delay={gd}, phase_offset={po:.3f}, BER={ber:.4f}")
demod_bits, _ = demodulate_burst_laurent(modulated_signal, sps=4, group_delay=7, phase_offset=-np.pi/2, num_bits=148)
print("BER:", np.sum(bit_array != demod_bits) / 148)
print("demod:", demod_bits)
# demod_bits, soft = demodulate_burst_laurent(noisy_signal, sps=4, group_delay=7, phase_offset=-np.pi/2, num_bits=148)
# print("[DEMOD] bits:", demod_bits)
# # --- Анализ (BER, лог) ---
# bit_array = np.array(bit_array[:148])
# ber = np.sum(bit_array != demod_bits) / 148
# print(f"[BER] Bit Error Rate: {ber:.4f} при SNR={SNR_DB} дБ")