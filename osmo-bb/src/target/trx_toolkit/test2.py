import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt

# ===== Конфигурация =====
@dataclass
class Ray:
    PropagationDistance: float  # в метрах
    RelativePower: float = 1.0  # относительная мощность

# ===== Генерация случайных битов =====
def generate_random_bits(n_bits=148):
    return np.random.randint(0, 2, n_bits)

# ===== GMSK модуляция =====
def gmsk_modulate(bits):
    """GMSK модуляция без гауссовского фильтра (для упрощения)"""
    nrz = 2 * np.array(bits, dtype=np.float32) - 1.0
    phase = np.cumsum(nrz) * (np.pi / 2)
    signal = np.exp(1j * phase)
    return signal

# ===== Многолучевой канал + Path Loss =====
def multipath(signal, frequency, rays):
    CH_BANDWIDTH = 200e3  # GSM канал 200 кГц
    SPEED_LIGHT = 3e8     # скорость света
    ray_distances = np.array([ray.PropagationDistance for ray in rays])
    delays = np.round((ray_distances - ray_distances[0]) / SPEED_LIGHT / (1/CH_BANDWIDTH)).astype(int)
    max_delay = delays[-1]
    result = np.zeros(len(signal) + max_delay, dtype=complex)

    for i, delay in enumerate(delays):
        result[delay:delay+len(signal)] += signal * rays[i].RelativePower

    return result

def cost_hata(signal, tx_height, rx_height, distance_km, frequency):
    a_height_rx = (1.1 * np.log10(frequency/1e6) - 0.7) * rx_height - (1.56 * np.log10(frequency/1e6) - 0.8)
    PL = 46.3 + 33.9 * np.log10(frequency/1e6) - 13.82 * np.log10(tx_height) - a_height_rx + \
         (44.9 - 6.55 * np.log10(tx_height)) * np.log10(distance_km)
    attenuation = 10 ** (-PL / 20)
    return signal * attenuation

# ===== AWGN шум =====
def add_awgn(signal, snr_db):
    signal_power = np.mean(np.abs(signal)**2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*signal.shape) + 1j*np.random.randn(*signal.shape))
    return signal + noise

# ===== Демодуляция =====
def differential_demod(signal):
    """Дифференциальная демодуляция через разность фаз"""
    phase_diff = np.angle(signal[1:] * np.conj(signal[:-1]))
    return (phase_diff > 0).astype(np.uint8)

# ===== Визуализация =====
def plot_bitstream(bits, title="Bitstream"):
    plt.figure(figsize=(12, 2))
    plt.stem(bits)
    plt.title(title)
    plt.xlabel("Бит")
    plt.ylabel("Значение")
    plt.ylim(-0.1, 1.1)

def plot_signal(signal, title="Сигнал"):
    plt.figure(figsize=(12, 4))
    plt.plot(np.real(signal), label="I (реальная часть)")
    plt.plot(np.imag(signal), label="Q (мнимая часть)")
    plt.title(title)
    plt.xlabel("Отсчеты")
    plt.ylabel("Амплитуда")
    plt.legend()

def plot_constellation(signal, title="Созвездие", subsample=1):
    """Визуализация сигнала в виде диаграммы созвездия"""
    plt.figure(figsize=(6, 6))
    plt.scatter(np.real(signal[::subsample]), np.imag(signal[::subsample]), s=10, alpha=0.7)
    plt.title(title)
    plt.xlabel("I (реальная часть)")
    plt.ylabel("Q (мнимая часть)")
    plt.grid(True)

# ===== Основной пайплайн =====
def main():
    # Параметры
    N_BITS = 148
    SNR_DB = 10
    TX_HEIGHT = 30.0
    RX_HEIGHT = 1.5
    FREQUENCY = 900e6

    # Генерация битов
    bit_array = generate_random_bits(N_BITS)
    print(f"[RAW] Исходные биты: {bit_array.tolist()}")
    plot_bitstream(bit_array, title="Исходные биты")

    # GMSK модуляция
    modulated = gmsk_modulate(bit_array)
    print(f"[MOD] Модулированный сигнал: {modulated.tolist()}")
    plot_signal(modulated, title="Модулированный сигнал")
    plot_constellation(modulated, title="Созвездие: После модуляции", subsample=1)

    # Многолучевой канал
    rays = [
        Ray(PropagationDistance=1000.0, RelativePower=1.0),
        Ray(PropagationDistance=1200.0, RelativePower=0.7),
        Ray(PropagationDistance=1500.0, RelativePower=0.5)
    ]
    multipath_signal = multipath(modulated, FREQUENCY, rays)
    plot_signal(multipath_signal, title="Сигнал после многолучевости")
    plot_constellation(multipath_signal, title="Созвездие: После многолучевости", subsample=1)

    # Path Loss
    received_signal = cost_hata(multipath_signal, TX_HEIGHT, RX_HEIGHT, 1.5, FREQUENCY)
    plot_signal(received_signal, title="Сигнал после path loss")
    plot_constellation(received_signal, title="Созвездие: После path loss", subsample=1)

    # AWGN шум
    noisy_signal = add_awgn(received_signal, SNR_DB)
    plot_signal(noisy_signal, title="Сигнал после AWGN")
    plot_constellation(noisy_signal, title="Созвездие: После AWGN", subsample=1)

    # Демодуляция
    demodulated_bits = differential_demod(noisy_signal)
    print(f"[DEMOD] Демодулированные биты: {demodulated_bits.tolist()}")
    plot_bitstream(demodulated_bits, title="Демодулированные биты")

    # BER
    payload = bit_array[1:]  # Первый бит теряется из-за дифференциальной демодуляции
    demodulated = demodulated_bits[:len(payload)]
    ber = np.sum(payload != demodulated) / len(payload)
    print(f"[BER]: {ber:.4f}")

    # === Отображение всех графиков в конце ===
    plt.show()

# ===== Запуск =====
if __name__ == "__main__":
    main()


		# --- Получаем исходные биты ---
		# bit_array = np.array(list(map(int, src_msg.burst)), dtype=np.uint8)
		# print(f"[RAW] Burst bits: {bit_array.tolist()}")

		# # --- Дифференциальное кодирование для GMSK ---
		# diff = np.zeros_like(bit_array)
		# diff[0] = bit_array[0]
		# for i in range(1, len(bit_array)):
		# 	diff[i] = diff[i-1] ^ bit_array[i]
		# print(f"[DIFF] Diff-encoded bits: {diff.tolist()}")

		# --- GMSK модуляция ---
		# modulated_signal = gmsk_modulate(diff, sps=1, bt=0.3)

		# # --- Многолучёвость и затухание ---
		# rays = [
		# 	Ray(PropagationDistance=1000.0),
		# 	Ray(PropagationDistance=1200.0),
		# 	Ray(PropagationDistance=1500.0)
		# ]
		# frequency = 900e6
		# tx_height = 30.0
		# rx_height = 1.5

		# data_with_delays, ray_distances, rays_quantity = multipath(modulated_signal, frequency, rays)
		# data_with_pathloss = cost_hata(data_with_delays, tx_height, rx_height, ray_distances, frequency, rays_quantity)
		# received_signal = np.sum(data_with_pathloss, axis=0)[:len(modulated_signal)]
		# SNR_DB = 10
		# noisy_signal = add_awgn(received_signal, snr_db=SNR_DB)
		# noisy_signal = add_awgn(modulated_signal, snr_db=SNR_DB)

        # --- Дифференциальная декодировка ---
		# demod_dec = np.zeros_like(demod)
		# demod_dec[0] = demod[0]
		# for i in range(1, len(demod)):
		# 	demod_dec[i] = demod_dec[i-1] ^ demod[i]

		# # --- Анализ BER (сравниваем только diff[1:] и demod_dec) ---
		# ber = np.sum(diff[1:] != demod_dec) / (N-1)
		# print(f"[BER] Bit Error Rate (DIFF-chain, SNR={SNR_DB} dB): {ber:.4f} (по {N-1} битам)")

# def gmsk_modulate(bits, sps=4, bt=0.3):
#     nrz = 2 * np.array(bits, dtype=np.float32) - 1.0
#     up = np.zeros(len(nrz) * sps)
#     up[::sps] = nrz
#     # h = gaussian_filter(bt, sps, ntaps=4)
#     # filtered = lfilter(h, 1.0, up)
#     # phase = np.cumsum(filtered) * (np.pi/2) / sps
#     phase = np.cumsum(up) * (np.pi/2) / sps   # <-- чистая интеграция NRZ!
#     signal = np.exp(1j * phase)
#     return signal


