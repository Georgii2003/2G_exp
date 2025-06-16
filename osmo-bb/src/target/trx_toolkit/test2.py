import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.signal import lfilter, correlate
from scipy.interpolate import CubicSpline
import matplotlib.ticker as mticker

# ===== Конфигурация канала =====
@dataclass
class Ray:
    PropagationDistance: float
    RelativePower: float = 1.0

# ===== Функции для GMSK =====
def gaussian_filter(bt, sps, ntaps=4):
    n = ntaps * sps
    t = np.linspace(-ntaps/2, ntaps/2, n, endpoint=False)
    alpha = np.sqrt(np.log(2)) / (2 * np.pi * bt)
    h = np.exp(-0.5 * (t / alpha)**2)
    h /= np.sum(h)
    return h

def gmsk_modulate(bits, sps=4, bt=0.3):
    nrz = 2 * np.array(bits, dtype=np.float32) - 1.0
    up = np.zeros(len(nrz) * sps)
    up[::sps] = nrz
    h = gaussian_filter(bt, sps, ntaps=4)
    filtered = lfilter(h, 1.0, up)
    phase = np.cumsum(filtered) * (np.pi/2) / sps
    return np.exp(1j * phase)

def add_awgn(signal, snr_db):
    signal_power = np.mean(np.abs(signal)**2)
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    return signal + noise

def multipath(signal, frequency, rays, sps=4):
    CH_BANDWIDTH = 200e3
    SPEED_LIGHT = 3e8
    ray_distances = np.array([ray.PropagationDistance for ray in rays])
    delays = np.round((ray_distances - ray_distances[0]) / SPEED_LIGHT / (1/(CH_BANDWIDTH*sps))).astype(int)
    max_delay = delays[-1]
    result = np.zeros(len(signal) + max_delay, dtype=complex)
    
    # Суммирование лучей
    for i, delay in enumerate(delays):
        result[delay:delay+len(signal)] += signal * rays[i].RelativePower

    # Нормализация мощности сигнала
    total_power = sum(ray.RelativePower for ray in rays)
    result /= total_power  # Деление на суммарную мощность всех лучей

    return result[:len(signal)]

def cost_hata(signal, tx_height, rx_height, distance_km, frequency):
    a_height_rx = (1.1 * np.log10(frequency/1e6) - 0.7) * rx_height - (1.56 * np.log10(frequency/1e6) - 0.8)
    PL = 46.3 + 33.9 * np.log10(frequency/1e6) - 13.82 * np.log10(tx_height) - a_height_rx + \
         (44.9 - 6.55 * np.log10(tx_height)) * np.log10(distance_km)
    attenuation = 10 ** (-PL / 20)
    return signal * attenuation

def matched_filter(rx, sps=4, bt=0.3, ntaps=4):
    h = gaussian_filter(bt, sps, ntaps)
    return lfilter(h[::-1], 1.0, rx)

def differential_demod(signal):
    phase_diff = np.angle(signal[1:] * np.conj(signal[:-1]))
    return (phase_diff > 0).astype(np.uint8)

def time_sync(ref_bits, rx_signal, sps=4, bt=0.3, ntaps=4):
    """Точная временная синхронизация методом корреляции"""
    ref_mod = gmsk_modulate(ref_bits, sps=sps, bt=bt)
    h = gaussian_filter(bt, sps, ntaps)
    ref_mf = lfilter(h[::-1], 1.0, ref_mod)
    group_delay = len(h) // 2
    ref_mf = ref_mf[group_delay:]
    corr = correlate(rx_signal, ref_mf, mode='valid')
    corr_norm = np.abs(corr) / np.sqrt(np.sum(np.abs(ref_mf)**2) * np.cumsum(np.abs(rx_signal)**2)[len(ref_mf)-1:])
    offset = np.argmax(corr_norm)
    return offset + group_delay, ref_mf

# ===== Визуализация =====
def plot_bitstream(bits, title="Bitstream"):
    plt.figure(figsize=(12, 2))
    plt.stem(bits)
    plt.title(title)
    plt.xlabel("Бит")
    plt.ylabel("Значение")
    plt.ylim(-0.1, 1.1)

def plot_signal(signal, title="Сигнал", zoom=None, ylim=None):
    plt.figure(figsize=(12, 4))
    if zoom:
        plt.plot(np.real(signal)[:zoom], label="I (реальная часть)")
        plt.plot(np.imag(signal)[:zoom], label="Q (мнимая часть)")
    else:
        plt.plot(np.real(signal), label="I (реальная часть)")
        plt.plot(np.imag(signal), label="Q (мнимая часть)")
    plt.title(title)
    plt.xlabel("Отсчеты")
    plt.ylabel("Амплитуда")
    plt.legend()
    plt.grid(True)
    if ylim is not None:
        plt.ylim(*ylim)
    ax = plt.gca()
    formatter = mticker.FuncFormatter(lambda x, pos: '{:.1e}'.format(x))
    ax.yaxis.set_major_formatter(formatter)

def plot_constellation(signal, title="Созвездие", subsample=1):
    plt.figure(figsize=(6, 6))
    plt.scatter(np.real(signal[::subsample]), np.imag(signal[::subsample]), s=10, alpha=0.7)
    plt.title(title)
    plt.xlabel("I (реальная часть)")
    plt.ylabel("Q (мнимая часть)")
    plt.grid(True)

# ===== Основной пайплайн =====
def main():
    np.random.seed(42)
    N_BITS = 148
    SNR_DB = 10
    TX_HEIGHT = 300.0
    RX_HEIGHT = 1.5
    FREQUENCY = 900e6
    sps = 4
    bt = 0.3
    ts_len = 48

    # Синхрослово + полезная нагрузка
    syncword = np.random.randint(0, 2, ts_len)
    payload = np.random.randint(0, 2, N_BITS)
    bits_full = np.concatenate([syncword, payload])
    print(f"[RAW] Sync+Payload биты: {bits_full.tolist()}")
    # plot_bitstream(bits_full, title="Исходные биты (с синхрословом)")

    # GMSK модуляция
    modulated = gmsk_modulate(bits_full, sps=sps, bt=bt)
    plot_signal(modulated, title="Модулированный сигнал")
    # plot_constellation(modulated, title="Созвездие после модуляции", subsample=sps)

    # Многолучевой канал с 10 лучами
    rays = [
        Ray(PropagationDistance=1000.0, RelativePower=1.0),
        Ray(PropagationDistance=2500.0, RelativePower=0.8),
        Ray(PropagationDistance=3000.0, RelativePower=0.6),
        Ray(PropagationDistance=4500.0, RelativePower=0.5),
        Ray(PropagationDistance=5000.0, RelativePower=0.5),
        Ray(PropagationDistance=6500.0, RelativePower=0.3),
        Ray(PropagationDistance=7000.0, RelativePower=0.25),
        Ray(PropagationDistance=8500.0, RelativePower=0.2),
        Ray(PropagationDistance=9000.0, RelativePower=0.15),
        Ray(PropagationDistance=10500.0, RelativePower=0.1)
    ]
    
    # Анализ задержек
    CH_BANDWIDTH = 10e3
    SPEED_LIGHT = 3e8
    ray_distances = np.array([ray.PropagationDistance for ray in rays])
    delays = np.round((ray_distances - ray_distances[0]) / SPEED_LIGHT / (1/(CH_BANDWIDTH*sps))).astype(int)
    
    print("\n[МНОГОЛУЧЕВОСТЬ] Анализ задержек:")
    print(f"Скорость света: {SPEED_LIGHT} м/с")
    print(f"Длительность отсчета: {1/(CH_BANDWIDTH*sps)*1e6:.2f} мкс")
    print(f"Расстояние на отсчет: {SPEED_LIGHT/(CH_BANDWIDTH*sps):.0f} м")
    print("Лучи и их задержки:")
    for i, ray in enumerate(rays):
        delay_us = (ray.PropagationDistance - ray_distances[0]) / SPEED_LIGHT * 1e6
        print(f"  - Луч {i+1}: расстояние={ray.PropagationDistance:.0f} м, "
              f"относит. мощность={ray.RelativePower:.2f}, "
              f"задержка={delay_us:.2f} мкс, "
              f"отсчетов={delays[i]}")

    Y_LIM = (-2 * 1e-7, 2 * 1e-7)
    multipath_signal = multipath(modulated, FREQUENCY, rays, sps=sps)
    plot_signal(multipath_signal, title="После многолучевого канала")
    # plot_constellation(multipath_signal, title="Созвездие после многолучевости", subsample=sps)

    # Path loss
    received_signal = cost_hata(multipath_signal, TX_HEIGHT, RX_HEIGHT, 1.5, FREQUENCY)
    plot_signal(received_signal, title="После path loss (Cost-Hata)")
    # plot_constellation(received_signal, title="Созвездие после path loss", subsample=sps)

    # AWGN
    noisy_signal = add_awgn(received_signal, SNR_DB)
    plot_signal(noisy_signal, title="После AWGN")
    # plot_constellation(noisy_signal, title="Созвездие после AWGN", subsample=sps)

    # Matched filter
    mf = matched_filter(noisy_signal, sps=sps, bt=bt)

    # Синхронизация
    offset, ref_mf = time_sync(syncword, mf, sps=sps, bt=bt)
    start_idx = max(0, offset - 2*sps)
    end_idx = min(len(mf), start_idx + len(bits_full)*sps + 10*sps)
    mf_sync = mf[start_idx:end_idx]

    # Коррекция фазы
    ref_sync = gmsk_modulate(syncword, sps=sps, bt=bt)
    rx_sync_segment = mf_sync[offset-start_idx:offset-start_idx+len(ref_sync)]
    phase_offset = np.angle(np.sum(rx_sync_segment * np.conj(ref_sync)))
    mf_corr = mf_sync * np.exp(-1j*phase_offset)

    # Кубический ресемплинг
    t_original = np.arange(len(mf_corr))
    t_target = (offset - start_idx) + np.arange(0, len(bits_full)) * sps
    I_spline = CubicSpline(t_original, mf_corr.real)
    Q_spline = CubicSpline(t_original, mf_corr.imag)
    resampled = I_spline(t_target) + 1j*Q_spline(t_target)
    # plot_constellation(resampled, title="Созвездие после ресемплинга", subsample=1)

    # Демодуляция
    demod_bits = differential_demod(resampled)
    # plot_bitstream(demod_bits, title="Демодулированные биты")

    # BER (только payload)
    payload_tx = bits_full[ts_len+1:]        # +1 из-за дифференциальной демодуляции
    payload_rx = demod_bits[ts_len:ts_len+len(payload_tx)]
    errors = np.sum(payload_tx != payload_rx)
    ber = errors / len(payload_tx) if len(payload_tx) > 0 else 0
    print(f"\n[BER]: {ber:.4f} (ошибок: {errors}/{len(payload_tx)})")

    plt.tight_layout()
    plt.show()

# ===== Запуск =====
if __name__ == "__main__":
    main()



