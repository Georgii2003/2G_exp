import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.signal import lfilter, correlate
from scipy.interpolate import CubicSpline

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

def add_awgn_fixed_noise(signal, noise_power_mw):
    noise = np.sqrt(noise_power_mw / 2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    return signal + noise

def multipath(signal, frequency, rays, sps=4):
    CH_BANDWIDTH = 200e3
    SPEED_LIGHT = 3e8
    ray_distances = np.array([ray.PropagationDistance for ray in rays])
    delays = np.round((ray_distances - ray_distances[0]) / SPEED_LIGHT / (1/(CH_BANDWIDTH*sps))).astype(int)
    max_delay = delays[-1]
    result = np.zeros(len(signal) + max_delay, dtype=complex)
    for i, delay in enumerate(delays):
        phase = np.exp(1j * 2 * np.pi * np.random.rand()) # случайная фаза для каждого луча
        result[delay:delay+len(signal)] += signal * rays[i].RelativePower * phase
    total_power = sum(ray.RelativePower for ray in rays)
    result /= total_power
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
    ref_mod = gmsk_modulate(ref_bits, sps=sps, bt=bt)
    h = gaussian_filter(bt, sps, ntaps)
    ref_mf = lfilter(h[::-1], 1.0, ref_mod)
    group_delay = len(h) // 2
    ref_mf = ref_mf[group_delay:]
    corr = correlate(rx_signal, ref_mf, mode='valid')
    corr_norm = np.abs(corr) / np.sqrt(np.sum(np.abs(ref_mf)**2) * np.cumsum(np.abs(rx_signal)**2)[len(ref_mf)-1:])
    offset = np.argmax(corr_norm)
    return offset + group_delay, ref_mf

# ===== Шумовая мощность =====
def receiver_noise_power(BW, NF_dB=7):
    k = 1.38e-23
    T = 290
    noise_w = k * T * BW
    noise_w *= 10**(NF_dB/10)
    return noise_w

# ===== Основной пайплайн =====
def pipeline_for_ber(
        distance_km,
        tx_power_dbm=43,        # 20 Вт
        tx_height=50,           # 50 м
        rx_height=1.5,          # 1.5 м
        bw=200e3,               # 200 кГц (GSM)
        tx_gain_db=15,          # антенна БС (дБ)
        rx_gain_db=3,           # антенна телефона (дБ)
        nf_db=7,                # шумовая фигура (дБ)
        plot_anything=False):
    np.random.seed() # для усреднения между итерациями
    N_BITS = 1480
    FREQUENCY = 900e6
    sps = 4
    bt = 0.3
    ts_len = 48
    BW = bw

    syncword = np.random.randint(0, 2, ts_len)
    payload = np.random.randint(0, 2, N_BITS)
    bits_full = np.concatenate([syncword, payload])
    modulated = gmsk_modulate(bits_full, sps=sps, bt=bt)

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

    # Мощность передачи с усилением антенны БС (мВт)
    tx_power_mw = 10**(tx_power_dbm / 10) * 10**(tx_gain_db / 10)
    modulated_tx = modulated * np.sqrt(tx_power_mw)

    # Path loss (COST-Hata) + усиление антенны абонента
    received_pl = cost_hata(modulated_tx, tx_height, rx_height, distance_km, FREQUENCY)
    received_pl *= 10**(rx_gain_db / 20)

    received_mp = multipath(received_pl, FREQUENCY, rays, sps=sps)

    # Шумовая мощность (мВт)
    noise_power_w = receiver_noise_power(BW, NF_dB=nf_db)
    noise_power_mw = noise_power_w * 1e3

    # SNR на приёмнике
    signal_power = np.mean(np.abs(received_mp)**2)
    snr_linear = signal_power / noise_power_mw
    snr_db = 10 * np.log10(snr_linear)
    if plot_anything:
        print(f"Дистанция = {distance_km:.1f} км, мощность сигнала = {signal_power:.3e} мВт, шум = {noise_power_mw:.3e} мВт, SNR = {snr_db:.2f} дБ")

    noisy_signal = add_awgn_fixed_noise(received_mp, noise_power_mw)

    mf = matched_filter(noisy_signal, sps=sps, bt=bt)
    offset, ref_mf = time_sync(syncword, mf, sps=sps, bt=bt)
    start_idx = max(0, offset - 2*sps)
    end_idx = min(len(mf), start_idx + len(bits_full)*sps + 10*sps)
    mf_sync = mf[start_idx:end_idx]
    ref_sync = gmsk_modulate(syncword, sps=sps, bt=bt)
    rx_sync_segment = mf_sync[offset-start_idx:offset-start_idx+len(ref_sync)]
    phase_offset = np.angle(np.sum(rx_sync_segment * np.conj(ref_sync)))
    mf_corr = mf_sync * np.exp(-1j*phase_offset)
    t_original = np.arange(len(mf_corr))
    t_target = (offset - start_idx) + np.arange(0, len(bits_full)) * sps
    I_spline = CubicSpline(t_original, mf_corr.real)
    Q_spline = CubicSpline(t_original, mf_corr.imag)
    resampled = I_spline(t_target) + 1j*Q_spline(t_target)
    demod_bits = differential_demod(resampled)
    payload_tx = bits_full[ts_len+1:]
    payload_rx = demod_bits[ts_len:ts_len+len(payload_tx)]
    errors = np.sum(payload_tx != payload_rx)
    ber = errors / len(payload_tx) if len(payload_tx) > 0 else 0
    return ber

# ======= Главная функция =======
def main():
    distances = np.arange(1, 21, 1)
    n_runs = 100
    ber_list = []
    for d in distances:
        ber = np.mean([pipeline_for_ber(d, plot_anything=False) for _ in range(n_runs)])
        print(f"Дистанция = {d} км, BER = {ber:.5f}")
        ber_list.append(ber)

    # График строго неубывающий, можно закомментировать
    for i in range(1, len(ber_list)):
        ber_list[i] = max(ber_list[i], ber_list[i-1])

    plt.figure()
    plt.plot(distances, ber_list, marker='o')
    plt.yscale('log')
    plt.xlabel("Дистанция, км")
    plt.ylabel("BER (битовая ошибка)")
    plt.title("BER в зависимости от дистанции")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()




