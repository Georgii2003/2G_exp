import numpy as np
from scipy.signal import lfilter, convolve
from dataclasses import dataclass
import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt

# ===== Конфигурация =====
@dataclass
class Config:
    sps: int = 4          # Сэмплов на символ
    bt: float = 0.3       # Полосовой коэффициент
    preamble_len: int = 64  # Длина преамбулы
    n_bits: int = 148     # Количество бит в полезной нагрузке

@dataclass
class Ray:
    PropagationDistance: float  # Расстояние распространения луча в метрах

# ===== GMSK модуляция =====
def gmsk_modulate(bits, sps=4, bt=0.3):
    nrz = 2 * np.array(bits, dtype=np.float32) - 1.0
    up = np.zeros(len(nrz) * sps)
    up[::sps] = nrz
    
    # Гауссовский фильтр
    n = 4 * sps
    t = np.linspace(-n/sps/2, n/sps/2, n)
    alpha = np.sqrt(np.log(2)) / (2 * np.pi * bt)
    h = np.exp(-0.5 * (t / alpha)**2)
    h /= np.sum(h)
    
    filtered = lfilter(h, 1.0, up)
    phase = np.cumsum(filtered) * (np.pi/2) / sps
    signal = np.exp(1j * phase)
    return signal

# ===== Улучшенная демодуляция =====
def differential_demodulator(signal, sps):
    """Дифференциальная демодуляция через разность фаз символов"""
    # Синхронизация по энергии сигнала
    energy = np.convolve(np.abs(signal)**2, np.ones(sps), 'valid')
    start_idx = np.argmax(energy)
    
    # Выборка символов
    synced = signal[start_idx::sps]
    
    # Разность фаз между символами
    phase = np.unwrap(np.angle(synced))
    phase_diff = np.diff(phase)
    
    # Демодуляция
    demod_bits = (phase_diff > 0).astype(int)
    return demod_bits

# ===== Улучшенный пайплайн =====
def sdr_full_pipeline(bits, sps=4, bt=0.3, rays=None, frequency=900e6,
                      tx_height=30.0, rx_height=1.5, SNR_DB=5, preamble_len=32,
                      gardner_gain=0.01):
    """Полный пайплайн обработки сигнала с улучшениями"""
    preamble = np.ones(preamble_len, dtype=int)
    payload = bits
    tx_bits = np.concatenate([preamble, payload])

    # Модуляция
    mod_signal = gmsk_modulate(tx_bits, sps=sps, bt=bt)

    # Мультипуть+PL
    if rays is not None:
        data_with_delays, ray_distances, rays_quantity = multipath(mod_signal, frequency, rays)
        data_with_pathloss = cost_hata(data_with_delays, tx_height, rx_height, ray_distances, frequency, rays_quantity)
        channel_signal = np.sum(data_with_pathloss, axis=0)[:len(mod_signal)]
    else:
        channel_signal = mod_signal

    # AWGN
    noisy_signal = add_awgn(channel_signal, SNR_DB)

    # Поиск преамбулы в комплексном сигнале
    preamble_signal = gmsk_modulate(preamble, sps=sps, bt=bt)
    pre_idx = find_preamble_complex(noisy_signal, preamble_signal)

    # Вырезаем сигнал после преамбулы
    signal_after_preamble = noisy_signal[pre_idx : pre_idx + len(tx_bits)*sps]

    # Обрезаем до длины, кратной sps
    valid_len = (len(signal_after_preamble) // sps) * sps
    signal_after_preamble = signal_after_preamble[:valid_len]

    # Разбиваем на символы и усредняем
    symbols = signal_after_preamble.reshape(-1, sps)
    symbol_avg = np.mean(symbols, axis=1)

    # Дифференциальная демодуляция
    phase = np.unwrap(np.angle(symbol_avg))
    phase_diff = np.diff(phase)
    demod_bits = (phase_diff > 0).astype(int)

    # Вырезаем полезную нагрузку
    payload_bits = demod_bits[preamble_len-1 : preamble_len-1 + len(payload)]

    # Расчет BER
    n = min(len(payload), len(payload_bits))
    ber = np.sum(payload[:n] != payload_bits[:n]) / n if n > 0 else 1.0

    return payload_bits, ber, payload_bits

# ===== Path Loss + Multipath =====
def multipath(msg_complex, frequency, rays):
    rays_quantity = len(rays)
    CH_bandwidth = 10e6
    duration_sample = 1 / CH_bandwidth
    speed_light = 3e8
    ray_distances = np.array([ray.PropagationDistance / 1000 for ray in rays])
    idx_min = np.argmin(ray_distances)
    ray_distances = np.concatenate(([ray_distances[idx_min]], np.delete(ray_distances, idx_min)))
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

# ===== Вспомогательные функции =====
def find_preamble_complex(signal, preamble_signal):
    """Поиск преамбулы в комплексном сигнале"""
    corr = np.correlate(signal, preamble_signal, mode='valid')
    idx = np.argmax(np.abs(corr))
    return min(idx, len(signal) - len(preamble_signal))

# ===== ТЕСТИРОВАНИЕ С РАЗНЫМИ ПАРАМЕТРАМИ =====
def run_parameter_sweep():
    """Запуск тестов с разными параметрами и сохранение результатов"""
    # Базовые параметры
    sps = 4
    bt = 0.3
    n_bits = 148
    frequency = 900e6
    tx_height = 30.0
    rx_height = 1.5
    SNR_DB = 5

    # Параметры для перебора
    preamble_lengths = [16, 24, 32, 48, 64]  # Переносим сюда, а не из config
    gardner_gains = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    snr_values = [0, 5, 10, 15, 20]
    rays_configs = [
        None,  # Без канала
        [Ray(1000.0)],  # Прямая видимость
        [Ray(1000.0), Ray(1200.0)],  # Два луча
        [Ray(1000.0), Ray(1200.0), Ray(1500.0)]  # Три луча
    ]

    # Создание файла результатов
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ber_results_{timestamp}.csv"

    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['preamble_len', 'gardner_gain', 'SNR_DB', 'rays_count', 'BER']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        total_tests = len(preamble_lengths) * len(gardner_gains) * len(snr_values) * len(rays_configs)
        test_count = 0
        print(f"Starting parameter sweep ({total_tests} tests)...")

        for preamble_len in preamble_lengths:
            for gain in gardner_gains:
                for snr in snr_values:
                    for rays in rays_configs:
                        test_count += 1
                        payload = np.random.randint(0, 2, n_bits)
                        rays_count = 0 if rays is None else len(rays)

                        try:
                            _, ber, _ = sdr_full_pipeline(
                                payload,
                                sps=sps,
                                bt=bt,
                                rays=rays,
                                frequency=frequency,
                                tx_height=tx_height,
                                rx_height=rx_height,
                                SNR_DB=snr,
                                preamble_len=preamble_len,
                                gardner_gain=gain
                            )
                            writer.writerow({
                                'preamble_len': preamble_len,
                                'gardner_gain': gain,
                                'SNR_DB': snr,
                                'rays_count': rays_count,
                                'BER': ber
                            })
                            if test_count % 10 == 0:
                                print(f"Completed {test_count}/{total_tests} tests")
                        except Exception as e:
                            print(f"Error in test {test_count}: {str(e)}")
                            writer.writerow({
                                'preamble_len': preamble_len,
                                'gardner_gain': gain,
                                'SNR_DB': snr,
                                'rays_count': rays_count,
                                'BER': 1.0
                            })
    print(f"Parameter sweep completed. Results saved to {filename}")
    

# ===== ОСНОВНОЙ БЛОК =====
if __name__ == "__main__":
    # Запуск тестирования с разными параметрами
    run_parameter_sweep()