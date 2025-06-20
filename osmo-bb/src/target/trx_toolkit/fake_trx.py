
APP_CR_HOLDERS = [("2017-2019", "Vadim Yanitskiy <axilirator@gmail.com>")]

import logging as log
import signal
import argparse
import random
import select
import sys
import re
import os
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.signal import lfilter, correlate
from scipy.interpolate import CubicSpline
import time

from app_common import ApplicationBase
from burst_fwd import BurstForwarder
from transceiver import Transceiver
from data_msg import Modulation
from clck_gen import CLCKGen
from trx_list import TRXList
from fake_pm import FakePM
from gsm_shared import *

from dataclasses import dataclass

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
    CH_BANDWIDTH = 5e3
    SPEED_LIGHT = 3e8
    ray_distances = np.array([ray.PropagationDistance for ray in rays])
    delays = np.round((ray_distances - ray_distances[0]) / SPEED_LIGHT / (1/(CH_BANDWIDTH*sps))).astype(int)
    max_delay = delays[-1]
    result = np.zeros(len(signal) + max_delay, dtype=complex)
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

def add_awgn_fixed_noise(signal, noise_power_mw):
    """Добавить к сигналу комплексный AWGN с заданной мощностью (мВт)"""
    noise = np.sqrt(noise_power_mw / 2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    return signal + noise

def receiver_noise_power(BW, NF_dB=7):
    """Расчет реальной шумовой мощности приемника"""
    k = 1.38e-23
    T = 290
    noise_w = k * T * BW
    noise_w *= 10**(NF_dB/10)
    return noise_w

class FakeTRX(Transceiver):
	""" Fake transceiver with RF path (burst loss, RSSI, TA, ToA) simulation.

	== ToA / RSSI measurement simulation

	Since this is a virtual environment, we can simulate different
	parameters of the physical RF interface:

	  - ToA (Timing of Arrival) - measured difference between expected
	    and actual time of burst arrival in units of 1/256 of GSM symbol
	    periods. A pair of both base and threshold values defines a range
	    of ToA value randomization:

	      from (toa256_base - toa256_rand_threshold)
	        to (toa256_base + toa256_rand_threshold).

	  - RSSI (Received Signal Strength Indication) - measured "power" of
	    the signal (per burst) in dBm. A pair of both base and threshold
	    values defines a range of RSSI value randomization:

	      from (rssi_base - rssi_rand_threshold)
	        to (rssi_base + rssi_rand_threshold).

	  - C/I (Carrier-to-Interference ratio) - value in cB (centiBels),
	    computed from the training sequence of each received burst, by
	    comparing the "ideal" training sequence with the actual one.
	    A pair of both base and threshold values defines a range of
	    C/I randomization:

	      from (ci_base - ci_rand_threshold)
	        to (ci_base + ci_rand_threshold).

	Please note that the randomization is optional and disabled by default.

	== Timing Advance handling

	The BTS is using ToA measurements for UL bursts in order to calculate
	Timing Advance value, that is then indicated to a MS, which in its turn
	shall apply this value to the transmitted signal in order to compensate
	the delay. Basically, every burst is transmitted in advance defined by
	the indicated Timing Advance value. The valid range is 0..63, where
	each unit means one GSM symbol advance. The actual Timing Advance value
	is set using SETTA control command from MS. By default, it's set to 0.

	== Path loss simulation

	=== Burst dropping

	In some cases, e.g. due to a weak signal or high interference, a burst
	can be lost, i.e. not detected by the receiver. This can also be
	simulated using FAKE_DROP command on the control interface:

	  - burst_drop_amount - the amount of DL/UL bursts
	    to be dropped (i.e. not forwarded towards the MS/BTS),

	  - burst_drop_period - drop a DL/UL burst if its (fn % period) == 0.

	== Configuration

	All simulation parameters mentioned above can be changed at runtime
	using the commands with prefix 'FAKE_' on the control interface.
	All of them are handled by our custom CTRL command handler.

	"""
	
	NOMINAL_TX_POWER_DEFAULT = 50 # dBm
	TX_ATT_DEFAULT = 0 # dB
	PATH_LOSS_DEFAULT = 110 # dB

	TOA256_BASE_DEFAULT = 0
	CI_BASE_DEFAULT = 90

	# Default values for NOPE / IDLE indications
	TOA256_NOISE_DEFAULT = 0
	RSSI_NOISE_DEFAULT = -110
	CI_NOISE_DEFAULT = -30

	def __init__(self, *trx_args, **trx_kwargs):
			Transceiver.__init__(self, *trx_args, **trx_kwargs)
			# fake RSSI is disabled by default
			self.fake_rssi_enabled = False
			self.rf_muted = False
			# Actual ToA, RSSI, C/I, TA values
			self.tx_power_base = self.NOMINAL_TX_POWER_DEFAULT
			self.tx_att_base = self.TX_ATT_DEFAULT
			self.toa256_base = self.TOA256_BASE_DEFAULT
			self.rssi_base = self.NOMINAL_TX_POWER_DEFAULT - self.TX_ATT_DEFAULT - self.PATH_LOSS_DEFAULT
			self.ci_base = self.CI_BASE_DEFAULT
			self.ta = 0
			# ToA, RSSI, C/I randomization thresholds
			self.toa256_rand_threshold = 0
			self.rssi_rand_threshold = 0
			self.ci_rand_threshold = 0
			# Path loss simulation (burst dropping)
			self.burst_drop_amount = 0
			self.burst_drop_period = 1

			# Переменная для хранения сырых данных
			self.raw_data = None
			self.tx_power_dbm = 43       # Мощность передачи БС в dBm (20 Вт)
			self.tx_gain_db = 15         # Усиление антенны БС в dB
			self.rx_gain_db = 3          # Усиление антенны телефона в dB
			self.nf_db = 7               # Шумовая фигура приемника в dB
			self.tx_height = 50          # Высота БС в метрах
			self.rx_height = 1.5         # Высота телефона в метрах
			self.freq = 900e6            # Частота в Гц (GSM-900)
			self.bw = 200e3              # Ширина полосы в Гц (GSM)
			self.distance_km = 5       # Дистанция между БС и телефоном в км

	@property
	def toa256(self):
		# Check if randomization is required
		if self.toa256_rand_threshold == 0:
			return self.toa256_base

		# Generate a random ToA value in required range
		toa256_min = self.toa256_base - self.toa256_rand_threshold
		toa256_max = self.toa256_base + self.toa256_rand_threshold
		return random.randint(toa256_min, toa256_max)

	@property
	def rssi(self):
		# Check if randomization is required
		if self.rssi_rand_threshold == 0:
			return self.rssi_base

		# Generate a random RSSI value in required range
		rssi_min = self.rssi_base - self.rssi_rand_threshold
		rssi_max = self.rssi_base + self.rssi_rand_threshold
		return random.randint(rssi_min, rssi_max)

	@property
	def tx_power(self):
		return self.tx_power_base - self.tx_att_base

	@property
	def ci(self):
		# Check if randomization is required
		if self.ci_rand_threshold == 0:
			return self.ci_base

		# Generate a random C/I value in required range
		ci_min = self.ci_base - self.ci_rand_threshold
		ci_max = self.ci_base + self.ci_rand_threshold
		return random.randint(ci_min, ci_max)

	def sim_burst_drop(self, msg):
		# Check if dropping is required
		if self.burst_drop_amount == 0:
			return False

		if msg.fn % self.burst_drop_period == 0:
			log.info("(%s) Simulation: dropping burst (fn=%u %% %u == 0)"
					% (self, msg.fn, self.burst_drop_period))
			self.burst_drop_amount -= 1
			return True
		return False

	def _handle_data_msg_v1(self, src_msg, msg):
		# C/I (Carrier-to-Interference ratio)
		msg.ci = self.ci

		# Pick modulation type by burst length
		bl = len(src_msg.burst)
		msg.mod_type = Modulation.pick_by_bl(bl)

		# Pick TSC (Training Sequence Code) and TSC set
		if msg.mod_type is Modulation.ModGMSK:
			ss = TrainingSeqGMSK.pick(src_msg.burst)
			msg.tsc = ss.tsc if ss is not None else 0
			msg.tsc_set = ss.tsc_set if ss is not None else 0
		else:  # TODO: other modulation types (at least 8-PSK)
			msg.tsc_set = 0
			msg.tsc = 0

	# Takes (partially initialized) TRXD Rx message,
	# simulates RF path parameters (such as RSSI),
	# and sends towards the L1
	def handle_data_msg(self, src_trx, src_msg, msg):
		if self.rf_muted:
			msg.nope_ind = True
		elif not msg.nope_ind:
			msg.nope_ind = self.sim_burst_drop(msg)
		if msg.nope_ind:
			if msg.ver < 0x01:
				del msg
				return
			del msg.burst
			msg.burst = None
			msg.toa256 = self.TOA256_NOISE_DEFAULT
			msg.rssi = self.RSSI_NOISE_DEFAULT
			msg.ci = self.CI_NOISE_DEFAULT
			self.data_if.send_msg(msg)
			return

		# Начало нашей работы
		sps = 4
		bt = 0.3
		ts_len = min(20, len(src_msg.burst)//3)  # Для short burst

		# Вытаскиваем из src_msg.burst массив бит
		bit_array = np.array(list(map(int, src_msg.burst)), dtype=np.uint8)
		N = len(bit_array)
		print(f"[RAW] Burst bits: {bit_array.tolist()}")

		# Модуляция GMSK
		sps = 4
		bt = 0.3
		modulated_signal = gmsk_modulate(bit_array, sps=sps, bt=bt)

		# Применяем мощность передачи и усиление антенны БС
		tx_power_mw = 10**(self.tx_power_dbm / 10) * 10**(self.tx_gain_db / 10)
		modulated_signal = modulated_signal * np.sqrt(tx_power_mw)

		# Многолучевой канал (multipath)
		frequency = self.freq
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
		# Сначала затухание в тракте (path loss) и усиление антенны телефона
		signal_pl = cost_hata(modulated_signal, self.tx_height, self.rx_height, self.distance_km, frequency)
		signal_pl = signal_pl * 10**(self.rx_gain_db / 20)

		# Затем моделируем многолучёвость (multipath)
		signal_mp = multipath(signal_pl, frequency, rays, sps=sps)

		# Шумовая мощность приемника (мВт)
		noise_power_w = receiver_noise_power(self.bw, NF_dB=self.nf_db)
		noise_power_mw = noise_power_w * 1e3

		# Оценка SNR на приёмнике
		signal_power = np.mean(np.abs(signal_pl) ** 2)
		snr_linear = signal_power / noise_power_mw
		snr_db = 10 * np.log10(snr_linear)
		print(f"[SNR] Дистанция = {self.distance_km} км, SNR = {snr_db:.2f} дБ")

		# Добавление шума с рассчитанной мощностью
		noisy_signal = add_awgn_fixed_noise(signal_pl, noise_power_mw)

		# Matched filter (выделение огибающей)
		mf = matched_filter(noisy_signal, sps=sps, bt=bt)

		# Временная синхронизация по syncword (или по первым ts_len битам)
		ts_len = min(20, len(src_msg.burst)//3)
		syncword = bit_array[:ts_len]
		offset, ref_mf = time_sync(syncword, mf, sps=sps, bt=bt)
		start_idx = max(0, offset - 2*sps)
		end_idx = min(len(mf), start_idx + len(bit_array)*sps + 10*sps)
		mf_sync = mf[start_idx:end_idx]

		# Фазовая коррекция
		ref_sync = gmsk_modulate(syncword, sps=sps, bt=bt)
		rx_sync_segment = mf_sync[offset-start_idx:offset-start_idx+len(ref_sync)]
		phase_offset = np.angle(np.sum(rx_sync_segment * np.conj(ref_sync)))
		mf_corr = mf_sync * np.exp(-1j*phase_offset)

		# Кубический ресемплинг символов)
		t_original = np.arange(len(mf_corr))
		t_target = (offset - start_idx) + np.arange(0, len(bit_array)) * sps
		I_spline = CubicSpline(t_original, mf_corr.real)
		Q_spline = CubicSpline(t_original, mf_corr.imag)
		resampled = I_spline(t_target) + 1j*Q_spline(t_target)

		# Дифференциальная демодуляция
		demod = differential_demod(resampled)

		# BER только по полезной нагрузке (payload)
		payload_tx = bit_array[ts_len+1:]
		payload_rx = demod[ts_len:ts_len+len(payload_tx)]
		if len(payload_tx) > 0 and len(payload_rx) == len(payload_tx):
			ber = np.sum(payload_tx != payload_rx) / len(payload_tx)
		else:
			ber = 1.0
		print(f"[RAW] Burst Demod bits: {demod.tolist()}")
		print(f"[BER]: {ber:.5f} (payload len: {len(payload_tx)})")

		#time.sleep(100)
		# Конец нашей работы

		msg.toa256 = self.toa256
		if not self.fake_rssi_enabled:
			msg.rssi = src_trx.tx_power - src_msg.pwr - self.PATH_LOSS_DEFAULT
		else:
			msg.rssi = self.rssi
		if msg.ver >= 0x01:
			self._handle_data_msg_v1(src_msg, msg)
		if src_trx.ta != 0:
			msg.toa256 -= src_trx.ta * 256

		Transceiver.handle_data_msg(self, msg)


	# Simulation specific CTRL command handler
	def ctrl_cmd_handler(self, request):
		# Timing Advance
		# Syntax: CMD SETTA <TA>
		if self.ctrl_if.verify_cmd(request, "SETTA", 1):
			log.debug("(%s) Recv SETTA cmd" % self)

			# Store indicated value
			self.ta = int(request[1])
			return 0

		# Timing of Arrival simulation
		# Absolute form: CMD FAKE_TOA <BASE> <THRESH>
		elif self.ctrl_if.verify_cmd(request, "FAKE_TOA", 2):
			log.debug("(%s) Recv FAKE_TOA cmd" % self)

			# Parse and apply both base and threshold
			self.toa256_base = int(request[1])
			self.toa256_rand_threshold = int(request[2])
			return 0

		# Timing of Arrival simulation
		# Relative form: CMD FAKE_TOA <+-BASE_DELTA>
		elif self.ctrl_if.verify_cmd(request, "FAKE_TOA", 1):
			log.debug("(%s) Recv FAKE_TOA cmd" % self)

			# Parse and apply delta
			self.toa256_base += int(request[1])
			return 0

		# RSSI simulation
		# Absolute form: CMD FAKE_RSSI <BASE> <THRESH>
		elif self.ctrl_if.verify_cmd(request, "FAKE_RSSI", 2):
			log.debug("(%s) Recv FAKE_RSSI cmd" % self)

			# Use negative threshold to disable fake_rssi if previously enabled:
			if int(request[2]) < 0:
				self.fake_rssi_enabled = False
				return 0

			# Parse and apply both base and threshold
			self.rssi_base = int(request[1])
			self.rssi_rand_threshold = int(request[2])
			self.fake_rssi_enabled = True
			return 0

		# RSSI simulation
		# Relative form: CMD FAKE_RSSI <+-BASE_DELTA>
		elif self.ctrl_if.verify_cmd(request, "FAKE_RSSI", 1):
			log.debug("(%s) Recv FAKE_RSSI cmd" % self)

			# Parse and apply delta
			self.rssi_base += int(request[1])
			return 0

		# C/I simulation
		# Absolute form: CMD FAKE_CI <BASE> <THRESH>
		elif self.ctrl_if.verify_cmd(request, "FAKE_CI", 2):
			log.debug("(%s) Recv FAKE_CI cmd" % self)

			# Parse and apply both base and threshold
			self.ci_base = int(request[1])
			self.ci_rand_threshold = int(request[2])
			return 0

		# C/I simulation
		# Relative form: CMD FAKE_CI <+-BASE_DELTA>
		elif self.ctrl_if.verify_cmd(request, "FAKE_CI", 1):
			log.debug("(%s) Recv FAKE_CI cmd" % self)

			# Parse and apply delta
			self.ci_base += int(request[1])
			return 0

		# Path loss simulation: burst dropping
		# Syntax: CMD FAKE_DROP <AMOUNT>
		# Dropping pattern: fn % 1 == 0
		elif self.ctrl_if.verify_cmd(request, "FAKE_DROP", 1):
			log.debug("(%s) Recv FAKE_DROP cmd" % self)

			# Parse / validate amount of bursts
			num = int(request[1])
			if num < 0:
				log.error("(%s) FAKE_DROP amount shall not "
						"be negative" % self)
				return -1

			self.burst_drop_amount = num
			self.burst_drop_period = 1
			return 0

		# Path loss simulation: burst dropping
		# Syntax: CMD FAKE_DROP <AMOUNT> <FN_PERIOD>
		# Dropping pattern: fn % period == 0
		elif self.ctrl_if.verify_cmd(request, "FAKE_DROP", 2):
			log.debug("(%s) Recv FAKE_DROP cmd" % self)

			# Parse / validate amount of bursts
			num = int(request[1])
			if num < 0:
				log.error("(%s) FAKE_DROP amount shall not "
						"be negative" % self)
				return -1

			# Parse / validate period
			period = int(request[2])
			if period <= 0:
				log.error("(%s) FAKE_DROP period shall "
						"be greater than zero" % self)
				return -1

			self.burst_drop_amount = num
			self.burst_drop_period = period
			return 0

		# Artificial delay for the TRXC interface
		# Syntax: CMD FAKE_TRXC_DELAY <DELAY_MS>
		elif self.ctrl_if.verify_cmd(request, "FAKE_TRXC_DELAY", 1):
			log.debug("(%s) Recv FAKE_TRXC_DELAY cmd", self)

			self.ctrl_if.rsp_delay_ms = int(request[1])
			log.info("(%s) Artificial TRXC delay set to %d",
					self, self.ctrl_if.rsp_delay_ms)

		# Unhandled command
		return None

class Application(ApplicationBase):
	def __init__(self):
		self.app_print_copyright(APP_CR_HOLDERS)
		self.argv = self.parse_argv()

		# Set up signal handlers
		signal.signal(signal.SIGINT, self.sig_handler)

		# Configure logging
		self.app_init_logging(self.argv)

		# List of all transceivers
		self.trx_list = TRXList()

		# Init shared clock generator
		self.clck_gen = CLCKGen([], sched_rr_prio = None if self.argv.sched_rr_prio is None else self.argv.sched_rr_prio + 1)
		# This method will be called on each TDMA frame
		self.clck_gen.clck_handler = self.clck_handler

		# Power measurement emulation
		# Noise: -120 .. -105
		# BTS: -75 .. -50
		self.fake_pm = FakePM(-120, -105, -75, -50)
		self.fake_pm.trx_list = self.trx_list

		# Init TRX instance for BTS
		self.append_trx(self.argv.bts_addr, self.argv.bts_base_port, name = "BTS")

		# Init TRX instance for BB
		self.append_trx(self.argv.bb_addr, self.argv.bb_base_port, name = "MS", child_mgt = False)

		# Additional transceivers (optional)
		if self.argv.trx_list is not None:
			for trx_def in self.argv.trx_list:
				(name, addr, port, idx) = trx_def
				self.append_child_trx(addr, port, name = name, child_idx = idx)

		# Burst forwarding between transceivers
		self.burst_fwd = BurstForwarder(self.trx_list.trx_list)

		log.info("Init complete")

	def append_trx(self, remote_addr, base_port, **kwargs):
		trx = FakeTRX(self.argv.trx_bind_addr, remote_addr, base_port,
			clck_gen = self.clck_gen, pwr_meas = self.fake_pm, **kwargs)
		self.trx_list.add_trx(trx)

	def append_child_trx(self, remote_addr, base_port, **kwargs):
		child_idx = kwargs.get("child_idx", 0)
		if child_idx == 0:  # Index 0 indicates parent transceiver
			self.append_trx(remote_addr, base_port, **kwargs)
			return

		# Find 'parent' transceiver for a new child
		trx_parent = self.trx_list.find_trx(remote_addr, base_port)
		if trx_parent is None:
			raise IndexError("Couldn't find parent transceiver "
				"for '%s:%d/%d'" % (remote_addr, base_port, child_idx))

		# Allocate a new child
		trx_child = FakeTRX(self.argv.trx_bind_addr, remote_addr, base_port,
			pwr_meas = self.fake_pm, **kwargs)
		self.trx_list.add_trx(trx_child)

		# Link a new 'child' with its 'parent'
		trx_parent.child_trx_list.add_trx(trx_child)

	def run(self):
		if self.argv.sched_rr_prio is not None:
			sched_param = os.sched_param(self.argv.sched_rr_prio)
			try:
				log.info("Setting real time process scheduler to SCHED_RR, priority %u" % (self.argv.sched_rr_prio))
				os.sched_setscheduler(0, os.SCHED_RR, sched_param)
			except OSError:
				log.error("Failed to set real time process scheduler to SCHED_RR, priority %u" % (self.argv.sched_rr_prio))

		# Compose list of to be monitored sockets
		sock_list = []
		for trx in self.trx_list.trx_list:
			sock_list.append(trx.ctrl_if.sock)
			sock_list.append(trx.data_if.sock)

		# Enter main loop
		while True:
			# Wait until we get any data on any socket
			r_event, _, _ = select.select(sock_list, [], [])

			# Iterate over all transceivers
			for trx in self.trx_list.trx_list:
				# DATA interface
				if trx.data_if.sock in r_event:
					trx.recv_data_msg()

				# CTRL interface
				if trx.ctrl_if.sock in r_event:
					trx.ctrl_if.handle_rx()

	# This method will be called by the clock thread
	def clck_handler(self, fn):
		# We assume that this list is immutable at run-time
		for trx in self.trx_list.trx_list:
			trx.clck_tick(self.burst_fwd, fn)

	def shutdown(self):
		log.info("Shutting down...")

		# Stop clock generator
		self.clck_gen.stop()

	# Parses a TRX definition of the following
	# format: REMOTE_ADDR:BIND_PORT[/TRX_NUM]
	# e.g. [2001:0db8:85a3:0000:0000:8a2e:0370:7334]:5700/5
	# e.g. 127.0.0.1:5700 or 127.0.0.1:5700/1
	# e.g. foo@127.0.0.1:5700 or bar@127.0.0.1:5700/1
	@staticmethod
	def trx_def(val):
		try:
			result = re.match(r"(.+@)?(.+):([0-9]+)(/[0-9]+)?", val)
			(name, addr, port, idx) = result.groups()
		except:
			raise argparse.ArgumentTypeError("Invalid TRX definition: %s" % val)

		if idx is not None:
			idx = int(idx[1:])
		else:
			idx = 0

		# Cut '@' from TRX name
		if name is not None:
			name = name[:-1]

		return (name, addr, int(port), idx)

	def parse_argv(self):
		parser = argparse.ArgumentParser(prog = "fake_trx",
			description = "Virtual Um-interface (fake transceiver)")

		# Register common logging options
		self.app_reg_logging_options(parser)

		trx_group = parser.add_argument_group("TRX interface")
		trx_group.add_argument("-b", "--trx-bind-addr",
			dest = "trx_bind_addr", type = str, default = "0.0.0.0",
			help = "Set FakeTRX bind address (default %(default)s)")
		trx_group.add_argument("-R", "--bts-addr",
			dest = "bts_addr", type = str, default = "127.0.0.1",
			help = "Set BTS remote address (default %(default)s)")
		trx_group.add_argument("-r", "--bb-addr",
			dest = "bb_addr", type = str, default = "127.0.0.1",
			help = "Set BB remote address (default %(default)s)")
		trx_group.add_argument("-P", "--bts-base-port",
			dest = "bts_base_port", type = int, default = 5700,
			help = "Set BTS base port number (default %(default)s)")
		trx_group.add_argument("-p", "--bb-base-port",
			dest = "bb_base_port", type = int, default = 6700,
			help = "Set BB base port number (default %(default)s)")
		trx_group.add_argument("-s", "--sched-rr-prio",
			dest = "sched_rr_prio", type = int, default = None,
			help = "Set Scheduler RR Priority (default None)")

		mtrx_group = parser.add_argument_group("Additional transceivers")
		mtrx_group.add_argument("--trx",
			metavar = "REMOTE_ADDR:BASE_PORT[/TRX_NUM]",
			dest = "trx_list", type = self.trx_def, action = "append",
			help = "Add a transceiver for BTS or MS (e.g. 127.0.0.1:5703)")

		argv = parser.parse_args()

		# Make sure there is no overlap between ports
		if argv.bts_base_port == argv.bb_base_port:
			parser.error("BTS and BB base ports shall be different")

		return argv

	def sig_handler(self, signum, frame):
		log.info("Signal %d received" % signum)
		if signum == signal.SIGINT:
			self.shutdown()
			sys.exit(0)

if __name__ == '__main__':
	app = Application()
	app.run()
