import numpy as np
import matplotlib.pyplot as plt

N = 8
sps = 4
np.random.seed(1)
bits = np.random.randint(0,2,N).astype(np.uint8)

diff = np.zeros_like(bits)
diff[0] = bits[0]
for i in range(1, N):
    diff[i] = diff[i-1] ^ bits[i]

nrz = 2*diff - 1
upsampled = np.zeros(N*sps)
upsampled[::sps] = nrz
signal = np.exp(1j * np.pi/2 * np.cumsum(upsampled))
rx = signal

sampled = rx[sps//2::sps][:N]
phase = np.angle(sampled)

phase_diff = np.angle(sampled[1:] * np.conj(sampled[:-1]))

# Попробовать разные пороги
for thresh in [0, np.pi/4, np.pi/2]:
    demod = (phase_diff > thresh).astype(np.uint8)
    demod_dec = np.zeros_like(demod)
    demod_dec[0] = demod[0]
    for i in range(1, len(demod)):
        demod_dec[i] = demod_dec[i-1] ^ demod[i]
    ber = np.sum(diff[1:] != demod_dec) / (N-1)



demod = (phase_diff > 0).astype(np.uint8)
print("phase_diff:", phase_diff)
print("demod:", demod)
print("diff[1:]:", diff[1:])
print("demod == diff[1:]:", demod == diff[1:])
print("demod != diff[1:]:", demod != diff[1:])
ber = np.sum(diff[1:] != demod) / (N-1)
print("BER:", ber)

demod = (phase_diff > 0).astype(np.uint8)
demod = 1 - demod
print("demod:", demod)
print("diff[1:]:", diff[1:])
print("demod == diff[1:]:", demod == diff[1:])
ber = np.sum(diff[1:] != demod) / (N-1)
print("BER:", ber)






