import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt

# 1. Load CSV data
data = pd.read_csv("C:/Users/ADMIN/2025_PC/datahub/XYZ_N(1).csv", header=None)
signal = data[0].values  # Use first column as the vibration signal
fs = 1000  # Sampling frequency (Hz), based on provided info

# 2. Perform FFT to analyze frequency spectrum
N = len(signal)
freq = np.fft.rfftfreq(N, 1/fs)  # Frequency axis
fft_signal = np.abs(np.fft.rfft(signal)) / N  # Normalized FFT magnitude

# 3. Design low-pass Butterworth filter
cutoff_freq = 100  # Cutoff frequency in Hz (to be adjusted based on FFT)
order = 4  # Filter order
b, a = scipy.signal.butter(order, cutoff_freq / (fs / 2), btype='low', analog=False)

# 4. Apply the filter
filtered_signal = scipy.signal.filtfilt(b, a, signal)

# 5. Compute FFT of filtered signal
fft_filtered = np.abs(np.fft.rfft(filtered_signal)) / N

# 6. Plot results
plt.figure(figsize=(12, 8))

# Time-domain signals
plt.subplot(2, 1, 1)
time = np.arange(N) / fs
plt.plot(time, signal, label='Original Signal (Normal Bearing)', alpha=0.7)
plt.plot(time, filtered_signal, label='Filtered Signal', alpha=0.7)
plt.title('Time-Domain Signal (Normal Bearing, fs=1000 Hz)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# Frequency-domain spectra
plt.subplot(2, 1, 2)
plt.plot(freq, fft_signal, label='Original Spectrum', alpha=0.7)
plt.plot(freq, fft_filtered, label='Filtered Spectrum', alpha=0.7)
plt.title('Frequency Spectrum (Normal Bearing)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.xlim(0, 200)  # Focus on lower frequencies (up to 200 Hz)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('normal_bearing_signal_analysis_1000hz.png')
plt.show()

# 7. Suggest optimal cutoff frequency based on FFT
max_freq_idx = np.argmax(fft_signal[1:]) + 1  # Find peak frequency (skip DC)
peak_freq = freq[max_freq_idx]
print(f"Peak frequency in original signal: {peak_freq:.2f} Hz")
print(f"Suggested cutoff frequency: {peak_freq * 5:.2f} Hz (5x peak frequency for normal bearing)")