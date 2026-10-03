import pyfstat
import numpy as np


SFT_FILE = "../data/pyfstat_cw/*test_cw*sft"


freqs, times, data = pyfstat.utils.sft.get_sft_as_arrays(
    SFT_FILE
)

print("Frequency shape:")
print(freqs.shape)

print("\nTimes:")
for detector, detector_times in times.items():
    print(detector, detector_times.shape)

print("\nData:")
for detector, detector_data in data.items():
    print(detector, detector_data.shape)

print("\nFrequency range:")
print(freqs[0], "to", freqs[-1])

print("\nFirst few frequencies:")
print(freqs[:10])