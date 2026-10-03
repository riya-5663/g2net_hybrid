"""Shared constants for the G2Net hybrid project (competition data layout)."""

T_SFT = 1800                 # seconds per SFT
N_FREQ = 360                 # frequency bins per SFT (0.2 Hz band)
N_TIME = 5760                # 1800 s * 5760 = 120 days of time slots
GPS_T0 = 1238166018          # GPS time of time slot 0 (same convention as Koda's code)

CNN_TIME_BINS = 128          # 5760 / 128 = 45 SFT slots averaged per CNN column
PHYS_TIME_CHUNK = 32         # SFT slots per physics time chunk -> 180 chunks
PHYS_TIME_BINS = N_TIME // PHYS_TIME_CHUNK

FDOT_MAX_BINS = 120.0        # max spin-down magnitude, in frequency bins over the full 120 days
