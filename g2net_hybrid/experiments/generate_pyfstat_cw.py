import os
import pyfstat


# --------------------------------------------------
# Simulation parameters
# --------------------------------------------------

OUTDIR = "../data/pyfstat_cw"
LABEL = "testcw"

# GPS start time
TSTART = 1_000_000_000

# Generate 1 day of data
DURATION = 24 * 3600

# SFT duration
TSFT = 1800

# Detector noise amplitude spectral density
SQRT_SX = 1e-23


# --------------------------------------------------
# CW signal parameters
# --------------------------------------------------

signal_parameters = {
    "F0": 100.0,       # frequency [Hz]
    "F1": -1e-10,      # first frequency derivative [Hz/s]
    "F2": 0.0,

    # Sky position
    "Alpha": 1.0,
    "Delta": 0.5,

    # Signal amplitude / orientation
    "h0": 5e-23,
    "cosi": 0.0,

    # Reference time
    "tref": TSTART,
}


# --------------------------------------------------
# Data parameters
# --------------------------------------------------

data_parameters = {
    "sqrtSX": SQRT_SX,
    "tstart": TSTART,
    "duration": DURATION,
    "Tsft": TSFT,
    "detectors": "H1",
}


# --------------------------------------------------
# Generate data
# --------------------------------------------------

os.makedirs(OUTDIR, exist_ok=True)

writer = pyfstat.Writer(
    label=LABEL,
    outdir=OUTDIR,
    **data_parameters,
    **signal_parameters,
)

writer.make_data()

print("CW generation complete.")
print()
print("SFT file:")
print(writer.sftfilepath)