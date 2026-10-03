import numpy as np
import matplotlib.pyplot as plt

from physics.power_score import trajectory_power_score


# -------------------------------------------------
# Configuration
# -------------------------------------------------

rng = np.random.default_rng(42)

n_frequency = 360
n_time = 500

f_min = 99.9
df = 1 / 1800

frequency = (
    f_min
    + np.arange(n_frequency) * df
)

time = np.arange(n_time)


# -------------------------------------------------
# Create complex Gaussian noise
# -------------------------------------------------

noise_real = rng.normal(
    0,
    1,
    size=(n_frequency, n_time),
)

noise_imag = rng.normal(
    0,
    1,
    size=(n_frequency, n_time),
)

sfts = noise_real + 1j * noise_imag


# -------------------------------------------------
# Inject a simple CW signal
# -------------------------------------------------

f0 = 99.97
fdot = 2e-6

true_frequency = (
    f0
    + fdot * time
)

# Find the nearest frequency bin for injection.
true_bins = np.rint(
    (true_frequency - f_min) / df
).astype(int)

valid = (
    (true_bins >= 0)
    & (true_bins < n_frequency)
)

signal_amplitude = 5.0

for t in range(n_time):
    if valid[t]:
        sfts[true_bins[t], t] += signal_amplitude


# -------------------------------------------------
# Score the correct trajectory
# -------------------------------------------------

true_result = trajectory_power_score(
    sfts,
    frequency,
    true_frequency,
)


# -------------------------------------------------
# Score an incorrect trajectory
# -------------------------------------------------

wrong_frequency = np.full(
    n_time,
    f0,
)

wrong_result = trajectory_power_score(
    sfts,
    frequency,
    wrong_frequency,
)

true_score = true_result["score"]
wrong_score = wrong_result["score"]

print("True:", true_result)
print("Wrong:", wrong_result)


print("True trajectory score :", true_score)
print("Wrong trajectory score:", wrong_score)
print("Score ratio           :", true_score / wrong_score)


# -------------------------------------------------
# Plot spectrogram
# -------------------------------------------------

power = np.abs(sfts) ** 2

plt.figure(figsize=(12, 5))

plt.imshow(
    np.log10(power + 1e-8),
    aspect="auto",
    origin="lower",
    extent=[
        0,
        n_time,
        frequency[0],
        frequency[-1],
    ],
)

plt.plot(
    time,
    true_frequency,
    linewidth=2,
    label="True trajectory",
)

plt.plot(
    time,
    wrong_frequency,
    linewidth=2,
    label="Wrong trajectory",
)

plt.xlabel("Time index")
plt.ylabel("Frequency (Hz)")
plt.title("Synthetic CW signal")
plt.legend()

plt.tight_layout()
plt.show()