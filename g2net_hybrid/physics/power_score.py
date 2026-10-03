import numpy as np


def compute_power(sfts: np.ndarray) -> np.ndarray:
    """Compute power from complex SFT coefficients."""
    return np.abs(sfts) ** 2


def trajectory_power_score(
    sfts: np.ndarray,
    frequency_bins: np.ndarray,
    signal_frequency: np.ndarray,
) -> dict:
    """
    Compute mean power along a proposed frequency trajectory.

    Parameters
    ----------
    sfts:
        Complex SFT array of shape (frequency, time).

    frequency_bins:
        Frequency values of shape (frequency,).

    signal_frequency:
        Candidate frequency at each time, shape (time,).
    """

    power = compute_power(sfts)

    df = frequency_bins[1] - frequency_bins[0]

    # Convert physical frequency to fractional bin position.
    fractional_bins = (
        signal_frequency - frequency_bins[0]
    ) / df

    # Linear interpolation between neighboring frequency bins.
    lower = np.floor(fractional_bins).astype(int)
    upper = lower + 1

    weights = fractional_bins - lower

    time_indices = np.arange(len(signal_frequency))

    valid = (
        (lower >= 0)
        & (upper < len(frequency_bins))
    )

    sampled_power = (
        (1.0 - weights[valid])
        * power[lower[valid], time_indices[valid]]
        +
        weights[valid]
        * power[upper[valid], time_indices[valid]]
    )

    if sampled_power.size == 0:
        return {
        "score": 0.0,
        "n_samples": 0,
    }

    return {
        "score": float(sampled_power.mean()),
        "n_samples": int(sampled_power.size),
    }