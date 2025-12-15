import numpy as np


def beamform(X, delay, spectral_dim, sensor_dim):
    """
    Simple delay-and-sum beamformer.

    Parameters
    ----------
    X: DataArray
        Spectral data to be beamformed with dimensions including `spectral_dim` and
        `sensor_dim`.
    delay: DataArray
        Delay table for each sensor and grid point. Must have one dimension
        corresponding to `sensor_dim`.
    spectral_dim: str
        The spectral dimension in the input data.
    sensor_dim: str
        The sensor dimension in the input data.

    Returns
    -------
    DataArray
        Complex beamformed output.
    """
    f = X[spectral_dim].to_dataarray()
    v = np.exp(-2j * np.pi * f * delay)
    return (np.conj(v) * X).sum(sensor_dim)


def power(X, dim):
    """
    Compute power of the signal along a given dimension.

    Parameters
    ----------
    X: DataArray
        Input data.
    dim: str
        Dimension along which to compute power.

    Returns
    -------
    DataArray
        Power of the input data.
    """
    return np.real(np.conj(X) * X).sum(dim)
