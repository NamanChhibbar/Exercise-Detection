import numpy as np
from scipy.signal import find_peaks
from statsmodels.nonparametric.smoothers_lowess import lowess


def max_variance_series(array_series: np.ndarray) -> np.ndarray:
  '''
  Returns the series with the maximum variance from an array of series of shape (length_series, num_series).

  Parameters:
    array_series (np.ndarray): Array of input series.

  Returns:
    np.ndarray: The series with the maximum variance.
  '''
  index = np.argmax(np.var(array_series, axis=0))
  return array_series[:, index]

def extrema_indices(series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  '''
  Finds the indices of local extremas (maximas and minimas) in a series.

  Parameters:
    series (np.ndarray): Input series.

  Returns:
    (np.ndarray, np.ndarray): Indices of maximas and minimas repectively.  
  '''
  maxima_indices, _ = find_peaks(series)
  minima_indices, _ = find_peaks(-series)
  return maxima_indices, minima_indices

def count_cycles(series: np.ndarray, frac: float = 0.1) -> int:
  '''
  Counts the number of cycles (a minima and a maxima) in a series using the lowess smoothing.

  Parameters:
    series (np.ndarray): Input series.
    frac (float): The fraction of the data used for lowess smoothing.

  Returns:
    int: The number of cycles detected.
  '''
  smooth_series = lowess(series, np.arange(len(series)), frac=frac, return_sorted=False)
  maximas, minimas = extrema_indices(smooth_series)
  return min(len(maximas), len(minimas))
