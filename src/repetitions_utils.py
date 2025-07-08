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

def maxima_indices(series: np.ndarray) -> np.ndarray:
  '''Finds the indices of local maxima in a series.'''
  indices, _ = find_peaks(series)
  return indices

def minima_indices(series: np.ndarray) -> np.ndarray:
  '''Finds the indices of local minima in a series.'''
  indices, _ = find_peaks(-series)
  return indices

def count_cycles(series: np.ndarray, frac: float = 0.1) -> int:
  '''
  Counts the number of cycles (a minima and a maxima) in a series using the lowess smoothening.

  Parameters:
    series (np.ndarray): The input series.
    frac (float): The fraction of the data used for lowess smoothening.

  Returns:
    int: The number of cycles detected.
  '''
  smoothed_series = lowess(series, np.arange(len(series)), frac=frac, return_sorted=False)
  maximas = maxima_indices(smoothed_series)
  minimas = minima_indices(smoothed_series)
  return min(len(maximas), len(minimas))
