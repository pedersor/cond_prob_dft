import numpy as np
from scipy import special
import matplotlib.pyplot as plt

from DFT_1d import ext_potentials


def blue_el_approx(
    grids,
    ref_point,
    lam=1,
):
  potential = -1 * lam * ext_potentials.exp_hydrogenic(grids - ref_point)
  return potential


def default_plotting_params():
  plt.rcParams["axes.titlesize"] = 24
  plt.rcParams["axes.labelsize"] = 20
  plt.rcParams["lines.linewidth"] = 3
  plt.rcParams["lines.markersize"] = 8
  plt.rcParams["xtick.labelsize"] = 16
  plt.rcParams["ytick.labelsize"] = 16
  plt.rcParams["font.size"] = 24
  plt.rcParams["legend.fontsize"] = 16
  plt.rcParams["figure.figsize"] = (9, 9)


def to_shape(a, shape):
  """ pads an array of shape `a.shape` to `shape` using zeros. """

  y_, x_ = shape
  y, x = a.shape
  y_pad = (y_ - y)
  x_pad = (x_ - x)
  return np.pad(a, (((y_pad // 2) - 2, (y_pad // 2 + y_pad % 2) + 2),
                    (x_pad // 2, x_pad // 2 + x_pad % 2)),
                mode='constant')


def dat_file_to_2d_array(file, grids):
  """ exact_pair_density.dat -> 2d numpy array.
  
  e.g. file.dat format:
  230_1 = 0.000000
  230_2 = 0.000000
  230_3 = 0.000000
  230_4 = 0.000000
  ...
  """

  array_2d = []
  with open(file) as f:
    lines = f.readlines()

    counter = 0
    array_1d = []
    for line in lines:
      array_1d.append(float(line.split()[2]))

      counter += 1
      if counter == len(grids):
        array_1d = np.asarray(array_1d)
        array_2d.append(array_1d)
        counter = 0
        array_1d = []

  array_2d = np.asarray(array_2d)
  return array_2d


def get_pair_density(raw_pair_density, density, h):
  """ Postprocessing since it was not done in C++ output
  
  * double counting at i=j
  * scaling from grid spacing h
  
  """

  pair_density = raw_pair_density - (np.diag(density) * h)
  pair_density = pair_density / (h * h)
  return pair_density


def latex_to_dir(in_string, reverse=False):
  """ converts latex str format to directory. If reverse=True, then
  converts directory str to latex format.

  Note: basic ions only.
  """

  subs = [
      ('$^+$', '_p'),
      ('$^{++}$', '_2p'),
      ('$^{3+}$', '_3p'),
  ]

  if type(in_string) != str:
    in_string = str(in_string)

  if reverse:
    subs = [(sub[1], sub[0]) for sub in subs]
    out_string = in_string.capitalize()
  else:
    out_string = in_string.lower()

  for old, new in subs:
    out_string = out_string.replace(old, new)

  return out_string
