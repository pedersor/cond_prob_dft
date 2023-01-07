import numpy as np
import matplotlib.pyplot as plt


def exp_hydrogenic(
    grids,
    amp=1.071295,
    kappa=1 / 2.385345,
    center=0,
    charge=1,
):
  """Exponential potential for 1D Hydrogenic atom.

    A 1D potential which can be used to mimic corresponding 3D
    electronic structure. Similar in form to the soft-Coulomb
    interaction. Please refer to:

    Thomas E Baker, E Miles Stoudenmire, Lucas O Wagner, Kieron Burke,
    and  Steven  R  White. One-dimensional mimicking of electronic structure:
    The case for exponentials. Physical Review B,91(23):235141, 2015.

    The cusp should lie exactly on a grid point to avoid missing any
    kinetic energy.

    Args:
      grids: numpy array of grid points for evaluating 1d potential (num_grids,)
      amp: float, fitting parameter (see `A` in the Ref. above).
      kappa: float, fitting parameter (see `\kappa` in the Ref. above).
      center: float, the center position of the potential.
      charge: float, the nuclear “charge”.

    Returns:
      vp: numpy array of the potential on a grid (num_grid,).
    """
  vp = -charge * amp * np.exp(-kappa * np.abs(grids - center))
  return vp


def blue_el_approx(
    grids,
    ref_point,
    lam=1,
):
  """Returns the blue electron approximation conditional probability potential.
  
  Args:
    grids: numpy array of grids with shape (num_grids,).
    ref_point: float, the reference position (of the blue electron).
    lam: float, the coupling-constant interaction strength, `\lambda`.
  
  Returns:
    potential: numpy array of the potential with shape (num_grids,).
  """

  potential = -1 * lam * exp_hydrogenic(grids - ref_point)
  return potential


def default_plotting_params():
  """Sets default plotting params for matplotlib. """

  plt.rcParams["axes.titlesize"] = 24
  plt.rcParams["axes.labelsize"] = 20
  plt.rcParams["lines.linewidth"] = 3
  plt.rcParams["lines.markersize"] = 8
  plt.rcParams["xtick.labelsize"] = 16
  plt.rcParams["ytick.labelsize"] = 16
  plt.rcParams["font.size"] = 24
  plt.rcParams["legend.fontsize"] = 16
  plt.rcParams["figure.figsize"] = (9, 9)


def to_shape(arr, shape):
  """Pads a 2D array of shape `a.shape` to `shape` using zeros. 
  
  Args:
    arr: numpy array of shape (y, x).
    shape: tuple of shape (y_, x_).
  
  Raises:
    ValueError: if `shape` dims. are smaller than `a.shape`.
  """

  y_, x_ = shape
  y, x = arr.shape
  y_pad = (y_ - y)
  x_pad = (x_ - x)

  if y_pad < 0 or x_pad < 0:
    raise ValueError('Desired `shape` dims. must be larger than `a.shape`.')

  pad_width_y = (y_pad // 2) - 2, (y_pad // 2 + y_pad % 2) + 2
  pad_width_x = x_pad // 2, x_pad // 2 + x_pad % 2
  padded_array = np.pad(arr, (pad_width_y, pad_width_x), mode='constant')

  return padded_array


def dat_file_to_2d_array(file, grids):
  """Read file.dat into 2d numpy array.
  
  e.g. file.dat format:
  230_1 = 0.000000
  230_2 = 0.000000
  230_3 = 0.000000
  230_4 = 0.000000
  ...

  Where e.g. 230_2 is the pair density corresponding to 
  P(x_230, y_2) = 0.000000.

  Args:
    file: str, path to file.dat with expected format.
    grids: numpy array of utilized grids with shape (num_grids,).

  Returns:
    array_2d: numpy array of shape (num_ref_points, num_grids).
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
  """Obtain pair density from raw array output.
  
  The raw array output has double counting on the diagonal and is 
  not scaled by the grid spacing.
  
  Args:
    raw_pair_density: numpy array of shape (num_ref_points, num_grids).
    density: numpy array of shape (num_grids,).
    h: float, grid spacing.
  
  Returns:
    pair_density: numpy array of shape (num_ref_points, num_grids).
  """

  pair_density = raw_pair_density - (np.diag(density) * h)
  pair_density = pair_density / (h * h)
  return pair_density


def latex_to_dir(in_string, reverse=False):
  """Converts latex str format to compatible directory name. 
  
  Note: basic ions only.

  Args:
    in_string: str, latex format or directory str.
    reverse: bool, if True, converts directory str to latex format.

  Returns:
    out_string: str, latex format or directory str.
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
