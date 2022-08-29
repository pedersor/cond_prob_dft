from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

from DFT_1d import ks_inversion
from DFT_1d import ext_potentials
from DFT_1d.non_interacting_solver import EigenSolver
from DFT_1d.utils import IntegralTool

from blue_electron_theory import utils

if __name__ == '__main__':

  # set plot style
  utils.default_plotting_params()

  # setup grids with uniform spacing h
  h = 0.08
  grids = np.arange(-256, 257) * h

  ions = {
      'latex_symbols': np.load('dataset/latex_symbols.npy'),
      'densities': np.load('dataset/densities.npy'),
      'external_potentials': np.load('dataset/external_potentials.npy')
  }

  density = ions['densities'][ions['latex_symbols'] == 'He'][0]
  external_potential = lambda _: ions['external_potentials'][ions[
      'latex_symbols'] == 'He'][0]

  # exact pair density from DMRG calculation
  raw_pair_density = utils.dat_file_to_2d_array('exact_pair_density.dat', grids)
  pair_density = utils.get_pair_density(raw_pair_density, density, h)

  # Note: numpy will pass divide by zero warnings despite filtering such cases.
  cond_prob_density = np.where(
      np.expand_dims(density, axis=1) > 1e-4,
      pair_density / np.expand_dims(density, axis=1), 0)

  # reference points to evaluate on
  ref_points = [0.00, 0.80]
  combine_pdf = False
  blue_el_results = True

  pdf_base_name = 'he_1d_example'

  if combine_pdf:
    pdf = PdfPages(pdf_base_name + 's.pdf')
  for i, ref_point in enumerate(ref_points):

    # CP density should integrate to num_electrons - 1
    curr_cond_prob_density = cond_prob_density[grids == ref_point][0]
    num_electrons = int(np.rint(np.sum(density) * h))
    cp_num_electrons = int(np.rint(np.sum(curr_cond_prob_density) * h))
    np.testing.assert_equal(cp_num_electrons, num_electrons - 1)

    # blue electron approximation potential
    cp_v_s_blue = external_potential(grids) - ext_potentials.exp_hydrogenic(
        grids, center=ref_point)
    solver = EigenSolver(
        grids,
        potential_fn=lambda _: cp_v_s_blue,
        num_electrons=1,
    )
    solver.solve_ground_state()
    blue_cp_density = solver.density

    v_h = IntegralTool(grids).hartree_matrix.dot(curr_cond_prob_density)
    v_xc_guess = -v_h - ext_potentials.exp_hydrogenic(grids, center=ref_point)
    cp_ks_inv = ks_inversion.two_iter_KS_inversion(
        grids,
        external_potential,
        curr_cond_prob_density,
        cp_num_electrons,
        init_v_xc=v_xc_guess,
        t_tol=0.0001,
    )

    cp_v_s = cp_ks_inv._get_v_eff()
    cp_density_from_inv = cp_ks_inv.f_density

    ks_inv = ks_inversion.two_iter_KS_inversion(
        grids,
        external_potential,
        density,
        num_electrons,
        t_tol=0.0001,
    )
    v_s = ks_inv._get_v_eff()

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    if ref_point == 0.0:
      ref_point = int(0)
    ax1.axvline(ref_point, color='black', linestyle='--')
    ax1.text(
        0.75,
        0.75,
        f'$y = {ref_point}$',
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax1.transAxes,
    )
    ax1.text(
        0.15,
        0.85,
        '1D He',
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax1.transAxes,
    )
    if blue_el_results:
      ax1.plot(
          grids,
          blue_cp_density,
          label=r'$\tilde n^{\rm{BEA}}_y(y^{\prime})$',
          color='tab:blue',
      )
    ax1.plot(
        grids,
        curr_cond_prob_density,
        label=r'$\tilde n_y(y^{\prime})$',
        color='tab:orange',
    )
    ax1.plot(grids, density, label=r'$n(y^{\prime})$', color='gray')

    ax1.set_xlim(-4, 4)
    ax1.set_ylim(bottom=0)
    ax1.grid(alpha=0.4)
    if i == 0:
      ax1.legend(loc='lower left')

    if blue_el_results:
      ax2.plot(
          grids,
          cp_v_s_blue,
          label=r'$\tilde v_{{\rm S}, y}^{\mathrm{BEA}}(y^{\prime})$',
          color='tab:blue',
      )
    ax2.plot(
        grids,
        cp_v_s,
        label=r'$\tilde v_{{\rm S}, y}(y^{\prime})$',
        color='tab:orange',
    )
    ax2.plot(grids, v_s, label=r'$v_{\rm S}(y^{\prime})$', color='gray')
    ax2.axvline(ref_point, color='black', linestyle='--')

    ax2.set_xlim(-4, 4)
    ax2.set_ylim(top=0, bottom=-1.5)
    ax2.set_xlabel(r'$y^{\prime}$')
    ax2.grid(alpha=0.4)
    if i == 0:
      ax2.legend(loc='lower left')

    plt.tight_layout()

    if combine_pdf:
      pdf.savefig(bbox_inches='tight')
    else:
      plt.savefig(pdf_base_name + f'_{i}.pdf', bbox_inches='tight')

    plt.close()

  if combine_pdf:
    pdf.close()
