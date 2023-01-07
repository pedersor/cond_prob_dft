from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

from DFT_1d import ks_inversion
from DFT_1d import ext_potentials
from DFT_1d import functionals
from DFT_1d import ks_dft

from cond_prob_dft_1d import utils

if __name__ == '__main__':

  # set plot style
  utils.default_plotting_params()

  # setup grids with uniform spacing h
  h = 0.08
  grids = np.arange(-256, 257) * h
  num_grids = len(grids)

  ions = {
      'densities': np.load('dataset/densities.npy'),
      # mag_density = n_up - n_down
      'mag_densities': np.load('dataset/magnetization_densities.npy'),
      'external_potentials': np.load('dataset/external_potentials.npy'),
      'num_electrons': np.load('dataset/num_electrons.npy')
  }

  density = ions['densities'][0]
  mag_density = ions['mag_densities'][0]
  up_density = (density + mag_density) / 2
  dn_density = (density - mag_density) / 2

  external_potential = lambda _: ions['external_potentials'][0]
  num_electrons = ions['num_electrons'][0]

  # check normalization
  num_up_electrons = int(np.rint(np.sum(up_density) * h))
  num_dn_electrons = int(np.rint(np.sum(dn_density) * h))
  np.testing.assert_equal(num_up_electrons, 2)
  np.testing.assert_equal(num_dn_electrons, 1)
  np.testing.assert_equal(num_up_electrons + num_dn_electrons, num_electrons)

  # exact pair density from DMRG calculation
  raw_up_pair_density = utils.dat_file_to_2d_array('exact_up_pair_density.dat',
                                                   grids)
  raw_up_pair_density = utils.to_shape(raw_up_pair_density,
                                       (num_grids, num_grids))
  raw_dn_pair_density = utils.dat_file_to_2d_array('exact_dn_pair_density.dat',
                                                   grids)
  raw_dn_pair_density = utils.to_shape(raw_dn_pair_density,
                                       (num_grids, num_grids))
  up_pair_density = utils.get_pair_density(raw_up_pair_density, up_density, h)
  dn_pair_density = utils.get_pair_density(raw_dn_pair_density, dn_density, h)

  # Note: numpy will pass divide by zero warnings despite filtering such cases.
  up_cond_prob_density = np.where(
      np.expand_dims(up_density, axis=1) > 1e-4,
      up_pair_density / np.expand_dims(up_density, axis=1), 0)
  dn_cond_prob_density = np.where(
      np.expand_dims(dn_density, axis=1) > 1e-4,
      dn_pair_density / np.expand_dims(dn_density, axis=1), 0)

  # reference points to evaluate on
  ref_points = [0.00, 0.80]
  combine_pdf = False
  blue_el_results = True

  pdf_base_name = 'li_1d_example'
  latex = {'up': r'\uparrow', 'dn': r'\downarrow'}

  if combine_pdf:
    pdf = PdfPages(pdf_base_name + 's.pdf')
  for sigma, (cond_prob_density, sigma_density, num_sigma_electrons) in {
      'up': (up_cond_prob_density, up_density, 2),
      'dn': (dn_cond_prob_density, dn_density, 1)
  }.items():
    for i, ref_point in enumerate(ref_points):

      # CP density should integrate to num_electrons - 1
      curr_cond_prob_density = cond_prob_density[grids == ref_point][0]
      cp_num_electrons = np.sum(curr_cond_prob_density) * h
      np.testing.assert_allclose(
          cp_num_electrons,
          num_electrons - 1,
          atol=1e-3,
          rtol=1e-3,
      )
      cp_num_electrons = int(np.rint(cp_num_electrons))

      # spin CP-DFT
      if sigma == 'up':
        occ_per_state = 2
      elif sigma == 'dn':
        occ_per_state = 1

      cp_v_blue = lambda _: external_potential(
          grids) - ext_potentials.exp_hydrogenic(grids, center=ref_point)
      cp_ks_blue = ks_dft.Spinless_KS_Solver(
          grids,
          v_ext=cp_v_blue,
          xc=functionals.ExponentialLDAFunctional,
          num_electrons=cp_num_electrons,
          occupation_per_state=occ_per_state,
      )
      cp_ks_blue.solve_self_consistent_density()

      cp_ks_inv = ks_inversion.two_iter_KS_inversion(
          grids,
          external_potential,
          curr_cond_prob_density,
          cp_num_electrons,
          occupation_per_state=occ_per_state,
          t_tol=0.0001)

      cp_v_s = cp_ks_inv.get_v_s()
      cp_density_from_inv = cp_ks_inv.f_density

      ks_inv_sigma = ks_inversion.two_iter_KS_inversion(
          grids,
          external_potential,
          sigma_density,
          num_sigma_electrons,
          total_density=density,
          occupation_per_state=1,
          t_tol=0.0001,
      )

      v_s_sigma = ks_inv_sigma.get_v_s()

      fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
      if ref_point == 0.0:
        ref_point = int(0)
      ax1.text(
          0.75,
          0.75,
          f'$ x = ({ref_point} \, , {latex[sigma]})$',
          horizontalalignment='center',
          verticalalignment='center',
          transform=ax1.transAxes,
      )
      ax1.text(
          0.15,
          0.85,
          '1D Li',
          horizontalalignment='center',
          verticalalignment='center',
          transform=ax1.transAxes,
      )
      ax1.axvline(ref_point, color='black', linestyle='--')
      if blue_el_results:
        ax1.plot(
            grids,
            cp_ks_blue.density,
            label=r'$\tilde n_x^{\mathrm{BEA, LDA}}(y^{\prime})$',
            color='tab:blue',
        )
      ax1.plot(
          grids,
          curr_cond_prob_density,
          label=r'$\tilde n_x(y^{\prime})$',
          color='tab:red',
      )
      if not blue_el_results:
        ax1.plot(
            grids,
            sigma_density,
            label=r'$n_{\sigma}(y^{\prime})$',
            color='black',
        )
      ax1.plot(grids, density, label=r'$n(y^{\prime})$', color='gray')

      ax1.grid(alpha=0.4)
      ax1.set_ylim(bottom=0)
      if i == 0 and sigma == 'up':
        ax1.legend(loc='lower left')

      if blue_el_results:
        ax2.plot(
            grids,
            cp_ks_blue.v_s(grids),
            label=r'$\tilde v^{\mathrm{BEA, LDA}}_{{\rm S}, x}(y^{\prime})$',
            color='tab:blue',
        )
      ax2.plot(
          grids,
          cp_v_s,
          label=r'$\tilde v_{{\rm S}, x}(y^{\prime})$',
          color='tab:red',
      )
      if not blue_el_results:
        ax2.plot(
            grids,
            v_s_sigma,
            label=r'$v_{{\rm S}, \sigma}(y^{\prime})$',
            color='black',
        )
      ax2.axvline(ref_point, color='black', linestyle='--')
      ax2.set_ylim(bottom=-2)
      ax2.set_xlim(-5, 5)
      ax2.set_xlabel(r'$y^{\prime}$')
      ax2.grid(alpha=0.4)
      if i == 0 and sigma == 'up':
        ax2.legend(loc='lower left')

      plt.tight_layout()

      if combine_pdf:
        pdf.savefig(bbox_inches='tight')
      else:
        plt.savefig(pdf_base_name + f'_{sigma}_{i}.pdf', bbox_inches='tight')

    plt.close()

  if combine_pdf:
    pdf.close()
