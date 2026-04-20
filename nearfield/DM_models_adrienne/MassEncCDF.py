import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend, saves to files without opening windows
import matplotlib.pyplot as plt
import warnings

from scipy.interpolate import interp1d

from colossus.halo.concentration import modelDiemer19
from colossus.cosmology import cosmology
from colossus.utils import utilities
from colossus import defaults
defaults.persistence = ''  # tell colossus to not cache files
from colossus.lss.mass_function import massFunction
from colossus.lss import peaks
from colossus.halo import mass_so as halo_mass
from colossus.halo import profile_nfw

import pickle


class ModifiedPowerSpectrum(object):
    """
    This class makes modified P(k) models, tabulates them, and then uses colossus to compute halo mass functions and concentration mass relations
    """
    def __init__(self, k_pivot=0.05):
        """
        Initialize cosmology and a LambdaCDM power spectrum
        """
        cosmo = cosmology.setCosmology('planck18', persistence='r')
        self.kref = 10**np.linspace(-5.0, 3.7, 3000)
        self.k_pivot = k_pivot
        self.Pkref = cosmo.matterPowerSpectrum(self.kref)

    def pk_from_npy(self, k_file, pk_file, pk_ref_file=None):
        """
        Load P(k) from .npy files. If pk_ref_file is provided, computes the transfer
        function T(k) = sqrt(P(k) / P_ref(k)) and applies T(k)^2 * self.Pkref.
        Otherwise, interpolates the provided P(k) directly onto self.kref.

        k_file: path to .npy file with k values (h/Mpc)
        pk_file: path to .npy file with P(k) values
        pk_ref_file: optional path to .npy file with reference LCDM P(k)
        """
        k_data = np.load(k_file)
        pk_data = np.load(pk_file)
        if np.any(self.kref > k_data[-1]):
            warnings.warn(
                f"kref has values up to {self.kref.max():.4e}, "
                f"but npy data only extends to {k_data[-1]:.4e}. "
                "Extrapolating beyond data range."
            )
        if pk_ref_file is not None:
            pk_ref_data = np.load(pk_ref_file)
            T_data = np.sqrt(pk_data / pk_ref_data)
            T_interp = interp1d(np.log(k_data), np.log(T_data), kind='cubic',
                                bounds_error=False, fill_value="extrapolate")
            T_at_kref = np.exp(T_interp(np.log(self.kref)))
            T_at_kref[self.kref < k_data[0]] = 1.0
            return T_at_kref**2 * self.Pkref
        else:
            pk_interp = interp1d(np.log(k_data), np.log(pk_data), kind='cubic',
                                 bounds_error=False, fill_value="extrapolate")
            return np.exp(pk_interp(np.log(self.kref)))

    def pk_general(self, transfer_file):
        """
        Modify the power spectrum by a transfer function T(k) read from a file.
        transfer_file: path to a two-column file (k [h/Mpc], T)
        Returns T(k)^2 * Pkref, with T interpolated onto self.kref.
        T(k) = 1 for k below the file range; warns if kref exceeds the file range.
        """
        k_file, T_file = np.loadtxt(transfer_file, unpack=True, skiprows=1, delimiter=',')
        if np.any(self.kref > k_file[-1]):
            warnings.warn(
                f"kref has values up to {self.kref.max():.4e}, "
                f"but transfer file only extends to {k_file[-1]:.4e}. "
                "Extrapolating beyond file range."
            )
        T_interp = interp1d(np.log(k_file), np.log(T_file), kind='cubic',
                            bounds_error=False, fill_value="extrapolate")
        T_at_kref = np.exp(T_interp(np.log(self.kref)))
        T_at_kref[self.kref < k_file[0]] = 1.0
        return T_at_kref**2 * self.Pkref

    def pk_tilted(self, n_s, ar, min_k=None):
        """
        Add a tilt to the power spectrum; if min_k is specified then the tilt only kicks in for k > min_k

        ns: a constant tilt (k/k0)^ns
        ar: a running term (k/k0)^a*log(k/k0)
        """
        running = n_s + ar * np.log(self.kref / self.k_pivot)
        if min_k is None:
            tilt = (self.kref / self.k_pivot) ** running
        else:
            tilt = (self.kref / min_k) ** running
            tilt[np.where(self.kref < min_k)[0]] = 1.0
        return tilt * self.Pkref

    def pk_bump(self, amp, k0, k_sigma_dex):
        """
        Adds a bump parameterized as a Gaussian with amplitude, position, and width in dex
        """
        tilt = 1 + amp * np.exp(-0.5 * (np.log10(self.kref) - np.log10(k0))**2 / k_sigma_dex**2)
        return tilt * self.Pkref

    def pk_cutoff(self, k0):
        """
        A WDM-like cutoff in the matter power spectrum
        """
        nu = 1.12
        tilt = (1 + (self.kref / k0)**nu) ** (-5 / nu)
        return tilt * self.Pkref

    # to get LCDM, param1=0, param2=0, param3=None, pk_model='tilt'
    # to get tilt, param1=ns, param2=ar, param3=min_k, pk_model='tilt'
    # to get bump, param1=amp, param2=k0, param3=width, pk_model='bump'
    # to get wdm, param1 = k0, param2=None, param3=None, pk_model='wdm'
    # for a transfer function CSV, param1 = file, pk_model='file'
    # for npy files, param1 = k_file, param2 = pk_file, param3 = pk_ref_file (optional), pk_model='npy'
    def evaluate_concentrationsMine(self, m, z, param1, param2, param3=None, pk_model='tilt'):
        """
        Compute concentrations for the model
        """
        cosmo = cosmology.setCosmology('planck18', persistence='r')
        custom_pk = np.empty((len(self.kref), 2))
        custom_pk[:, 0] = np.log10(self.kref)
        if pk_model == 'tilt':
            ns = param1
            ar = param2
            min_k = param3
            custom_pk[:, 1] = np.log10(self.pk_tilted(ns, ar, min_k))
        elif pk_model == 'bump':
            amp = param1
            k0 = param2
            width = param3
            custom_pk[:, 1] = np.log10(self.pk_bump(amp, k0, width))
        elif pk_model == 'wdm':
            k0 = param1
            custom_pk[:, 1] = np.log10(self.pk_cutoff(k0))
        elif pk_model == 'file':
            tkfile = param1
            custom_pk[:, 1] = np.log10(self.pk_general(tkfile))
        elif pk_model == 'npy':
            k_file = param1
            pk_file = param2
            pk_ref_file = param3
            custom_pk[:, 1] = np.log10(self.pk_from_npy(k_file, pk_file, pk_ref_file))
        file_path = './custom.txt'
        np.savetxt(file_path, X=custom_pk)
        ps_args = dict(model='custom', path=file_path, persistence='r')
        c, _ = modelDiemer19(m, z, ps_args=ps_args)
        return c

    # to get LCDM, param1=0, param2=0, param3=None, pk_model='tilt'
    # to get tilt, param1=ns, param2=ar, param3=min_k, pk_model='tilt'
    # to get bump, param1=amp, param2=k0, param3=width, pk_model='bump'
    # to get wdm, param1 = k0, param2=None, param3=None, pk_model='wdm'
    # for a transfer function CSV, param1 = file, pk_model='file'
    # for npy files, param1 = k_file, param2 = pk_file, param3 = pk_ref_file (optional), pk_model='npy'
    def evaluate_mass_function(self, m, z, param1, param2, param3=None, pk_model='tilt'):
        """
        Evaluate the halo mass function for this model
        """
        cosmo = cosmology.setCosmology('planck18', persistence='r')
        custom_pk = np.empty((len(self.kref), 2))
        custom_pk[:, 0] = np.log10(self.kref)
        if pk_model == 'tilt':
            ns = param1
            ar = param2
            min_k = param3
            custom_pk[:, 1] = np.log10(self.pk_tilted(ns, ar, min_k))
        elif pk_model == 'bump':
            amp = param1
            k0 = param2
            width = param3
            custom_pk[:, 1] = np.log10(self.pk_bump(amp, k0, width))
        elif pk_model == 'wdm':
            k0 = param1
            custom_pk[:, 1] = np.log10(self.pk_cutoff(k0))
        elif pk_model == 'file':
            tkfile = param1
            custom_pk[:, 1] = np.log10(self.pk_general(tkfile))
        elif pk_model == 'npy':
            k_file = param1
            pk_file = param2
            pk_ref_file = param3
            custom_pk[:, 1] = np.log10(self.pk_from_npy(k_file, pk_file, pk_ref_file))
        file_path = './custom.txt'
        np.savetxt(file_path, X=custom_pk)
        ps_args = dict(model='custom', path=file_path, persistence='r')
        dndlogm = massFunction(m, z, q_in='M', q_out='dndlnM', ps_args=ps_args)
        return dndlogm


def sample_from_mass_function(num_draw, m, dndlnm_model):
    """
    Draw halo masses from a tabulated mass function dn/dlnM via inverse-CDF
    sampling.  The CDF is built with the trapezoidal rule so that every
    tabulated value contributes.  The M grid is uniform in log10(M), and
    the constant d(log10 M) factor cancels in the normalization, so the
    CDF shape is correct.
    """
    log10m = np.log10(m)
    # Trapezoidal bin weights on the uniform log10(M) grid; the constant
    # d(log10 M) factor cancels in the normalization below.
    bin_weights = 0.5 * (dndlnm_model[:-1] + dndlnm_model[1:])
    cdf = np.concatenate(([0.0], np.cumsum(bin_weights)))
    cdf /= cdf[-1]
    cdf_inverse = interp1d(cdf, log10m)
    u = np.random.uniform(0.0, 1.0, num_draw)
    return 10**cdf_inverse(u)


def sample_mc_relation(m_eval, m, mc_model):
    mc_relation_interp = interp1d(np.log10(m), mc_model)
    return mc_relation_interp(np.log10(m_eval))

# This uses Eq. (1) of Nadler et al. 2020, ApJ 893 48 (arXiv:1912.03303)
def rhalf_from_mass_kpc(m):
    cosmo_local = cosmology.getCurrent()
    h = cosmo_local.H0 / 100.
    rvir = halo_mass.M_to_R(m, 0, 'vir')  # in kpc/h
    return 0.040 * ((rvir / h) / 10.0)  


def menclosed_from_m(m, m_model, cm_model):
    z = 0
    c = sample_mc_relation(m, m_model, cm_model)  # find the concentration for a halo with mass m
    c = np.exp(np.log(c) + np.random.normal(0, 0.15))  # add scatter to concentration
    p_nfw = profile_nfw.NFWProfile(M=m, c=c, z=z, mdef='vir')
    rs = p_nfw.par['rs']
    rhos = p_nfw.par['rhos']
    rhalf = np.exp(np.log(rhalf_from_mass_kpc(m)) + np.random.normal(0, 0.6))  # in kpc, with scatter
    # integrate profile to rhalf
    X = rhalf / rs
    fc = np.log(1 + X) - X / (1 + X)
    m_enclosed = 4 * np.pi * rhos * rs**3 * fc
    return rhalf, m_enclosed


def generate_samples(N, m_tabulated, dndm_tabulated, cm_tabulated, m_min=10**8, r_half_min=0.01,
                     m_half_mode=None):
    """
    Generate a mock halo population by sampling from the mass function, assigning
    NFW concentrations, computing half-light radii, and integrating the NFW profile to get enclosed masses.

    Returns: rhalf, m_enclosed, m_samples, c_samples (all filtered by m_min and r_half_min)
    """
    z = 0
    cosmo = cosmology.setCosmology('planck18', persistence='r')
    h = cosmo.H0 / 100.

    # Draw halo masses from mass function and cut below m_min
    m_samples = sample_from_mass_function(N, m_tabulated, dndm_tabulated)
    inds = np.where(m_samples >= m_min)[0]
    m_samples = m_samples[inds]

    # Interpolate concentration-mass relation
    c = sample_mc_relation(m_samples, m_tabulated, cm_tabulated)

    # Add lognormal scatter to concentrations
    c_samples = np.exp(np.log(c) + np.random.normal(0, 0.15, size=len(c)))

    # Compute half-light radii with lognormal scatter
    rhalf = np.exp(np.log(rhalf_from_mass_kpc(m_samples)) + np.random.normal(0, 0.6, size=len(m_samples)))

    # Filter by minimum half-light radius
    inds = np.where(rhalf > r_half_min)[0]
    m_samples = m_samples[inds]
    c_samples = c_samples[inds]
    rhalf = rhalf[inds]

    # Compute NFW rhos and rs for all halos at once (vectorized)
    rhos, rs = profile_nfw.NFWProfile.nativeParameters(m_samples, c_samples, z, 'vir')
    rs = rs / h        # convert from kpc/h to kpc
    rhos = rhos * h**2  # convert from Msun/h / (kpc/h)^3 to Msun/kpc^3
    X = rhalf / rs
    fc = np.log(1 + X) - X / (1 + X)
    m_enclosed = 4 * np.pi * rhos * rs**3 * fc

    return rhalf, m_enclosed, m_samples, c_samples


def cumulative_mass_function(m_enclosed, m_min=5, m_max=10, num_bins=10, normed=True):
    """
    Compute the complementary CDF of enclosed masses.

    m_enclosed: array of enclosed mass values (from generate_samples)
    m_min, m_max: log10 of the bin edges
    num_bins: number of bins
    normed: if True, return 1 - CDF (fraction with mass > M);
            if False, return un-normalized counts with mass > M

    Returns: (bin_edges, complementary_cumulative)
    """
    masses = np.logspace(m_min, m_max, num_bins + 1)
    h, b = np.histogram(m_enclosed, bins=masses)
    m_cumulative = []
    for i in range(0, len(h)):
        m_cumulative.append(np.sum(h[0:i]))
    m_cumulative = np.array(m_cumulative)
    if normed:
        m_cumulative = 1 - m_cumulative / m_cumulative[-1]
    else:
        m_cumulative = m_cumulative[-1] - m_cumulative
    return masses[0:-1], m_cumulative


def main():
    np.random.seed(42)
    cosmo = cosmology.setCosmology('planck18', persistence='r')

    # Model parameters
    z = 0

    # Tilt parameters
    ns = 0.0
    ar = 0.05
    min_k = 10

    # Bump parameters
    amp = 100
    k0bump = 150
    bump_width_dex = 0.2

    # WDM parameters
    kcut = 150.

    # Transfer file for vEDE: This is the R = 60, log10 ac = -6.92 transfer function as in Figure 9 of arXiv:2409.06778, but with other parameters set to their Planck18 values. 
    tkfile = './vEDE_transfers/vEDE_transfer_60.0_-6.92.csv'

    # Axion kination npy files: this is the two-field model shown in Fig. 12 of arXiv:2510.01308. 
    ak1_kfile = './AxionKinationTransfers/kklist_fiducial_2field_TRM3keV_final.npy'
    ak1_pkfile = './AxionKinationTransfers/Pklist_fiducial_2field_TRM3keV_final.npy'
    ak1_pkref = './AxionKinationTransfers/Pklist_LCDM_bestfit_k1e4.npy'

    # Scale range
    k = 10**np.linspace(-1.5, 2.5, 200)
    M = np.logspace(6., 12, 50)
    R = peaks.lagrangianR(M)

    # --- Compute power spectra and sigma ---
    print('Initializing power spectrum...', flush=True)
    power_spectrum = ModifiedPowerSpectrum()

    reference_power_spectrum = np.empty((len(power_spectrum.kref), 2))
    reference_power_spectrum[:, 0] = np.log10(power_spectrum.kref)
    reference_power_spectrum[:, 1] = np.log10(power_spectrum.pk_tilted(0.0, 0.0))
    file_path = './reference.txt'
    np.savetxt(file_path, X=reference_power_spectrum)
    ps_args_reference = dict(model='reference_model', path=file_path, persistence='r')
    print('Computing P(k) and sigma: reference...', flush=True)
    pk_reference = cosmo.matterPowerSpectrum(k, **ps_args_reference)
    sigma_cdm = cosmo.sigma(R, ps_args=ps_args_reference)

    print('Computing P(k) and sigma: tilt...', flush=True)
    custom_pk = np.empty((len(power_spectrum.kref), 2))
    custom_pk[:, 0] = np.log10(power_spectrum.kref)
    custom_pk[:, 1] = np.log10(power_spectrum.pk_tilted(ns, ar, min_k))
    file_path = './custom.txt'
    np.savetxt(file_path, X=custom_pk)
    ps_args = dict(model='custom_model1', path=file_path, persistence='r')
    pk_tilt = cosmo.matterPowerSpectrum(k, **ps_args)
    sigma_tilt = cosmo.sigma(R, ps_args=ps_args)

    print('Computing P(k) and sigma: bump...', flush=True)
    custom_pk = np.empty((len(power_spectrum.kref), 2))
    custom_pk[:, 0] = np.log10(power_spectrum.kref)
    custom_pk[:, 1] = np.log10(power_spectrum.pk_bump(amp, k0bump, bump_width_dex))
    file_path = './custom.txt'
    np.savetxt(file_path, X=custom_pk)
    ps_args = dict(model='custom_model2', path=file_path, persistence='r')
    pk_bump = cosmo.matterPowerSpectrum(k, **ps_args)
    sigma_bump = cosmo.sigma(R, ps_args=ps_args)

    print('Computing P(k) and sigma: WDM...', flush=True)
    custom_pk = np.empty((len(power_spectrum.kref), 2))
    custom_pk[:, 0] = np.log10(power_spectrum.kref)
    custom_pk[:, 1] = np.log10(power_spectrum.pk_cutoff(kcut))
    file_path = './custom.txt'
    np.savetxt(file_path, X=custom_pk)
    ps_args = dict(model='custom_model3', path=file_path, persistence='r')
    pk_wdm = cosmo.matterPowerSpectrum(k, **ps_args)
    sigma_wdm = cosmo.sigma(R, ps_args=ps_args)

    print('Computing P(k) and sigma: vEDE...', flush=True)
    custom_pk = np.empty((len(power_spectrum.kref), 2))
    custom_pk[:, 0] = np.log10(power_spectrum.kref)
    custom_pk[:, 1] = np.log10(power_spectrum.pk_general(tkfile))
    file_path = './custom.txt'
    np.savetxt(file_path, X=custom_pk)
    ps_args = dict(model='custom_model4', path=file_path, persistence='r')
    pk_vEDE1 = cosmo.matterPowerSpectrum(k, **ps_args)
    sigma_vEDE1 = cosmo.sigma(R, ps_args=ps_args)

    print('Computing P(k) and sigma: AK1...', flush=True)
    custom_pk = np.empty((len(power_spectrum.kref), 2))
    custom_pk[:, 0] = np.log10(power_spectrum.kref)
    custom_pk[:, 1] = np.log10(power_spectrum.pk_from_npy(ak1_kfile, ak1_pkfile, ak1_pkref))
    file_path = './custom.txt'
    np.savetxt(file_path, X=custom_pk)
    ps_args = dict(model='custom_model5', path=file_path, persistence='r')
    pk_AK1 = cosmo.matterPowerSpectrum(k, **ps_args)
    sigma_AK1 = cosmo.sigma(R, ps_args=ps_args)

    # --- Plot P(k) and sigma(M) ---
    plt.figure()
    plt.loglog()
    plt.xlabel('k(h / Mpc)')
    plt.ylabel('P(k) (Mpc/h)^3')
    plt.plot(k, pk_reference, '-', color='k', label='CDM')
    #plt.plot(k, pk_tilt, '-', color='b')
    #plt.plot(k, pk_bump, '-', color='g')
    plt.plot(k, pk_wdm, '-', color='r', label='WDM')
    plt.plot(k, pk_vEDE1, '-', color='m', label='vEDE')
    plt.plot(k, pk_AK1, '-', color='c', label='AK1')
    plt.legend()
    plt.savefig('power_spectrum.pdf', bbox_inches='tight')

    plt.figure()
    plt.loglog()
    plt.xlabel('M (Msun/h)')
    plt.ylabel('sigma')
    plt.plot(M, sigma_cdm, '-', color='k', label='CDM')
    #plt.plot(M, sigma_tilt, '-', color='b')
    #plt.plot(M, sigma_bump, '-', color='g')
    plt.plot(M, sigma_wdm, '-', color='r', label='WDM')
    plt.plot(M, sigma_vEDE1, '-', color='m', label='vEDE')
    plt.plot(M, sigma_AK1, '-', color='c', label='AK1')
    plt.legend()
    plt.savefig('sigma_M.pdf', bbox_inches='tight')

    # --- Concentration-mass relations ---
    print('Computing concentrations: reference...', flush=True)
    cref = power_spectrum.evaluate_concentrationsMine(M, z, 0, 0)
    print('Computing concentrations: tilt...', flush=True)
    ctilt = power_spectrum.evaluate_concentrationsMine(M, z, ns, ar, min_k, pk_model='tilt')
    print('Computing concentrations: bump...', flush=True)
    cbump = power_spectrum.evaluate_concentrationsMine(M, z, amp, k0bump, bump_width_dex, pk_model='bump')
    print('Computing concentrations: WDM...', flush=True)
    cwdm = power_spectrum.evaluate_concentrationsMine(M, z, kcut, None, None, pk_model='wdm')
    print('Computing concentrations: vEDE...', flush=True)
    cvEDE1 = power_spectrum.evaluate_concentrationsMine(M, z, tkfile, None, None, pk_model='file')
    print('Computing concentrations: AK1...', flush=True)
    cAK1 = power_spectrum.evaluate_concentrationsMine(M, z, ak1_kfile, ak1_pkfile, ak1_pkref, pk_model='npy')

    plt.figure()
    plt.plot(M, cref, color='k', label='CDM')
    #plt.plot(M, ctilt, color='b')
    #plt.plot(M, cbump, color='g')
    plt.plot(M, cwdm, color='r', label='WDM')
    plt.plot(M, cvEDE1, color='m', label='vEDE')
    plt.plot(M, cAK1, color='c', label='AK1')
    plt.xlabel('halo mass (Msun/h)')
    plt.ylabel('concentration')
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.legend()
    plt.savefig('concentration_mass.pdf', bbox_inches='tight')

    # --- Mass functions ---
    print('Computing mass functions: reference...', flush=True)
    dndlnMref = power_spectrum.evaluate_mass_function(M, z, 0, 0)
    print('Computing mass functions: tilt...', flush=True)
    dndlnMtilt = power_spectrum.evaluate_mass_function(M, z, ns, ar, min_k, pk_model='tilt')
    print('Computing mass functions: bump...', flush=True)
    dndlnMbump = power_spectrum.evaluate_mass_function(M, z, amp, k0bump, bump_width_dex, pk_model='bump')
    print('Computing mass functions: WDM...', flush=True)
    dndlnMwdm = power_spectrum.evaluate_mass_function(M, z, kcut, None, None, pk_model='wdm')
    print('Computing mass functions: vEDE...', flush=True)
    dndlnMvEDE1 = power_spectrum.evaluate_mass_function(M, z, tkfile, None, None, pk_model='file')
    print('Computing mass functions: AK1...', flush=True)
    dndlnMAK1 = power_spectrum.evaluate_mass_function(M, z, ak1_kfile, ak1_pkfile, ak1_pkref, pk_model='npy')

    plt.figure()
    plt.plot(M, dndlnMref, color='k', label='CDM')
    #plt.plot(M, dndlnMtilt, color='b')
    #plt.plot(M, dndlnMbump, color='g')
    plt.plot(M, dndlnMwdm, color='r', label='WDM')
    plt.plot(M, dndlnMvEDE1, color='m', label='vEDE')
    plt.plot(M, dndlnMAK1, color='c', label='AK1')
    plt.xlabel('halo mass (Msun/h)')
    plt.ylabel('dndlnM (h/Mpc)^3')
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.legend()
    plt.savefig('mass_function.pdf', bbox_inches='tight')

    # Scale factors: ratio of total mass function integral relative to CDM.
    # dndlnM is dn/d(lnM), so the true integral is ∫ dn/d(lnM) d(lnM).
    # Since d(lnM) = ln(10) * d(log10 M), and the ln(10) cancels in the
    # ratio, we can integrate over log10(M) directly.
    log10M = np.log10(M)
    Nref = np.trapz(dndlnMref, log10M)
    scale_vEDE = np.trapz(dndlnMvEDE1, log10M) / Nref
    scale_wdm = np.trapz(dndlnMwdm, log10M) / Nref
    #scale_tilt = np.trapz(dndlnMtilt, log10M) / Nref
    scale_AK1 = np.trapz(dndlnMAK1, log10M) / Nref

    # --- Sample from mass functions ---
    num_draw_ref = 50000000
    print(f'Sampling from mass functions ({num_draw_ref} draws)...', flush=True)
    minlogM = np.log10(M[0])
    maxlogM = np.log10(M[-1])
    m_samples = sample_from_mass_function(num_draw_ref, M, dndlnMref)
    h, b = np.histogram(m_samples, bins=np.logspace(minlogM, maxlogM, 10))
    plt.figure()
    plt.loglog(b[0:-1], h, color='k', label='CDM')
    plt.xlabel('halo mass (Msun/h)')
    plt.ylabel('sampled N(M)')

    num_draw_vEDE = int(scale_vEDE * num_draw_ref)
    print(f'  vEDE: scale={scale_vEDE:.2f}, num_draw={num_draw_vEDE}', flush=True)
    m_samples = sample_from_mass_function(num_draw_vEDE, M, dndlnMvEDE1)
    h, b = np.histogram(m_samples, bins=np.logspace(minlogM, maxlogM, 10))
    plt.loglog(b[0:-1], h, color='m', label='vEDE')

    num_draw_wdm = int(scale_wdm * num_draw_ref)
    print(f'  WDM: scale={scale_wdm:.2f}, num_draw={num_draw_wdm}', flush=True)
    m_samples = sample_from_mass_function(num_draw_wdm, M, dndlnMwdm)
    h, b = np.histogram(m_samples, bins=np.logspace(minlogM, maxlogM, 10))
    plt.loglog(b[0:-1], h, color='r', label='WDM')

    #num_draw_tilt = int(scale_tilt * num_draw_ref)
    #print(f'  Tilt: scale={scale_tilt:.2f}, num_draw={num_draw_tilt}', flush=True)
    #m_samples = sample_from_mass_function(num_draw_tilt, M, dndlnMtilt)
    #h, b = np.histogram(m_samples, bins=np.logspace(minlogM, maxlogM, 10))
    #plt.loglog(b[0:-1], h, color='b')

    num_draw_AK1 = int(scale_AK1 * num_draw_ref)
    print(f'  AK1: scale={scale_AK1:.2f}, num_draw={num_draw_AK1}', flush=True)
    m_samples = sample_from_mass_function(num_draw_AK1, M, dndlnMAK1)
    h, b = np.histogram(m_samples, bins=np.logspace(minlogM, maxlogM, 10))
    plt.loglog(b[0:-1], h, color='c', label='AK1')

    plt.legend()
    plt.savefig('sampled_mass_distribution.pdf', bbox_inches='tight')

    # --- Generate mock halo populations and plot m_enc vs r_half ---
    m_min_gen = 10**8
    N_cdm = 100000000
    N_vEDE1 = int(N_cdm * scale_vEDE)
    N_wdm = int(N_cdm * scale_wdm)
    N_AK1 = int(N_cdm * scale_AK1)
    print(f'Generating samples: CDM (N={N_cdm})...', flush=True)
    rhalf_cdm, menc_cdm, m_samples_cdm, c_samples_cdm = generate_samples(N_cdm, M, dndlnMref, cref, m_min=m_min_gen)
    #print('Generating samples: tilt...', flush=True)
    #rhalf_tilt, menc_tilt, m_samples_tilt, c_samples_tilt = generate_samples(
    #    int(N_cdm * scale_tilt), M, dndlnMtilt, ctilt, m_min=m_min_gen)
    print(f'Generating samples: vEDE (N={N_vEDE1})...', flush=True)
    rhalf_vEDE1, menc_vEDE1, m_samples_vEDE1, c_samples_vEDE1 = generate_samples(
        N_vEDE1, M, dndlnMvEDE1, cvEDE1, m_min=m_min_gen)
    print(f'Generating samples: WDM (N={N_wdm})...', flush=True)
    rhalf_wdm, menc_wdm, m_samples_wdm, c_samples_wdm = generate_samples(
        N_wdm, M, dndlnMwdm, cwdm, m_min=m_min_gen)
    print(f'Generating samples: AK1 (N={N_AK1})...', flush=True)
    rhalf_AK1, menc_AK1, m_samples_AK1, c_samples_AK1 = generate_samples(
        N_AK1, M, dndlnMAK1, cAK1, m_min=m_min_gen)

    fig = plt.figure()
    ax = plt.subplot(111)
    plot_frac = 500.0/len(rhalf_cdm) # plot 500 points for CDM
    n_cdm = int(len(rhalf_cdm) * plot_frac)
    n_vEDE1 = int(len(rhalf_vEDE1) * plot_frac)
    n_wdm = int(len(rhalf_wdm) * plot_frac)
    n_AK1 = int(len(rhalf_AK1) * plot_frac)
    idx_cdm = np.random.choice(len(rhalf_cdm), n_cdm, replace=False)
    idx_vEDE1 = np.random.choice(len(rhalf_vEDE1), n_vEDE1, replace=False)
    idx_wdm = np.random.choice(len(rhalf_wdm), n_wdm, replace=False)
    idx_AK1 = np.random.choice(len(rhalf_AK1), n_AK1, replace=False)
    ax.scatter(rhalf_cdm[idx_cdm], menc_cdm[idx_cdm] / m_samples_cdm[idx_cdm], color='k', s=1, marker='o', label='CDM')
    #ax.scatter(rhalf_tilt, menc_tilt / m_samples_tilt, color='b', s=1, label='Tilt')
    ax.scatter(rhalf_vEDE1[idx_vEDE1], menc_vEDE1[idx_vEDE1] / m_samples_vEDE1[idx_vEDE1], color='m', s=1, marker='^', label='vEDE')
    ax.scatter(rhalf_wdm[idx_wdm], menc_wdm[idx_wdm] / m_samples_wdm[idx_wdm], color='r', s=1, marker='s', label='WDM')
    ax.scatter(rhalf_AK1[idx_AK1], menc_AK1[idx_AK1] / m_samples_AK1[idx_AK1], color='c', s=1, marker='d', label='AK1')
    ax.set_xlabel('r_half (kpc)')
    ax.set_ylabel('M_enc(r_half)/ M')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    plt.savefig('menc_vs_rhalf.pdf', bbox_inches='tight')

    # --- Complementary CDF of enclosed mass ---
    print('Computing enclosed mass CDFs...', flush=True)
    num_bins = 50
    log10m_min = 4
    log10m_max = 10
    mplot, cdf_cdm = cumulative_mass_function(menc_cdm, m_min=log10m_min, m_max=log10m_max, num_bins=num_bins, normed=False)
    #_, cdf_tilt = cumulative_mass_function(menc_tilt, m_min=log10m_min, m_max=log10m_max, num_bins=num_bins, normed=False)
    _, cdf_vede = cumulative_mass_function(menc_vEDE1, m_min=log10m_min, m_max=log10m_max, num_bins=num_bins, normed=False)
    _, cdf_wdm = cumulative_mass_function(menc_wdm, m_min=log10m_min, m_max=log10m_max, num_bins=num_bins, normed=False)
    _, cdf_AK1 = cumulative_mass_function(menc_AK1, m_min=log10m_min, m_max=log10m_max, num_bins=num_bins, normed=False)

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(mplot, cdf_cdm, color='k', label='CDM')
    #ax.plot(mplot, cdf_tilt, color='b', label='Tilt')
    ax.plot(mplot, cdf_vede, color='m', label='vEDE')
    ax.plot(mplot, cdf_wdm, color='r', label='WDM')
    ax.plot(mplot, cdf_AK1, color='c', label='AK1')
    ax.set_xlabel('enclosed mass (Msun)')
    ax.set_ylabel('N(>M_enc)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    plt.savefig('menc_cdf.pdf', bbox_inches='tight')



if __name__ == '__main__':
    main()

