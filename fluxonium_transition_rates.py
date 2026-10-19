r"""
Fluxonium Qubit Transition Rate Simulation & Plotting
======================================================
This script simulates a standalone fluxonium qubit with parameter set:
    EJ = 5.3 GHz, EC = 1.36 GHz, EL = 0.146 GHz, flux = 0.0, cutoff = 110, N_LEVELS = 6

It models the transition rates Gamma_{j -> i} between all pairs of energy levels (0, 1, 2, 3, 4, 5)
for three environmental coupling scenarios at two temperatures (T = 0 K and T = 70 mK):
    1. Uncoupled / Intrinsic (Dielectric loss in shunt capacitance)
    2. Capacitively coupled to a Transmission Line (Z0 = 50 Ohm)
    3. Capacitively coupled to a Bandpass Filter (Purcell Filter)

Formulas and Literature References:
------------------------------------
1. Fermi's Golden Rule for Charge-Noise / Capacitive Decay:
   \Gamma_{j \to i} = \left(\frac{2e}{\hbar}\right)^2 |\langle i | \hat{n} | j \rangle|^2 S_{VV}(\omega_{ij})
   Reference:
   - Groszkowski et al., "scqubits: a Python package for superconducting qubits", Quantum 5, 583 (2021). arXiv:2107.08552
   - Schoelkopf et al., "Qubits as spectrometers of quantum noise", Physical Review Letters (2003) / arXiv:cond-mat/0205339

2. Environmental Voltage Noise Power Spectral Density S_{VV}(\omega):
   S_{VV}(\omega) = \frac{2 \hbar \omega}{1 - e^{-\hbar \omega / (k_B T)}} \text{Re}[Z_{\text{env}}(\omega)]
   Reference:
   - Clerk et al., "Introduction to quantum noise, measurement, and amplification", Rev. Mod. Phys. 82, 1155 (2010). arXiv:0810.4729

3. Environmental Effective Impedances \text{Re}[Z_{\text{env}}(\omega)]:
   a) Uncoupled (Dielectric loss with quality factor Q_cap):
      \text{Re}[Z_{\text{uncoupled}}(\omega)] = \frac{1}{\omega C_{\Sigma} Q_{\text{cap}}}
      Reference:
      - Nguyen et al., "High-coherence fluxonium qubit", Phys. Rev. X 9, 041041 (2019). arXiv:1810.11006
      - Smith et al., "Superconducting circuit elements with high kinetic inductance", NPJ Quantum Information (2020)

   b) Transmission Line (Capacitive coupling C_c to 50 Ohm line):
      \text{Re}[Z_{\text{TL}}(\omega)] = \frac{\omega^2 C_c^2 Z_0}{1 + \omega^2 C_c^2 Z_0^2}
      Reference:
      - Houck et al., "Controlling the spontaneous emission rate of a superconducting qubit", Phys. Rev. Lett. 101, 080502 (2008). arXiv:0803.4490

   c) Bandpass Purcell Filter (Capacitive coupling C_c to LC Bandpass Filter):
      \text{Re}[Z_{\text{BPF}}(\omega)] = \omega^2 C_c^2 \frac{Z_0}{1 + Q_f^2 (\omega/\omega_f - \omega_f/\omega)^2}
      Reference:
      - Reed et al., "Fast reset and readout of a superconducting qubit using a bandpass filter", Appl. Phys. Lett. 96, 203110 (2010). arXiv:1004.4323
      - Cleland et al., "Broadband Bandpass Purcell Filter for Circuit Quantum Electrodynamics", arXiv:2306.06258 (2023)

Dependencies:
------------
    pip install scqubits qutip matplotlib numpy scipy
"""

import os
import sys

# Ensure installed site-packages qutip is imported if running from inside qutip source directory
repo_dir = os.path.abspath(os.path.dirname(__file__))
if repo_dir in sys.path:
    sys.path.remove(repo_dir)
if '' in sys.path:
    sys.path.remove('')

import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import qutip as qt
import scqubits as scq

# Re-add script directory to sys.path for standard imports
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

# Set plot style for publication-quality figures
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'font.family': 'sans-serif'
})


def setup_fluxonium_system():
    """
    Constructs the Fluxonium qubit using scqubits and extracts system properties
    (eigenenergies, number operator matrix elements, and qutip representation).
    """
    EJ = 5.3    # Josephson energy in GHz
    EC = 1.36   # Charging energy in GHz
    EL = 0.146  # Inductive energy in GHz
    flux = 0.0  # External flux bias in units of Phi_0
    cutoff = 110 # Harmonic oscillator cutoff basis size
    N_LEVELS = 6 # Number of energy levels to extract

    print("=" * 70)
    print("FLUXONIUM QUBIT PARAMETERS")
    print("=" * 70)
    print(f"EJ = {EJ} GHz | EC = {EC} GHz | EL = {EL} GHz | flux = {flux} Phi_0")
    print(f"Cutoff basis size = {cutoff} | N_LEVELS = {N_LEVELS}")

    # Build scqubits Fluxonium object
    fluxonium = scq.Fluxonium(EJ=EJ, EC=EC, EL=EL, flux=flux, cutoff=cutoff)
    evals, evecs = fluxonium.eigensys(evals_count=N_LEVELS)

    # Calculate transition energies (relative to ground state E0)
    evals_rel_ghz = evals - evals[0]
    print("\nCalculated Eigenenergies (relative to E0):")
    for idx, E in enumerate(evals_rel_ghz):
        print(f"  E_{idx} = {E:.4f} GHz")

    # Extract matrix elements of charge operator n = Q / (2e)
    n_mat = fluxonium.matrixelement_table('n_operator', evals_count=N_LEVELS)

    # QuTiP representation of Hamiltonian & n_operator in qubit eigenbasis
    H_qutip = qt.Qobj(np.diag(evals_rel_ghz))
    n_qutip = qt.Qobj(n_mat)

    return {
        'EJ': EJ,
        'EC': EC,
        'EL': EL,
        'N_LEVELS': N_LEVELS,
        'evals_ghz': evals_rel_ghz,
        'n_mat': n_mat,
        'H_qutip': H_qutip,
        'n_qutip': n_qutip
    }


def compute_environmental_impedance(w_rad_s, env_type, params):
    """
    Computes real part of environmental impedance Re[Z_env(w)] in Ohms.
    """
    abs_w = np.abs(w_rad_s)
    if abs_w < 1e-6:
        return 0.0

    if env_type == 'uncoupled':
        # Re[Z(w)] = 1 / (w * C_sigma * Q_cap)
        C_sigma = params['C_sigma']
        Q_cap = params['Q_cap']
        return 1.0 / (abs_w * C_sigma * Q_cap)

    elif env_type == 'transmission_line':
        # Re[Z_eff(w)] = (w^2 * Cc^2 * Z0) / (1 + w^2 * Cc^2 * Z0^2)
        Cc = params['Cc']
        Z0 = params['Z0']
        num = (abs_w**2) * (Cc**2) * Z0
        den = 1.0 + (abs_w**2) * (Cc**2) * (Z0**2)
        return num / den

    elif env_type == 'bandpass_filter':
        # Bandpass Filter response: Z_bpf(w) = Z0 / [1 + Q_f^2 * (w/w_f - w_f/w)^2]
        # Effective Re[Z_eff(w)] = w^2 * Cc^2 * Z_bpf(w)
        Cc = params['Cc']
        Z0 = params['Z0']
        w_f = 2.0 * np.pi * params['f_filter_hz']
        Q_f = params['Q_filter']

        Z_bpf = Z0 / (1.0 + (Q_f**2) * ((abs_w / w_f) - (w_f / abs_w))**2)
        return (abs_w**2) * (Cc**2) * Z_bpf

    else:
        raise ValueError(f"Unknown environment type: {env_type}")


def compute_noise_psd(w_rad_s, T_kelvin, Re_Z):
    """
    Computes S_VV(w) voltage noise spectral density in V^2 / Hz.
    S_VV(w) = 2 * hbar * w / (1 - exp(-hbar * w / (kB * T))) * Re[Z(w)]
    """
    if Re_Z <= 0.0:
        return 0.0

    kB = const.k
    hbar = const.hbar

    if T_kelvin == 0.0:
        if w_rad_s > 0.0: # Emission (transition from higher to lower energy level)
            return 2.0 * hbar * w_rad_s * Re_Z
        else: # Absorption is impossible at T = 0 K
            return 0.0
    else:
        exponent = (hbar * w_rad_s) / (kB * T_kelvin)
        if exponent > 700.0:
            return 0.0
        elif exponent < -700.0:
            return -2.0 * hbar * w_rad_s * Re_Z
        else: # Standard thermal formula
            return (2.0 * hbar * w_rad_s / (1.0 - np.exp(-exponent))) * Re_Z


def calculate_transition_rates(system_data, T_kelvin, env_type, params):
    """
    Calculates transition rates Gamma_{j -> i} for all level pairs (j != i).
    """
    evals_ghz = system_data['evals_ghz']
    n_mat = system_data['n_mat']
    N_LEVELS = system_data['N_LEVELS']
    e = const.e
    hbar = const.hbar

    transitions = []

    for j in range(N_LEVELS): # Initial state j
        for i in range(N_LEVELS): # Final state i
            if i == j:
                continue

            # Energy difference E_j - E_i (GHz)
            delta_E_ghz = evals_ghz[j] - evals_ghz[i]
            w_ij = 2.0 * np.pi * delta_E_ghz * 1e9 # rad/s

            abs_delta_E_ghz = np.abs(delta_E_ghz)
            if abs_delta_E_ghz < 1e-6:
                continue

            # Environmental impedance Re[Z(w)]
            Re_Z = compute_environmental_impedance(w_ij, env_type, params)

            # Noise Spectral Density S_VV(w)
            S_vv = compute_noise_psd(w_ij, T_kelvin, Re_Z)

            # Matrix element squared |<i|n|j>|^2
            n_element_sq = np.abs(n_mat[i, j])**2

            # Fermi's Golden Rule rate Gamma_{j -> i}
            # Gamma = (2e / hbar)^2 * |n_ij|^2 * S_VV(w)
            rate = ((2.0 * e / hbar)**2) * n_element_sq * S_vv

            if rate > 0:
                transitions.append({
                    'from': j,
                    'to': i,
                    'label': f"{j}->{i}",
                    'transition_type': 'decay' if j > i else 'excitation',
                    'energy_ghz': abs_delta_E_ghz,
                    'rate_s1': rate
                })

    return transitions


def create_base_plot(transitions, env_name, temp_str, filename):
    """
    Generates one of the 6 individual transition rate vs transition frequency plots.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Split into decay (j > i) and excitation (j < i)
    decays = [t for t in transitions if t['transition_type'] == 'decay']
    excitations = [t for t in transitions if t['transition_type'] == 'excitation']

    if decays:
        x_d = [t['energy_ghz'] for t in decays]
        y_d = [t['rate_s1'] for t in decays]
        ax.scatter(x_d, y_d, color='blue', s=70, zorder=4, label='Decay ($j \\to i$, $j > i$)')

        for t in decays:
            ax.annotate(t['label'], (t['energy_ghz'], t['rate_s1']),
                        textcoords="offset points", xytext=(5, 5), ha='left',
                        fontsize=9, color='darkblue', fontweight='bold')

    if excitations:
        x_e = [t['energy_ghz'] for t in excitations]
        y_e = [t['rate_s1'] for t in excitations]
        ax.scatter(x_e, y_e, color='red', marker='^', s=70, zorder=4, label='Excitation ($j \\to i$, $j < i$)')

        for t in excitations:
            ax.annotate(t['label'], (t['energy_ghz'], t['rate_s1']),
                        textcoords="offset points", xytext=(5, -12), ha='left',
                        fontsize=9, color='darkred', fontweight='bold')

    ax.set_yscale('log')
    ax.set_xlabel('Transition Energy $\\hbar\\omega_{ij} / h$ (GHz)')
    ax.set_ylabel('Transition Rate $\\Gamma_{j \\to i}$ (s$^{-1}$)')
    ax.set_title(f'Fluxonium Transition Rates ({env_name}, T = {temp_str})')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved plot: {filename}")


def create_temp_overlay_plot(trans_0K, trans_70mK, env_name, filename):
    """
    Generates overlay plot comparing T = 0 K vs T = 70 mK for a given environment.
    """
    fig, ax = plt.subplots(figsize=(8.5, 6))

    x_0 = [t['energy_ghz'] for t in trans_0K]
    y_0 = [t['rate_s1'] for t in trans_0K]
    ax.scatter(x_0, y_0, color='blue', s=80, marker='o', alpha=0.8, zorder=4, label='T = 0 K (Decay only)')

    x_70 = [t['energy_ghz'] for t in trans_70mK]
    y_70 = [t['rate_s1'] for t in trans_70mK]
    ax.scatter(x_70, y_70, color='crimson', s=60, marker='x', zorder=5, label='T = 70 mK (Decay + Excitation)')

    for t in trans_70mK:
        color = 'darkred' if t['transition_type'] == 'excitation' else 'darkblue'
        offset = (5, -10) if t['transition_type'] == 'excitation' else (5, 5)
        ax.annotate(t['label'], (t['energy_ghz'], t['rate_s1']),
                    textcoords="offset points", xytext=offset, ha='left',
                    fontsize=8.5, color=color)

    ax.set_yscale('log')
    ax.set_xlabel('Transition Energy $\\hbar\\omega_{ij} / h$ (GHz)')
    ax.set_ylabel('Transition Rate $\\Gamma_{j \\to i}$ (s$^{-1}$)')
    ax.set_title(f'Temperature Overlay: 0 K vs 70 mK ({env_name})')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved overlay plot: {filename}")


def create_env_overlay_plot(trans_TL, trans_BPF, temp_str, filename):
    """
    Generates overlay plot comparing Transmission Line vs Bandpass Filter for a given temperature.
    """
    fig, ax = plt.subplots(figsize=(8.5, 6))

    x_tl = [t['energy_ghz'] for t in trans_TL]
    y_tl = [t['rate_s1'] for t in trans_TL]
    ax.scatter(x_tl, y_tl, color='teal', s=80, marker='s', alpha=0.8, zorder=4, label='Transmission Line ($Z_0 = 50\\,\\Omega$)')

    x_bpf = [t['energy_ghz'] for t in trans_BPF]
    y_bpf = [t['rate_s1'] for t in trans_BPF]
    ax.scatter(x_bpf, y_bpf, color='magenta', s=80, marker='D', alpha=0.8, zorder=5, label='Bandpass Filter (Purcell Protected)')

    for t in trans_BPF:
        ax.annotate(t['label'], (t['energy_ghz'], t['rate_s1']),
                    textcoords="offset points", xytext=(5, 5), ha='left',
                    fontsize=8.5, color='darkmagenta')

    ax.set_yscale('log')
    ax.set_xlabel('Transition Energy $\\hbar\\omega_{ij} / h$ (GHz)')
    ax.set_ylabel('Transition Rate $\\Gamma_{j \\to i}$ (s$^{-1}$)')
    ax.set_title(f'Environment Overlay: Transmission Line vs BPF (T = {temp_str})')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved overlay plot: {filename}")


def main():
    print("=" * 70)
    print("FLUXONIUM TRANSITION RATE SIMULATION (qutip & scqubits)")
    print("=" * 70)

    # 1. Setup system
    system_data = setup_fluxonium_system()

    # 2. Environmental circuit parameters
    EC_J = system_data['EC'] * 1e9 * const.h
    C_sigma = (const.e**2) / (2.0 * EC_J)

    env_params = {
        'C_sigma': C_sigma,       # Fluxonium total capacitance (F)
        'Q_cap': 1e6,             # Capacitive quality factor (Nguyen et al. 2019)
        'Cc': 10e-15,             # Coupling capacitance 10 fF
        'Z0': 50.0,               # Characteristic impedance 50 Ohm
        'f_filter_hz': 6.0e9,     # BPF center frequency 6 GHz
        'Q_filter': 10.0          # BPF quality factor
    }

    print("\nEnvironmental & Coupling Parameters:")
    print(f"  C_sigma (total qubit cap) = {C_sigma*1e15:.2f} fF")
    print(f"  Q_cap (dielectric Q)     = {env_params['Q_cap']:.1e}")
    print(f"  Cc (coupling cap)        = {env_params['Cc']*1e15:.2f} fF")
    print(f"  Z0 (line impedance)      = {env_params['Z0']} Ohm")
    print(f"  BPF center frequency     = {env_params['f_filter_hz']/1e9} GHz")
    print(f"  BPF quality factor Q     = {env_params['Q_filter']}")

    # Output directory for plots
    out_dir = "transition_rate_plots"
    os.makedirs(out_dir, exist_ok=True)

    # 3. Calculate transition rates for all combinations
    environments = [
        ('uncoupled', 'Uncoupled (Intrinsic)'),
        ('transmission_line', 'Transmission Line'),
        ('bandpass_filter', 'Bandpass Filter')
    ]
    temperatures = [
        (0.0, '0K'),
        (0.070, '70mK')
    ]

    all_results = {}

    print("\n" + "=" * 70)
    print("SIMULATION RESULTS TABLE")
    print("=" * 70)

    for env_key, env_name in environments:
        all_results[env_key] = {}
        for T_k, T_str in temperatures:
            rates = calculate_transition_rates(system_data, T_k, env_key, env_params)
            all_results[env_key][T_str] = rates

            print(f"\n---> Environment: {env_name} | Temperature: {T_str}")
            print(f"{'Transition':<12} {'Type':<12} {'Energy (GHz)':<15} {'Rate (s^-1)':<15}")
            print("-" * 55)
            for t in sorted(rates, key=lambda x: x['label']):
                print(f"{t['label']:<12} {t['transition_type']:<12} {t['energy_ghz']:<15.4f} {t['rate_s1']:<15.4e}")

            # Plot individual base plot (1 of 6)
            filename = os.path.join(out_dir, f"1_base_{env_key}_{T_str}.png")
            create_base_plot(rates, env_name, T_str, filename)

    # 4. Generate Overlay Plots
    print("\n" + "=" * 70)
    print("GENERATING OVERLAY PLOTS")
    print("=" * 70)

    # 0K vs 70mK Overlays for each environment (3 plots)
    for env_key, env_name in environments:
        filename = os.path.join(out_dir, f"2_overlay_temp_{env_key}_0K_vs_70mK.png")
        create_temp_overlay_plot(all_results[env_key]['0K'], all_results[env_key]['70mK'], env_name, filename)

    # TL vs BPF Overlays for each temperature (2 plots)
    for _, T_str in temperatures:
        filename = os.path.join(out_dir, f"3_overlay_env_TL_vs_BPF_{T_str}.png")
        create_env_overlay_plot(all_results['transmission_line'][T_str],
                                all_results['bandpass_filter'][T_str],
                                T_str, filename)

    print("\n" + "=" * 70)
    print("LITERATURE REFERENCES & SOURCES")
    print("=" * 70)
    print("1. scqubits & QuTiP Numerical Architecture:")
    print("   - Groszkowski & Koch, 'scqubits: a Python package for superconducting qubits', Quantum 5, 583 (2021). DOI:10.22331/q-2021-07-09-583")
    print("2. Quantum Noise & Transition Rate Formulas (Fermi's Golden Rule):")
    print("   - Schoelkopf et al., 'Qubits as spectrometers of quantum noise', arXiv:cond-mat/0205339")
    print("   - Clerk et al., 'Introduction to quantum noise, measurement, and amplification', Rev. Mod. Phys. 82, 1155 (2010). DOI:10.1103/RevModPhys.82.1155")
    print("3. Intrinsic Dielectric Dissipation in Fluxonium:")
    print("   - Nguyen et al., 'High-coherence fluxonium qubit', Phys. Rev. X 9, 041041 (2019). DOI:10.1103/PhysRevX.9.041041")
    print("4. Transmission Line Purcell Effect:")
    print("   - Houck et al., 'Controlling the spontaneous emission rate of a superconducting qubit', Phys. Rev. Lett. 101, 080502 (2008). DOI:10.1103/PhysRevLett.101.080502")
    print("5. Bandpass Purcell Filter Protection:")
    print("   - Reed et al., 'Fast reset and readout of a superconducting qubit using a bandpass filter', Appl. Phys. Lett. 96, 203110 (2010). DOI:10.1063/1.3435463")
    print("   - Cleland et al., 'Broadband Bandpass Purcell Filter for Circuit Quantum Electrodynamics', arXiv:2306.06258 (2023)")
    print("=" * 70)
    print(f"Simulation finished successfully! All 11 plots saved in './{out_dir}/'")


if __name__ == '__main__':
    main()
