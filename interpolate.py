import h5py
import gvar as gv
import numpy as np
import matplotlib.pyplot as plt
import lsqfit
from scipy.optimize import fsolve

def collect_data(datafile):
    """return parameters and data for effective masses"""
    averaged_data = {}
    with h5py.File(datafile, 'r') as f:
        ds_group = f["D_s"]
        cfg_keys = [key for key in ds_group.keys() if key.startswith("cfg_")]
        masses = sorted(['0.35', '0.365', '0.385', '0.4'])
        print(f"Using masses: {masses}")
        for mass in masses:
            correlators = []
            for cfg_key in cfg_keys:
                correlator_key = f"D_s/{cfg_key}/mass_{mass}/smsm/gamma_15/mom_0/correlator"
                if correlator_key in f:
                    correlators.append(f[correlator_key][:])
            if correlators:
                avg_correlator = gv.dataset.avg_data(correlators)
                folded_correlator = 0.5 * (avg_correlator + avg_correlator[::-1])
                averaged_data[mass] = folded_correlator
                print(f"Debug: averaged_data[{mass}] sample: {folded_correlator[:5]} (gvar: {isinstance(folded_correlator[0], gv.GVar)})")

    # Compute effective masses and average over fit range to get single gvars
    effective_mass = {}
    fit_range = slice(8, 35)
    for mass in averaged_data:
        rolled = np.roll(averaged_data[mass][:-1], -1)
        effective_mass_array = np.log(averaged_data[mass][:-1] / rolled)
        if effective_mass_array.size:
            eff_mass_in_range = effective_mass_array[fit_range]
            if len(eff_mass_in_range) > 0:
                effective_mass[mass] = gv.gvar(gv.mean(eff_mass_in_range),gv.sdev(eff_mass_in_range))  # Single gvar per mass
                print(f"Effective mass for mass_{mass} (averaged): {effective_mass[mass]}")
        else:
            print(f"No valid effective mass data for mass_{mass}")

    # Parameters: (1/a, a*m) for each mass
    a = 0.065 
    hbarc = 197.3269804
    a_inv = hbarc / (a * 1000)  #GeV 
    param = {mass: (a_inv, float(mass) * a_inv) for mass in masses if mass in effective_mass}
    data = {mass: np.average(effective_mass[mass] * a_inv) for mass in effective_mass}  # convert to GeV
    print("Collected data:", data)
    return param, data

def make_fcn_prior(param):
    """returns fit function and prior
    see https://lsqfit.readthedocs.io/en/latest/case-spline.html
    """
    def F(p):
        f = gv.cspline.CSpline(p['mknot'], p['fknot'])
        ans = {}
        for s in param:
            ainv, am = param[s]
            m = am / ainv  # am -> phys
            ans[s] = f(m)
            for i, ci in enumerate(p['c']):
                ans[s] += ci * (am / ainv) ** (2 * (i + 1))  # poly correction
        return ans
    # Define prior for spline knots and coefficients
    prior = gv.gvar(dict(
        mknot=['0.3(1)', '0.4(1)', '0.5(1)', '0.6(1)'],  # knots in lattice units, adjusted to mass range
        fknot=['1.9(1)', '1.95(1)', '2.0(1)', '2.05(1)'],  # knot values in GeV, near D_s mass
        c=['0(1)'] * 5,  # poly coeffs
    ))
    return F, prior

def main():
    masses = sorted(['0.35', '0.365', '0.385', '0.4'])
    param, data = collect_data('D_s_final.h5')
    F, prior = make_fcn_prior(param)
    print('param', param)
    print('data', data)
    
    print('Performing fit..')
    fit = lsqfit.nonlinear_fit(data=data, prior=prior, fcn=F, debug=True)
    print("\nFit results:")
    print(fit)
    print("Fitted mknot:", fit.p['mknot'])
    print("Fitted fknot:", fit.p['fknot'])
    print("Fitted c:", fit.p['c'])

    f = gv.cspline.CSpline(fit.p['mknot'], fit.p['fknot'])

    # get F(p) and f(m) at the data points
    fit_output = F(fit.p)
    print("\nDebug: Fit function F(p) at data points:")
    for s in param:
        ainv, am = param[s]
        m = am / ainv
        print(f"Mass {s}: F(p) = {fit_output[s]}, Spline f(m) = {f(m)}, Data = {data[s]}")

    # Interpolate to physical D_s mass using the full fit function F(p)
    m_ds_physical = 1.96835 
    def residual_float(m):
        a = 0.065 
        hbarc = 197.3269804
        a_inv = hbarc / (a * 1000)  #GeV 
        temp_param = {'temp': (a_inv, m * a_inv)}  # Use actual a_inv, am = m * a_inv
        def F_temp(p):
            f = gv.cspline.CSpline(p['mknot'], p['fknot'])
            ans = {}
            for s in temp_param:
                ainv, am = temp_param[s]
                m_val = am / ainv
                ans[s] = f(m_val)
                for i, ci in enumerate(p['c']):
                    ans[s] += ci * (m_val) ** (2 * (i + 1))
            return ans
        F_temp_result = F_temp(fit.p)
        return gv.mean(F_temp_result['temp']) - m_ds_physical
    m_c_physical = fsolve(residual_float, x0=0.365)[0] 
    print('mc', m_c_physical)
    # def residual(m):
    #     a = 0.065 
    #     hbarc = 197.3269804
    #     a_inv = hbarc / a  #GeV 
    #     temp_param = {'temp': (1.0, m)}  # Dummy ainv=1.0, am=m
    #     # Define a new F function with the temporary parameter dictionary
    #     def F_temp(p):
    #         f = gv.cspline.CSpline(p['mknot'], p['fknot'])
    #         ans = {}
    #         for s in temp_param:
    #             ainv, am = temp_param[s]
    #             m_val = am / ainv
    #             ans[s] = f(m_val)
    #             for i, ci in enumerate(p['c']):
    #                 ans[s] += ci * (m_val) ** (2 * (i + 1))
    #         return ans
    #     F_temp_result = F_temp(fit.p)
    #     return gv.mean(F_temp_result['temp']) - m_ds_physical
    # m_c_physical = fsolve(residual, x0=0.365)[0]  # Adjusted initial guess
    # print('mc',m_c_physical)
    m_c_physical_gvar = gv.gvar(m_c_physical, 0.001)  # TODO FIX THIS SHOULD BE GV.SDEV
    print(f"Interpolated charm quark mass at physical D_s point: {m_c_physical_gvar:.4f} (lattice units)")

    # Create error budget
    outputs = {'f(1)': f(1), 'f(5)': f(5), 'f(9)': f(9), 'm_c_physical': m_c_physical_gvar}
    inputs = {'data': data}
    inputs.update(prior)

    plt.figure(figsize=(10, 6))
    x_positions = np.arange(len(masses))
    mass_values = np.array([float(m) for m in masses if m in data])
    eff_masses = np.array([gv.mean(data[m]) for m in masses if m in data])  # Already averaged floats
    eff_errors = np.array([gv.sdev(data[m]) for m in masses if m in data])  # Already averaged floats
    plt.errorbar(x_positions[:len(mass_values)], eff_masses, yerr=eff_errors, fmt='o', label="Effective Mass (smsm, g_15)", capsize=5)
    print('efff_masses',eff_masses)
    # plt.plot(x_positions[:len(mass_values)], eff_masses, 'o', label="Effective Mass (smsm, Gamma 15)")

    # Plot the full fit function F(p) instead of just the spline
    x_fit = np.linspace(min(mass_values), max(mass_values), 100)
    y_fit = []
    for m in x_fit:
        temp_param = {'temp': (1.0, m)}  # Dummy ainv=1.0, am=m
        def F_temp(p):
            f = gv.cspline.CSpline(p['mknot'], p['fknot'],extrap_order=2)
            ans = {}
            for s in temp_param:
                ainv, am = temp_param[s]
                m_val = am / ainv
                ans[s] = f(m_val)
                for i, ci in enumerate(p['c']):
                    ans[s] += ci * (m_val) ** (2 * (i + 1)) # polynomial
                    #ans[s] += ci * (m_val) # w/o poly

            return ans
        F_temp_result = F_temp(fit.p)
        y_fit.append(gv.mean(F_temp_result['temp']))
    plt.plot(np.linspace(0, len(masses)-1, 100), y_fit, 'b-', label="Fit Function (Spline + Polynomial)")

    plt.axhline(y=m_ds_physical, color='r', linestyle='--', label=f"D_s(PDG) = {m_ds_physical:.5f} GeV")
    plt.axvline(x=np.interp(m_c_physical, mass_values, x_positions), color='g', linestyle='--',
                label=f"Interpolated m_c = {m_c_physical:.6f}")

    plt.xticks(x_positions, masses)
    plt.xlabel("m_c (a_fm)")
    plt.ylabel("eff mass(GeV)")
    plt.title("Avg eff mass in t=[10,25) w/ Interpolation to Physical D_s")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Ds.png')
    plt.show()

if __name__ == "__main__":
    main()