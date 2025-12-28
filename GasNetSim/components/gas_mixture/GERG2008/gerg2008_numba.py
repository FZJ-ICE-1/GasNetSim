#   #!/usr/bin/env python
#   -*- coding: utf-8 -*-
#   ******************************************************************************
#     Copyright (c) 2025.
#     Developed by Yifei Lu
#     Last change on 1/6/25, 4:13 PM
#     Last change by yifei
#    *****************************************************************************
import math
from numba import njit, float64, types, int32, boolean
from numba.extending import overload

from .gerg2008_constants import *
from .gerg2008 import number_of_atoms


@njit(float64(float64), fastmath=True, cache=True, nogil=True)
def Tanh_numba(xx):
    return (math.exp(xx) - math.exp(-xx)) / (math.exp(xx) + math.exp(-xx))


@njit(float64(float64), fastmath=True, cache=True, nogil=True)
def Sinh_numba(xx):
    return (math.exp(xx) - math.exp(-xx)) / 2


@njit(float64(float64), fastmath=True, cache=True, nogil=True)
def Cosh_numba(xx):
    return (math.exp(xx) + math.exp(-xx)) / 2


@njit(float64(float64[:]), fastmath=True, cache=True, nogil=True)
def MolarMassGERG_numba(x):
    """
    Calculate the molar mass of a gas mixture using GERG-2008 reference equation.

        Inputs:
            x:  Composition (mole fraction)
                This should be an array(of format np.array()) containing the mole fractions of each component.
                Ensure the sum of the compositions in the x array is equal to one.
                The order of the fluids in this array must correspond to MMiGERG.

        return:
            mm: Molar mass (g/mol)
    """
    Mm = 0.0
    for _i in range(NcGERG):
        Mm += x[_i] * MMiGERG[_i]
    return Mm


@njit(types.UniTuple(float64, 2)(float64[:]), cache=True, nogil=True)
def PseudoCriticalPointGERG_numba(x):
    """
    Calculate a pseudo critical point as the mole fraction average of the critical temperatures and volumes.

        Inputs:
            x:   Composition (mole fraction)
                 This should be an array(of format np.array()) containing the mole fractions of each component.
                 Ensure the sum of the compositions in the x array is equal to one.
                 The order of the fluids in this array must correspond to MMiGERG.

        return:
            Tcx: Pseudo-critical temperature
            Dcx: Pseudo-critical density
    """
    Vcx = 0
    Tcx = 0
    Dcx = 0

    for _i in range(NcGERG):
        Tcx = Tcx + x[_i] * Tc[_i]
        Vcx = Vcx + x[_i] / Dc[_i]

    if Vcx > epsilon:
        Dcx = 1 / Vcx

    return Tcx, Dcx


@njit(cache=True, nogil=True)
def ReducingParametersGERG_numba(x):
    """
    Function to calculate reducing parameters in GERG equation of state.

    Inputs:
        x:   Composition (mole fraction)
             This should be an array(of format np.array()) containing the mole fractions of each component.
             Ensure the sum of the compositions in the x array is equal to one.
             The order of the fluids in this array must correspond to MMiGERG.

    return:
        Tr : Reduced temperature
        Dr : Reduced density
    """
    # global xold, Trold, Drold
    Tr, Dr = ReducingParametersGERG_numba_sub(x)
    # for i in range(NcGERG):
    #     xold[i] = x[i]
    # Trold = Tr
    # Drold = Dr
    return Tr, Dr


@njit(types.UniTuple(float64, 2)(float64[:]), cache=True, nogil=True)
def ReducingParametersGERG_numba_sub(x):
    """
    Sub-function to calculate reducing parameters in GERG equation of state.
    Note: Not to be used directly at any other scripts.

    Inputs:
         x:   Composition (mole fraction)
              This should be an array(of format np.array()) containing the mole fractions of each component.
              Ensure the sum of the compositions in the x array is equal to one.
              The order of the fluids in this array must correspond to MMiGERG.

     return:
         Tr : Reduced temperature
         Dr : Reduced density
    """

    # icheck = 0
    # for i in range(NcGERG):
    #     if abs(x[i] - xold[i]) > 0.0000001:
    #         icheck = 1
    #
    # if icheck == 0:
    #     return Trold, Drold

    Dr = 0.0
    Vr = 0.0
    Tr = 0.0
    for i in range(NcGERG):
        if x[i] > epsilon:
            F = 1
            for j in range(i, NcGERG):
                if x[j] > epsilon:
                    xij = F * (x[i] * x[j]) * (x[i] + x[j])
                    Vr = Vr + xij * gvij[i][j] / (bvij[i][j] * x[i] + x[j])
                    Tr = Tr + xij * gtij[i][j] / (btij[i][j] * x[i] + x[j])
                    F = 2
    if Vr > epsilon:
        Dr = 1.0 / Vr

    return Tr, Dr


# def ConvertCompositionGERG_numba(composition):
#     pass
#
#
# # @staticmethod
# @overload(ConvertCompositionGERG_numba)
# def ConvertCompositionGERG_numba(composition):
#     """
#         Converts a dictionary representing gas compositions into a GERG composition list.
#         https://numba.readthedocs.io/en/stable/reference/pysupported.html#typed-dict
#
#         Inputs:
#             composition (dict): A dictionary containing gas species and their compositions.
#
#         return:
#             gerg_composition (list): A list representing the GERG composition of gases.
#     """
#     gerg_composition = np.zeros(21)
#     global gerg_gas_spices
#
#     for gas_spice, composition in composition.items():
#         gerg_composition[np.where(gerg_gas_spices == gas_spice)] = composition
#
#     return np.array(gerg_composition)


@njit(float64(float64, float64, float64[:], boolean, boolean, float64), cache=True, nogil=True)
def CalculateHeatingValue_numba(
    MolarMass, MolarDensity, comp, hhv, per_mass, reference_temp
):
    """
    Calculate the heating value of a gas mixture based on its composition and other properties.

    Inputs:
        MolarMass (float64): The molar mass of the gas mixture.
        MolarDensity (float64): The molar density of the gas mixture.
        comp (np.array): A dictionary representing the composition of the gas mixture.
        hhv (bool): True for Higher Heating Value (HHV) calculation, False for Lower Heating Value (LHV) calculation.
        per_mass (bool): Specifies the parameter for heating value calculation. Options: 'mass' or 'volume'.
        reference_temp (float64): The reference temperature for the heating value calculation. Default is 25 degree Celsius.

    return:
        heating_value (float64): The calculated heating value based on the provided parameters.
    """
    # 298 K
    # dict_enthalpy_mole = {'methane': -74602.416533355,
    #                       'nitrogen': 0.0,
    #                       'carbon dioxide': -393517.79827154,
    #                       'ethane': -83856.2627150042,
    #                       'propane': -103861.117481869,
    #                       'isobutane': -135360.0,
    #                       'n-butane': -125849.99999999999,
    #                       'isopentane': -178400.0,
    #                       'n-pentane': -173500.0,
    #                       'n-hexane': -198490.0,
    #                       'n-heptane': -223910.0,
    #                       'n-hctane': -249730.0,
    #                       'n-nonane': -274700.0,
    #                       'n-decane': -300900.0,
    #                       'hydrogen': 0.0,
    #                       'oxygen': -4.40676212751828,
    #                       'carbon monoxide': -110525.0,
    #                       'water': -241833.418361837,
    #                       'hydrogen sulfide': -20600.0,
    #                       'helium': 0.0,
    #                       'argon': 0.0,
    #                       'carbon': 0.0}
    #                       # 'H': 218000.0,
    #                       # 'O': 249190.0,
    #                       # 'SO2': -296840.0}

    # 273 K
    enthalpy_mole = np.array(
        [
            -75483.51423273719,  # methane
            0.0,  # nitrogen
            -394431.82606764464,  # carbon dioxide
            -83856.2627150042,  # ethane
            -103861.117481869,  # propane
            -135360.0,  # isobutane
            -125849.99999999999,  # n-butane
            -178400.0,  # isopentane
            -173500.0,  # n-pentane
            -198490.0,  # n-hexane
            -223910.0,  # n-heptane
            -249730.0,  # n-octane
            -274700.0,  # n-nonane
            -300900.0,  # n-decane
            0.0,  # hydrogen
            -4.40676212751828,  # oxygen
            -111262.34509634285,  # carbon monoxide
            -242671.7203547155,  # water
            -20600.0,  # hydrogen sulfide
            0.0,  # helium
            0.0,  # argon
            -296840.0,
        ]
    )  # sulfur dioxide

    atom_list = number_of_atoms * comp[:, np.newaxis]
    reactants_atom = np.sum(atom_list, axis=0)

    # products
    n_CO2 = reactants_atom[1]
    n_SO2 = reactants_atom[6]
    n_H2O = reactants_atom[2] / 2
    products_dict = np.array([n_CO2, n_SO2, n_H2O])

    # oxygen for complete combustion
    n_O = (
        n_CO2 * 2 + n_SO2 * 2 + n_H2O * 1
    )  # 2 is number of O atoms in CO2 AND SO2 and 1 is number of O atoms in H2O
    n_O2 = n_O / 2
    reactants_dict = np.copy(comp)
    reactants_dict[15] = n_O2
    # reactants_dict.update({'oxygen': n_O2})

    # LHV calculation
    LHV = (reactants_dict * enthalpy_mole[:-1]).sum() - (
        products_dict * np.take(enthalpy_mole, [2, 21, 17])
    ).sum()

    # enthalpy of formation of water at different temperatures, data obtained using cantera
    supported_temps = [25.0, 15.0]

    if reference_temp == 25.0:
        hw_liq, hw_gas = -285839.09854950657, -241824.62162536496
    elif reference_temp == 15.0:
        hw_liq, hw_gas = -286593.59823661513, -242160.26451330166
    else:
        print(
            f"Unsupported reference temperature: {reference_temp} degree Celsius. Use one of {supported_temps}."
        )

    HHV = LHV + (hw_gas - hw_liq) * products_dict[2]

    if per_mass:
        # returns heating value in J/kg
        if hhv:
            heating_value = HHV / MolarMass * 1e3
        else:
            heating_value = LHV / MolarMass * 1e3
    else:
        # returns heating value in J/m3
        if hhv:
            heating_value = HHV * MolarDensity * 1e3
        else:
            heating_value = LHV * MolarDensity * 1e3

    return heating_value


@njit(float64(float64, float64[:]), cache=True, nogil=True)
def CalculateCO2Emission_numba(MolarMass, x):
    """
    Calculate the heating value of a gas mixture based on its composition and other properties.

    Inputs:
        MolarMass (float64): The molar mass of the gas mixture.
        MolarDensity (float64): The molar density of the gas mixture.
        comp (dict): A dictionary representing the composition of the gas mixture.
        hhv (bool): True for Higher Heating Value (HHV) calculation, False for Lower Heating Value (LHV) calculation.
        parameter (str): Specifies the parameter for heating value calculation. Options: 'mass' or 'volume'.

    return:
        heating_value (float64): The calculated heating value based on the provided parameters.
    """
    global gerg_gas_chemical_composition

    _x = np.ascontiguousarray(x)
    reactant_atoms = np.dot(gerg_gas_chemical_composition, _x)

    # products
    n_CO2 = reactant_atoms[0]

    return n_CO2 * 44.01 / MolarMass


@njit(float64[:](float64, float64, float64[:]), cache=True, nogil=True)
def Alpha0GERG_numba(Temp, MolarDensity, X):
    """
    Private Sub Alpha0GERG(T, D, x, a0)

    Calculate the ideal gas Helmholtz energy and its derivatives with respect to tau and delta.
    This routine is not needed when only P (or Z) is calculated.
    Inputs:
        Temp: Temperature (K)
        MolarDensity: Density (mol/l)
        x: Composition (mole fraction)
    return:
        a0:        a0(0) - Ideal gas Helmholtz energy (all dimensionless [i.e., divided by RT])
                   a0(1) - tau*partial(a0)/partial(tau)
                   a0(2) - tau^2*partial^2(a0)/partial(tau)^2
    """
    T = Temp
    D = MolarDensity
    x = X
    th0T = 0.0
    LogxD = 0.0
    SumHyp0 = 0.0
    SumHyp1 = 0.0
    SumHyp2 = 0.0
    hcn = 0.0
    hsn = 0.0

    a0 = np.array([0.0] * 3)
    if D > epsilon:
        LogD = math.log(D)
    else:
        LogD = math.log(epsilon)
    LogT = math.log(T)
    for i in range(NcGERG):
        if x[i] > epsilon:
            LogxD = LogD + math.log(x[i])
            SumHyp0 = 0
            SumHyp1 = 0
            SumHyp2 = 0
        for j in range(3, 7):
            if th0i[i][j] > epsilon:
                th0T = th0i[i][j] / T
                ep = math.exp(th0T)
                em = 1 / ep
                hsn = (ep - em) / 2
                hcn = (ep + em) / 2
                if j == 3 or j == 5:
                    LogHyp = math.log(abs(hsn))
                    SumHyp0 = SumHyp0 + n0i[i][j] * LogHyp
                    SumHyp1 = SumHyp1 + n0i[i][j] * th0T * hcn / hsn
                    SumHyp2 = SumHyp2 + n0i[i][j] * (th0T / hsn) * (th0T / hsn)
                else:
                    LogHyp = math.log(abs(hcn))
                    SumHyp0 = SumHyp0 - n0i[i][j] * LogHyp
                    SumHyp1 = SumHyp1 - n0i[i][j] * th0T * hsn / hcn
                    SumHyp2 = SumHyp2 + n0i[i][j] * (th0T / hcn) * (th0T / hcn)

        a0[0] += +x[i] * (
            LogxD + n0i[i][0] + n0i[i][1] / T - n0i[i][2] * LogT + SumHyp0
        )
        a0[1] += +x[i] * (n0i[i][2] + n0i[i][1] / T + SumHyp1)
        a0[2] += -x[i] * (n0i[i][2] + SumHyp2)
    return a0


@njit(cache=True, nogil=True)
def tTermsGERG_numba(lntau, x):
    """
    Private Sub tTermsGERG(lntau, x)
    Calculate temperature dependent parts of the GERG-2008 equation of state
    Inputs:
        lntau:  tau = Tr / T => lntau = math.log(tau)
        x:      Composition (mole fraction)
    return:
        null
    """
    # global taup, taupijk
    taup, taupijk = tTermsGERG_numba_sub(lntau, x)
    return taup, taupijk


@njit(cache=True, nogil=True)
def tTermsGERG_numba_sub(lntau, x):
    """
    Calculate temperature-dependent parts of the GERG-2008 equation of state.

    Inputs:
        taup :    List containing calculated temperature-dependent values for taup.
        taupijk : List containing calculated temperature-dependent values for taupijk.
        lntau :   Natural logarithm of tau, a term used in the calculation.
        x :       Composition (mole fraction) of the components.

    returns:
        taup :    Updated taup values.
        taupijk : Updated taupijk values.
    """
    taup0 = np.zeros(12)
    taup = np.zeros((MaxFlds, MaxTrmP))
    taupijk = np.zeros((MaxFlds, MaxTrmM))

    i = 4  # Use propane to get exponents for short form of EOS
    for k in range(
        int(kpol[i] + kexp[i])
    ):  # for (int k = 1; k <= kpol[i] + kexp[i]; ++k)
        taup0[k] = math.exp(toik[i][k] * lntau)
    for i in range(NcGERG):  # for (int i = 1; i <= NcGERG; ++i)
        if x[i] > epsilon:
            if (i > 3) and (i != 14) and (i != 17) and (i != 19):
                for k in range(
                    int(kpol[i] + kexp[i])
                ):  # for (int k = 1; k <= kpol[i] + kexp[i]; ++k)
                    taup[i][k] = noik[i][k] * taup0[k]
            else:
                for k in range(
                    int(kpol[i] + kexp[i])
                ):  # for (int k = 1; k <= kpol[i] + kexp[i]; ++k)
                    taup[i][k] = noik[i][k] * math.exp(toik[i][k] * lntau)

    for i in range(NcGERG):  # for (int i = 1; i <= NcGERG - 1; ++i)
        if x[i] > epsilon:
            for j in range(i + 1, NcGERG):  # for (int j = i + 1; j <= NcGERG; ++j)
                if x[j] > epsilon:
                    mn = int(mNumb[i][j] - 1)
                    if mn >= 0:
                        for k in range(
                            int(kpolij[mn])
                        ):  # for (int k = 1; k <= kpolij[mn]; ++k)
                            taupijk[mn][k] = nijk[mn][k] * math.exp(tijk[mn][k] * lntau)

    return taup, taupijk


# @overload(AlpharGERG_numba)
@njit(float64[:, :](float64, float64[:], int32, int32, float64), fastmath=True, cache=True, nogil=True)
def AlpharGERG_numba(T, x, itau, idelta, D):
    """
    Private Sub AlpharGERG(itau, idelta, T, D, x, ar)

    Calculate dimensionless residual Helmholtz energy and its derivatives with respect to tau and delta.

    :param itau:   Set this to 1 to calculate "ar" derivatives with respect to tau [i.e., ar(1,0), ar(1,1), and ar(2,0)], otherwise set it to 0.
    :param idelta: Currently not used, but kept as an input for future use in specifying the highest density derivative needed.
    :param T:      Temperature (K)
    :param D:      Density (mol/l)
    :param x:      Composition (mole fraction)
    :return:        ar(0,0) - Residual Helmholtz energy (dimensionless, =a/RT)
                    ar(0,1) -     delta*partial  (ar)/partial(delta)
                    ar(0,2) -   delta^2*partial^2(ar)/partial(delta)^2
                    ar(0,3) -   delta^3*partial^3(ar)/partial(delta)^3
                    ar(1,0) -       tau*partial  (ar)/partial(tau)
                    ar(1,1) - tau*delta*partial^2(ar)/partial(tau)/partial(delta)
                    ar(2,0) -     tau^2*partial^2(ar)/partial(tau)^2
    """

    # global Told, Trold, Trold2, Drold
    #
    # global Tr, Dr
    # global taup, taupijk
    # global doik, toik, kpol, kexp, coik, mNumb, fij, kpolij, dijk, tijk, kexpij, cijk, eijk, nijk, gijk
    delp = np.zeros(7, dtype=np.float64)
    Expd = np.zeros(7, dtype=np.float64)
    ar = np.zeros((4, 4), dtype=np.float64)
    # for i in range(4):
    #     for j in range(4):
    #         ar[i][j] = 0.
    # Set up del, tau, log(tau), and the first 7 calculations for del^i
    Tr, Dr = ReducingParametersGERG_numba(x)
    delta = D / Dr
    tau = Tr / T
    lntau = math.log(tau)
    delp[0] = delta
    Expd[0] = math.exp(-delp[0])
    for i in range(1, 7):
        delp[i] = delp[i - 1] * delta
        Expd[i] = math.exp(-delp[i])

    # # If temperature has changed, calculate temperature dependent parts
    # if (abs(T - Told) > 0.0000001) or (abs(Tr - Trold2) > 0.0000001):
    #     tTermsGERG_numba(lntau, x)
    # Told = T
    # Trold2 = Tr

    taup, taupijk = tTermsGERG_numba(lntau, x)

    # Calculate pure fluid contributions
    for i in range(NcGERG):
        if x[i] > epsilon:
            for k in range(int(kpol[i])):
                ndt = x[i] * delp[int(doik[i][k] - 1)] * taup[i][k]
                ndtd = ndt * doik[i][k]
                ar[0][1] += ndtd
                ar[0][2] += ndtd * (doik[i][k] - 1)
                if itau > 0:
                    ndtt = ndt * toik[i][k]
                    ar[0][0] += ndt
                    ar[1][0] += ndtt
                    ar[2][0] += ndtt * (toik[i][k] - 1)
                    ar[1][1] += ndtt * doik[i][k]
                    ar[1][2] += ndtt * doik[i][k] * (doik[i][k] - 1)
                    ar[0][3] += ndtd * (doik[i][k] - 1) * (doik[i][k] - 2)

            for k in range(int(kpol[i]), int(kpol[i] + kexp[i])):
                ndt = (
                    x[i]
                    * delp[int(doik[i][k] - 1)]
                    * taup[i][k]
                    * Expd[int(coik[i][k] - 1)]
                )
                ex = coik[i][k] * delp[int(coik[i][k] - 1)]
                ex2 = doik[i][k] - ex
                ex3 = ex2 * (ex2 - 1)
                ar[0][1] += ndt * ex2
                ar[0][2] += ndt * (ex3 - coik[i][k] * ex)
                if itau > 0:
                    ndtt = ndt * toik[i][k]
                    ar[0][0] += ndt
                    ar[1][0] += ndtt
                    ar[2][0] += ndtt * (toik[i][k] - 1)
                    ar[1][1] += ndtt * ex2
                    ar[1][2] += ndtt * (ex3 - coik[i][k] * ex)
                    ar[0][3] += ndt * (
                        ex3 * (ex2 - 2) - ex * (3 * ex2 - 3 + coik[i][k]) * coik[i][k]
                    )

    # Calculate mixture contributions
    for i in range(NcGERG - 1):  # for (int i = 1; i <= NcGERG - 1; ++i)
        if x[i] > epsilon:
            for j in range(i + 1, NcGERG):  # for (int j = i + 1; j <= NcGERG; ++j)
                if x[j] > epsilon:
                    mn = int(mNumb[i][j] - 1)
                    if mn >= 0:
                        xijf = x[i] * x[j] * fij[i][j]
                        for k in range(
                            int(kpolij[mn])
                        ):  # for (int k = 1; k <= kpolij[mn]; ++k)
                            ndt = xijf * delp[int(dijk[mn][k] - 1)] * taupijk[mn][k]
                            ndtd = ndt * dijk[mn][k]
                            ar[0][1] += ndtd
                            ar[0][2] += ndtd * (dijk[mn][k] - 1)
                            if itau > 0:
                                ndtt = ndt * tijk[mn][k]
                                ar[0][0] += ndt
                                ar[1][0] += ndtt
                                ar[2][0] += ndtt * (tijk[mn][k] - 1)
                                ar[1][1] += ndtt * dijk[mn][k]
                                ar[1][2] += ndtt * dijk[mn][k] * (dijk[mn][k] - 1)
                                ar[0][3] += ndtd * (dijk[mn][k] - 1) * (dijk[mn][k] - 2)

                        for k in range(
                            int(kpolij[mn]), int(kpolij[mn] + kexpij[mn])
                        ):  # for (int k = 1 + kpolij[mn]; k <= kpolij[mn] + kexpij[mn]; ++k)
                            cij0 = cijk[mn][k] * delp[1]
                            eij0 = eijk[mn][k] * delta
                            ndt = (
                                xijf
                                * nijk[mn][k]
                                * delp[int(dijk[mn][k] - 1)]
                                * math.exp(
                                    cij0 + eij0 + gijk[mn][k] + tijk[mn][k] * lntau
                                )
                            )
                            ex = dijk[mn][k] + 2 * cij0 + eij0
                            ex2 = ex * ex - dijk[mn][k] + 2 * cij0
                            ar[0][1] += ndt * ex
                            ar[0][2] += ndt * ex2
                            if itau > 0:
                                ndtt = ndt * tijk[mn][k]
                                ar[0][0] += ndt
                                ar[1][0] += ndtt
                                ar[2][0] += ndtt * (tijk[mn][k] - 1)
                                ar[1][1] += ndtt * ex
                                ar[1][2] += ndtt * ex2
                                ar[0][3] += ndt * (
                                    ex * (ex2 - 2 * (dijk[mn][k] - 2 * cij0))
                                    + 2 * dijk[mn][k]
                                )
    return ar


@njit(types.UniTuple(float64, 3)(float64, float64, float64[:]), cache=True, nogil=True)
def PressureGERG_numba(T, D, x):
    """
    Sub PressureGERG(T, D, x, P, Z)

    Calculate pressure as a function of temperature and density.  The derivative d(P)/d(D) is also calculated
    for use in the iterative DensityGERG subroutine (and is only returned as a common variable).

    Inputs:
        ar: Dimensionless residual Helmholtz energy and its derivatives with respect to tau and delta.
        T: Temperature (K)
        D: Density (mol/l)

    return:
        P:        Pressure (kPa)
        Z:        Compressibility factor
        dPdDsave: d(P)/d(D) [kPa/(mol/l)] (at constant temperature)
                  This variable is cached in the common variables for use in the iterative density solver, but not returned as an argument.
    """
    ar = AlpharGERG_numba(T=T, x=x, itau=0, idelta=0, D=D)

    Z = 1 + ar[0][1]
    P = D * RGERG * T * Z
    dPdDsave = RGERG * T * (1 + 2 * ar[0][1] + ar[0][2])
    return P, Z, dPdDsave


# @njit(types.Tuple((float64, types.unicode_type, float64))(float64[:, :], float64, float64, float64[:], int64))
# Tha above did not work since there is an empty return happening.
@njit(cache=True, nogil=True)
def DensityGERG_numba(P, T, x, iFlag=0):
    """
    Sub DensityGERG(iFlag, T, P, x, D, ierr, herr)

    Calculate density as a function of temperature and pressure.  This is an iterative routine that calls PressureGERG
    to find the correct state point.  Generally only 6 iterations at most are required.
    If the iteration fails to converge, the ideal gas density and an error message are returned.
    No checks are made to determine the phase boundary, which would have guaranteed that the output is in the gas phase (or liquid phase when iFlag=2).
    It is up to the user to locate the phase boundary, and thus identify the phase of the T and P inputs.
    If the state point is 2-phase, the output density will represent a metastable state.

                    (An initial guess for the density can be sent in D as the negative of the guess for roots that are in the liquid phase instead of using iFlag=2)
    Inputs:
        ar:      Dimensionless residual Helmholtz energy and its derivatives with respect to tau and delta.
        T:       Temperature (K)
        P:       Pressure (kPa)
        x :      Composition (mole fraction) of the components.
        iFlag:   Set to 0 for strict pressure solver in the gas phase without checks (fastest mode, but output state may not be stable single phase)
                 Set to 1 to make checks for possible 2-phase states (result may still not be stable single phase, but many unstable states will be identified)
                 Set to 2 to search for liquid phase (and make the same checks when iFlag=1)

    return:
        D:       Density (mol/l)
                 For the liquid phase, an initial value can be sent to the routine to avoid
                 a solution in the metastable or gas phases.
                 The initial value should be sent as a negative number.
        ierr:    Error number (0 indicates no error)
        herr:    Error message if ierr is not equal to zero
    """

    D = 0  # initial estimate of the density

    dPdD = 0.0
    d2PdTD = 0.0
    Cv = 0.0
    Cp = 0.0
    W = 0.0
    PP = 0.0

    ierr = 0
    herr = ""
    nFail = 0
    iFail = 0
    if P < epsilon:
        D = 0
        return ierr, herr, D
    tolr = 0.0000001
    Tcx, Dcx = PseudoCriticalPointGERG_numba(x)

    if D > -epsilon:
        D = P / RGERG / T  # Ideal gas estimate for vapor phase
        if iFlag == 2:
            D = Dcx * 3  # Initial estimate for liquid phase

    else:
        D = abs(D)  # If D<0, then use as initial estimate

    plog = math.log(P)
    vlog = -math.log(D)
    for it in range(1, 51):
        if (
            (vlog < -7)
            or (vlog > 100)
            or (it == 20)
            or (it == 30)
            or (it == 40)
            or (iFail == 1)
        ):
            # Current state is bad or iteration is taking too long.  Restart with completely different initial state
            iFail = 0
            if nFail > 2:
                # Iteration failed (above loop did not find a solution or checks made below indicate possible 2-phase state)
                ierr = 1
                herr = "Calculation failed to converge in GERG method, ideal gas density returned."
                D = P / RGERG / T
            nFail += 1
            if nFail == 1:
                D = (
                    Dcx * 3
                )  # If vapor phase search fails, look for root in liquid region
            elif nFail == 2:
                D = (
                    Dcx * 2.5
                )  # If liquid phase search fails, look for root between liquid and critical regions
            elif nFail == 3:
                D = Dcx * 2  # If search fails, look for root in critical region

            vlog = -math.log(D)
        D = math.exp(-vlog)
        P2, Z, dPdDsave = PressureGERG_numba(T, D, x)
        if (dPdDsave < epsilon) or (P2 < epsilon):
            # Current state is 2-phase, try locating a different state that is single phase
            vinc = 0.1
            if D > Dcx:
                vinc = -0.1
            if it > 5:
                vinc = vinc / 2
            if (it > 10) and (it < 20):
                vinc = vinc / 5
            vlog += vinc
        else:
            # Find the next density with a first order Newton's type iterative scheme, with
            # log(P) as the known variable and log(v) as the unknown property.
            # See AGA 8 publication for further information.
            dpdlv = -D * dPdDsave  # d(p)/d[log(v)]
            vdiff = (math.log(P2) - plog) * P2 / dpdlv
            vlog += -vdiff
            if abs(vdiff) < tolr:
                # Check to see if state is possibly 2-phase, and if so restart
                if dPdDsave < 0:
                    iFail = 1
                else:
                    D = math.exp(-vlog)

                    # If requested, check to see if point is possibly 2-phase
                    if iFlag > 0:
                        # PropertiesGERG_numba()
                        if ((PP <= 0) or (dPdD <= 0) or (d2PdTD <= 0)) or (
                            (Cv <= 0) or (Cp <= 0) or (W <= 0)
                        ):
                            # Iteration failed (above loop did find a solution or checks made below indicate possible 2-phase state)
                            ierr = 1
                            herr = "Calculation failed to converge in GERG method, ideal gas density returned."
                            D = P / RGERG / T
                        return ierr, herr, D
                    return ierr, herr, D  # Iteration converged
    # Iteration failed (above loop did not find a solution or checks made below indicate possible 2-phase state)
    ierr = 1
    herr = "Calculation failed to converge in GERG method, ideal gas density returned."
    D = P / RGERG / T
    return ierr, herr, D


# def AlpharGERG_numba(T, x, itau, idelta, D):
#     pass


@njit(types.UniTuple(float64, 20)(float64, float64, float64[:]), cache=True, nogil=True)
def PropertiesGERG_numba(T, P, x):
    """
    Sub PropertiesGERG(T, D, x, P, Z, dPdD, d2PdD2, d2PdTD, dPdT, U, H, S, Cv, Cp, W, G, JT, Kappa, A)

    Calculate thermodynamic properties as a function of temperature and density.  Calls are made to the subroutines
    ReducingParametersGERG, IdealGERG, and ResidualGERG.  If the density is not known, call subroutine DENSITY first
    with the known values of pressure and temperature.

    Inputs:
        T:      Temperature (K)
        P:      Pressure (kPa)
        x:      Composition (mole fraction)
        ar:     Dimensionless residual Helmholtz energy and its derivatives with respect to tau and delta.

    Returns:
        Tuple of thermodynamic properties:
            molar_mass:        Molar mass (g/mol)
            D:                 Density (mol/l)
            Z:                 Compressibility factor
            dPdD:              First derivative of pressure with respect to density at constant temperature [kPa/(mol/l)]
            d2PdD2:            Second derivative of pressure with respect to density at constant temperature [kPa/(mol/l)^2]
            dPdT:              First derivative of pressure with respect to temperature at constant density (kPa/K)
            U:                 Internal energy (J/mol)
            H:                 Enthalpy (J/mol)
            S:                 Entropy [J/(mol-K)]
            Cv:                Isochoric heat capacity [J/(mol-K)]
            Cp:                Isobaric heat capacity [J/(mol-K)]
            Cv_molar:           Specific heat capacity at constant volume [J/(kg-K)]
            Cp_molar:           Specific heat capacity at constant pressure [J/(kg-K)]
            W:                 Speed of sound (m/s)
            G:                 Gibbs energy (J/mol)
            JT:                Joule-Thomson coefficient (K/Pa)
            Kappa:             Isentropic exponent
            Vm:                Molar volume (m^3/mol)
            molar_mass_ratio:  Ratio of molar mass to air molar mass
            R_specific:        Specific gas constant [J/(kg-K)]
    """
    # Calculate molar mass
    molar_mass = MolarMassGERG_numba(x)

    # Calculate the density
    ierr, herr, D = DensityGERG_numba(P, T, x, iFlag=0)
    MolarDensity = D

    # Calculate the ideal gas Helmholtz energy, and its first and second derivatives with respect to temperature.
    a0 = Alpha0GERG_numba(T, D, x)
    # Calculate the real gas Helmholtz energy, and its derivatives with respect to temperature and/or density.
    ar = AlpharGERG_numba(T, x, itau=1, idelta=0, D=D)

    R = RGERG
    RT = R * T
    Z = 1 + ar[0][1]
    P = D * RT * Z
    dPdD = RT * (1 + 2 * ar[0][1] + ar[0][2])
    dPdT = D * R * (1 + ar[0][1] - ar[1][1])
    d2PdTD = R * (1 + 2 * ar[0][1] + ar[0][2] - 2 * ar[1][1] - ar[1][2])
    A = RT * (a0[0] + ar[0][0])
    G = RT * (1 + ar[0][1] + a0[0] + ar[0][0])
    U = RT * (a0[1] + ar[1][0])
    H = RT * (1 + ar[0][1] + a0[1] + ar[1][0])
    S = R * (a0[1] + ar[1][0] - a0[0] - ar[0][0])
    Cv = -R * (a0[2] + ar[2][0])
    if D > epsilon:
        Cp = Cv + T * (dPdT / D) * (dPdT / D) / dPdD
        d2PdD2 = RT * (2 * ar[0][1] + 4 * ar[0][2] + ar[0][3]) / D
        JT = (
            (T / D * dPdT / dPdD - 1) / Cp / D
        )  #  '=(dB/dT*T-B)/Cp for an ideal gas, but dB/dT is not known
    else:
        Cp = Cv + R
        d2PdD2 = 0
        JT = 1e20
    W = 1000 * Cp / Cv * dPdD / molar_mass
    if W < 0:
        W = 0
    W = math.sqrt(W)
    Kappa = pow(W, 2) * molar_mass / (RT * 1000 * Z)

    return (
        molar_mass,
        D,
        Z,
        dPdD,
        d2PdD2,
        dPdT,
        U,
        H,
        S,
        Cv,
        Cp,
        Cv * 1000 / molar_mass,
        Cp * 1000 / molar_mass,
        W,
        G,
        JT / 1e3,
        Kappa,
        molar_mass * P / Z / RT,
        molar_mass / air_molar_mass,
        R / molar_mass,
    )


# Assuming necessary helper functions like MolarMassGERG_numba, DensityGERG_numba, etc., are available
@njit(float64(float64[:]), cache=True, nogil=True)
def molar_mass_numba(x):
    return MolarMassGERG_numba(x)


@njit(float64(float64, float64, float64[:]), cache=True, nogil=True)
def density_numba(P, T, x):
    ierr, herr, D = DensityGERG_numba(P, T, x, iFlag=0)
    return D


@njit(types.Tuple((float64[:], float64[:, :]))(float64, float64, float64[:]), cache=True, nogil=True)
def common_properties_numba(T, D, x):
    a0 = Alpha0GERG_numba(T, D, x)
    ar = AlpharGERG_numba(T, x, itau=1, idelta=0, D=D)
    return a0, ar


@njit(float64(float64[:, :]), cache=True, nogil=True)
def compressibility_factor_numba(ar):
    return 1 + ar[0][1]


@njit(float64(float64, float64, float64, float64), cache=True, nogil=True)
def pressure_numba(D, R, T, Z):
    return D * R * T * Z


@njit(float64(float64, float64, float64[:, :]), cache=True, nogil=True)
def first_derivative_pressure_density_numba(R, T, ar):
    return R * T * (1 + 2 * ar[0][1] + ar[0][2])


@njit(float64(float64, float64, float64[:, :]), cache=True, nogil=True)
def first_derivative_pressure_temperature_numba(R, D, ar):
    return D * R * (1 + ar[0][1] - ar[1][1])


@njit(float64(float64, float64, float64, float64[:, :]), cache=True, nogil=True)
def second_derivative_pressure_temperature_density_numba(R, T, D, ar):
    return R * T * (2 * ar[0][1] + 4 * ar[0][2] + ar[0][3]) / D
    # return R * (1 + 2 * ar[0][1] + ar[0][2] - 2 * ar[1][1] - ar[1][2])


@njit(float64(float64, float64, float64[:], float64[:, :]), cache=True, nogil=True)
def internal_energy_numba(R, T, a0, ar):
    return R * T * (a0[1] + ar[1][0])


@njit(float64(float64, float64, float64[:], float64[:, :]), cache=True, nogil=True)
def enthalpy_numba(R, T, a0, ar):
    return R * T * (1 + ar[0][1] + a0[1] + ar[1][0])


@njit(float64(float64, float64[:], float64[:, :]), cache=True, nogil=True)
def entropy_numba(R, a0, ar):
    return R * (a0[1] + ar[1][0] - a0[0] - ar[0][0])


@njit(float64(float64, float64[:], float64[:, :]), cache=True, nogil=True)
def isochoric_heat_capacity_numba(R, a0, ar):
    return -R * (a0[2] + ar[2][0])


@njit(float64(float64, float64, float64, float64[:], float64[:, :]), cache=True, nogil=True)
def isobaric_heat_capacity_numba(T, D, R, a0, ar):
    Cv = -R * (a0[2] + ar[2][0])
    dPdT = D * R * (1 + ar[0][1] - ar[1][1])
    dPdD = R * T * (1 + 2 * ar[0][1] + ar[0][2])
    if D > epsilon:
        return Cv + T * (dPdT / D) * (dPdT / D) / dPdD
    else:
        return Cv + R


@njit(float64(float64, float64, float64, float64[:], float64[:, :], float64[:]), cache=True, nogil=True)
def speed_of_sound_numba(T, D, R, a0, ar, x):
    Cp = isobaric_heat_capacity_numba(T, D, R, a0, ar)
    Cv = isochoric_heat_capacity_numba(R, a0, ar)
    dPdD = first_derivative_pressure_density_numba(R, T, ar)
    molar_mass = molar_mass_numba(x)
    W = 1000 * Cp / Cv * dPdD / molar_mass
    if W < 0:
        W = 0
    return math.sqrt(W)


@njit(float64(float64, float64, float64[:], float64[:, :]), cache=True, nogil=True)
def gibbs_energy_numba(R, T, a0, ar):
    return R * T * (1 + ar[0][1] + a0[0] + ar[0][0])


@njit(float64(float64, float64, float64, float64, float64[:], float64[:, :]), cache=True, nogil=True)
def joule_thomson_coefficient_numba(T, D, epsilon, R, a0, ar):
    Cp = isobaric_heat_capacity_numba(T, D, R, a0, ar)
    dPdT = first_derivative_pressure_temperature_numba(R, D, ar)
    dPdD = first_derivative_pressure_density_numba(R, T, ar)
    if D > epsilon:
        return (T / D * dPdT / dPdD - 1) / Cp / D / 1e3
    else:
        return 1e20


@njit(float64(float64, float64, float64, float64[:], float64[:, :], float64[:]), cache=True, nogil=True)
def isentropic_exponent_numba(T, D, R, a0, ar, x):
    W = speed_of_sound_numba(T, D, R, a0, ar, x)
    molar_mass = molar_mass_numba(x)
    Z = compressibility_factor_numba(ar)
    return W**2 * molar_mass / (R * T * 1000 * Z)


@njit(float64(float64, float64, float64[:]), cache=True, nogil=True)
def molar_volume_numba(T, P, x):
    D = density_numba(P, T, x)
    molar_mass = molar_mass_numba(x)
    a0, ar = common_properties_numba(T, D, x)
    Z = compressibility_factor_numba(ar)
    R = RGERG
    return molar_mass * P / (Z * R * T)


@njit(float64(float64[:]), cache=True, nogil=True)
def molar_mass_ratio_numba(x):
    molar_mass = molar_mass_numba(x)
    return molar_mass / air_molar_mass


@njit(float64(float64[:]), cache=True, nogil=True)
def specific_gas_constant_numba(x):
    R = RGERG
    molar_mass = molar_mass_numba(x)
    return R / molar_mass
