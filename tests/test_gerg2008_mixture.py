#   #!/usr/bin/env python
#   -*- coding: utf-8 -*-
#   ******************************************************************************
#     Copyright (c) 2024.
#     Developed by Yifei Lu
#     Last change on 9/4/24, 10:15 AM
#     Last change by yifei
#    *****************************************************************************
from scipy.constants import bar

from GasNetSim.components.gas_mixture.GERG2008 import *
from GasNetSim.components.gas_mixture.GERG2008 import convert_to_gerg2008_composition
from GasNetSim.components.gas_mixture.typical_mixture_composition import NATURAL_GAS

# test over natural gas
gerg_ng_composition = convert_to_gerg2008_composition(NATURAL_GAS)

gas_mixture = GasMixtureGERG2008(P_Pa=1 * bar, T_K=298, composition=gerg_ng_composition)
print("Molar Mass [g/mol] = " + str(gas_mixture.MolarMass))
print("Z [-] = " + str(gas_mixture.Z))
print("Isochoric Heat Capacity [J/mol-K] = " + str(gas_mixture.Cv))
print("Isobaric Heat Capacity [J/mol-K] = " + str(gas_mixture.Cp))
print("Joule-Thomson coefficient [K/kPa] = " + str(gas_mixture.JT))

# test over methane and hydrogen blending
z_list = []
h_list = []

METHANE_HYDROGEN_MIXTURE = OrderedDict([("methane", 1.0), ("hydrogen", 0.0)])

gas_comp = convert_to_gerg2008_composition(METHANE_HYDROGEN_MIXTURE)
h_list.append(METHANE_HYDROGEN_MIXTURE["hydrogen"])
gas_mixture = GasMixtureGERG2008(P_Pa=1 * bar, T_K=298, composition=gas_comp)

z_list.append(gas_mixture.Z)

print("hydrogen content [%]      Z [-] ")
print(
    "{:7.3f}      {:7.3f}".format(
        METHANE_HYDROGEN_MIXTURE["hydrogen"] * 100, gas_mixture.Z
    )
)

while METHANE_HYDROGEN_MIXTURE["methane"] >= 0:

    METHANE_HYDROGEN_MIXTURE["methane"] -= 0.01
    METHANE_HYDROGEN_MIXTURE["hydrogen"] += 0.01
    h_list.append(METHANE_HYDROGEN_MIXTURE["hydrogen"])

    gas_comp = convert_to_gerg2008_composition(METHANE_HYDROGEN_MIXTURE)

    gas_mixture = GasMixtureGERG2008(P_Pa=1 * bar, T_K=298, composition=gas_comp)

    z_list.append(gas_mixture.Z)

    print(
        "{:7.3f}      {:7.3f}".format(
            METHANE_HYDROGEN_MIXTURE["hydrogen"] * 100, gas_mixture.Z
        )
    )
