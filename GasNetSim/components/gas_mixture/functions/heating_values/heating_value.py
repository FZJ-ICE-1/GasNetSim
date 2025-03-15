from pathlib import Path
import json
import numpy as np
from numba import njit, float64, types, int32, boolean

from ...GERG2008.gerg2008 import number_of_atoms

REF_TEMP_COMBUSTION = [0.0, 15.0, 25.0]  # Implemented reference temperatures

_enthalpy_cache = {}


def load_enthalpy_values():
    """
    Load enthalpy values from a JSON file for a specific reference temperature.

    Returns:
        dict: Dictionary mapping temperature to arrays of enthalpy values
    """
    filepath_enthalpy = Path(__file__).parent / "absolute_enthalpies.json"

    if not filepath_enthalpy.exists():
        raise FileNotFoundError(f"ERROR: Enthalpy data file not found at {filepath_enthalpy}")

    try:
        with open(filepath_enthalpy, 'r') as f:
            enthalpy_data = json.load(f)
    except json.JSONDecodeError:
        print(f"ERROR: Invalid JSON format in enthalpy file: {filepath_enthalpy}")
        print("Please check the file format and ensure it contains valid JSON.")
        return {}
    except Exception as e:
        print(f"ERROR: Failed to read enthalpy file: {e}")
        return {}

    # Define the order of compounds to match the original function
    compounds = [
        "methane", "nitrogen", "carbon dioxide", "ethane", "propane",
        "isobutane", "n-butane", "isopentane", "n-pentane", "n-hexane",
        "n-heptane", "n-octane", "n-nonane", "n-decane", "hydrogen",
        "oxygen", "carbon monoxide", "water (g)", "hydrogen sulfide",
        "helium", "argon", "sulfur dioxide", "water (l)"
    ]

    # Process each temperature
    result = {}
    for temp in REF_TEMP_COMBUSTION:
        temp_str = str(temp)
        cache_key = (filepath_enthalpy, temp)

        # Skip if already cached
        if cache_key in _enthalpy_cache:
            result[temp] = _enthalpy_cache[cache_key]
            continue

        # Extract enthalpy values for this temperature
        enthalpy_values = []
        for compound in compounds:
            if compound in enthalpy_data and temp_str in enthalpy_data[compound]:
                enthalpy_values.append(enthalpy_data[compound][temp_str])
            else:
                # Use a default value or handle missing data
                enthalpy_values.append(0.0)
                print(f"Warning: No enthalpy data for {compound} at {temp_str}C")

        # Cache the result
        enthalpy_array = np.array(enthalpy_values)
        _enthalpy_cache[cache_key] = enthalpy_array
        result[temp] = enthalpy_array

        # Also cache liquid water enthalpy for HHV calculations
        if "water (l)" in enthalpy_data and temp_str in enthalpy_data["water (l)"]:
            _enthalpy_cache[("water (l)", temp)] = enthalpy_data["water (l)"][temp_str]

    return result


# Load enthalpy values outside of the JIT function
_enthalpy_values = load_enthalpy_values()


# The actual JIT-compiled implementation
@njit
def _CalculateHeatingValue_numba_impl(
        MolarMass, MolarDensity, comp, hhv, per_mass, enthalpy_mole
):
    """
    Numba-accelerated implementation of heating value calculation.

    This is the core calculation logic, compiled with Numba for performance.
    """
    # Calculate atomic composition
    atom_list = number_of_atoms * comp[:, np.newaxis]
    reactants_atom = np.sum(atom_list, axis=0)

    # Calculate products
    n_CO2 = reactants_atom[1]
    n_SO2 = reactants_atom[6]
    n_H2O = reactants_atom[2] / 2

    # Array of products: [CO2, SO2, H2O]
    products = np.array([n_CO2, n_SO2, n_H2O], dtype=np.float64)

    # Calculate oxygen needed for complete combustion
    n_O = n_CO2 * 2 + n_SO2 * 2 + n_H2O
    n_O2 = n_O / 2

    # Copy the composition array and add oxygen
    reactants = np.copy(comp)
    reactants[15] = n_O2  # Index 15 corresponds to oxygen

    # Calculate Lower Heating Value (LHV)
    # Sum of (reactants * enthalpies) minus sum of (products * enthalpies)
    reactants_enthalpy_sum = 0.0
    for i in range(len(reactants)):
        if i < len(enthalpy_mole) - 1:  # Skip the last enthalpy value (water liquid)
            reactants_enthalpy_sum += reactants[i] * enthalpy_mole[i]

    # Calculate enthalpy of products
    # CO2 (index 2), SO2 (index 21), H2O gas (index 17)
    products_enthalpy_sum = products[0] * enthalpy_mole[2] + products[1] * enthalpy_mole[21] + products[2] * \
                            enthalpy_mole[17]

    LHV = reactants_enthalpy_sum - products_enthalpy_sum

    # Higher Heating Value includes the latent heat of water vapor condensation
    hw_liq, hw_gas = enthalpy_mole[22], enthalpy_mole[17]  # Water liquid, water gas
    HHV = LHV + (hw_gas - hw_liq) * products[2]

    # Return either HHV or LHV, per mass or per volume
    if per_mass:
        # J/kg
        if hhv:
            return HHV / MolarMass * 1e3
        else:
            return LHV / MolarMass * 1e3
    else:
        # J/m³
        if hhv:
            return HHV * MolarDensity * 1e3
        else:
            return LHV * MolarDensity * 1e3


# Original function name maintained for compatibility
def CalculateHeatingValue_numba(
        MolarMass, MolarDensity, comp, hhv=True, per_mass=True, reference_temp=25.0
):
    """
    Calculate the heating value of a gas mixture based on its composition and other properties.

    Inputs:
        MolarMass (float64): The molar mass of the gas mixture.
        MolarDensity (float64): The molar density of the gas mixture.
        comp (np.array): A numpy array representing the composition of the gas mixture.
        hhv (bool): True for Higher Heating Value (HHV) calculation, False for Lower Heating Value (LHV) calculation.
        per_mass (bool): Specifies the parameter for heating value calculation. Options: 'mass' or 'volume'.
        reference_temp (float64): The reference temperature for the heating value calculation. Default is 25 degree Celsius.

    return:
        heating_value (float64): The calculated heating value based on the provided parameters.
    """
    # Validate reference temperature
    if reference_temp not in REF_TEMP_COMBUSTION:
        raise ValueError(
            f"Unsupported reference temperature: {reference_temp} degree Celsius. "
            f"Use one of {REF_TEMP_COMBUSTION}."
        )

    # Get the enthalpy array for the specified reference temperature
    try:
        enthalpy_array = _enthalpy_values[reference_temp]
    except KeyError:
        raise ValueError(f"No enthalpy data available for reference temperature {reference_temp}")

    # Ensure comp is a numpy array with the right type and shape
    if not isinstance(comp, np.ndarray):
        comp = np.asarray(comp, dtype=np.float64)
    elif comp.dtype != np.float64:
        comp = comp.astype(np.float64)

    # Call the implementation with properly typed arguments
    MolarMass_f = float(MolarMass)
    MolarDensity_f = float(MolarDensity)
    hhv_b = bool(hhv)
    per_mass_b = bool(per_mass)

    return _CalculateHeatingValue_numba_impl(
        MolarMass_f,
        MolarDensity_f,
        comp,
        hhv_b,
        per_mass_b,
        enthalpy_array
    )