from pint import UnitRegistry
ureg=UnitRegistry()

def si(x):
    return ureg(x).to_base_units().magnitude


# VO2 material parameters to match Supplementary Table 1 of Shukla
shukla_vo2_params={
    "rho_m":si("1e-4 ohm cm"),
    "rho_i":si("2 ohm cm"),
    "J_MIT":si("1e6 A/cm^2"),
    "J_IMT":si("1e5 A/cm^2"),
    "V_met": 0
}

# Resistivities from Schlag and Scherber, current densities "made up"
mixed_vo2_params={
    "rho_m":si("3e-4 ohm cm"),
    "rho_i":si("30 ohm cm"),
    "J_MIT":si("1e6 A/cm^2"),
    "J_IMT":si(".5e4 A/cm^2"),
    "V_met": 0
}
