import os

from schnetpack.md.data import HDF5Loader
import numpy as np
import matplotlib.pyplot as plt
from schnetpack import units as spk_units, properties
from src.constants import WORKDIR


def show_logfile(log_file: str):
    data = HDF5Loader(log_file)

    # Get the energy logged via PropertiesStream
    energies_calculator = data.get_property(properties.energy, atomistic=False)
    # Get potential energies stored in the MD system
    energies_system = data.get_potential_energy()

    # Check the overall shape
    print("Shape:", energies_system.shape)

    # Get the time axis
    time_axis = np.arange(data.entries) * data.time_step / spk_units.fs  # in fs

    # Convert the system potential energy from internal units (kJ/mol) to kcal/mol
    energies_system *= spk_units.convert_units("kJ/mol", "kcal/mol")

    # Plot the energies
    plt.figure()
    plt.plot(time_axis, energies_system, label="E$_\mathrm{pot}$ (System)")
    plt.plot(time_axis, energies_calculator, label="E$_\mathrm{pot}$ (Logger)", ls="--")
    plt.ylabel("E [kcal/mol]")
    plt.xlabel("t [fs]")
    plt.xlim(98, 10)
    plt.legend()
    plt.tight_layout()
    plt.show()

log_files = [
    os.path.join(WORKDIR, "log_thermostat_10"),
    os.path.join(WORKDIR, "log_thermostat_300"),
    os.path.join(WORKDIR, "log_thermostat_500"),
    os.path.join(WORKDIR, "log_thermostat_1000")
]

for lf in log_files:
    show_logfile(lf)
