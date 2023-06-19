import datetime
import os

from schnetpack.md.calculators import SchNetPackCalculator

import torch

from schnetpack.md.simulation_hooks import LangevinThermostat
from constants import WORKDIR
from ethanol_simulation import EthanolSimulation

thermostats = [
    LangevinThermostat(300, 10)
]

log_files = [
    os.path.join(WORKDIR, "log_thermostat_300")
]

ethanol_simulation = EthanolSimulation(
    thermostats, log_files, torch.device("cpu"), SchNetPackCalculator)
ethanol_simulation.start_simulation(10)
