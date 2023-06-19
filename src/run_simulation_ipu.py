import datetime
import os

import torch

from schnetpack.md.simulation_hooks import LangevinThermostat
from constants import WORKDIR
from ethanol_simulation import EthanolSimulation
from SchNetPackCalcIpu import SchNetPackCalcIpu

thermostats = [
    LangevinThermostat(300, 10)
]

log_files = [
    os.path.join(WORKDIR, "log_thermostat_300")
]

start_ipu = datetime.datetime.now()

EthanolSimulation(thermostats, log_files, torch.device("ipu"), 1000, SchNetPackCalcIpu)

end_ipu = datetime.datetime.now()
print(f"The whole simulation(with compilation) took: {end_ipu - start_ipu} seconds")
