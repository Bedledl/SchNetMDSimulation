import os

from schnetpack.md.simulation_hooks import LangevinThermostat
from constants import WORKDIR
from ethanol_simulation import EthanolSimulation

thermostats = [
#    LangevinThermostat(10, 10),
    LangevinThermostat(300, 10)
#    LangevinThermostat(500, 10),
#    LangevinThermostat(1000, 10)
]

log_files = [
#    os.path.join(WORKDIR, "log_thermostat_10"),
    os.path.join(WORKDIR, "log_thermostat_300")
#    os.path.join(WORKDIR, "log_thermostat_500"),
#    os.path.join(WORKDIR, "log_thermostat_1000")
]

EthanolSimulation(thermostats, log_files, "ipu", 100000)
