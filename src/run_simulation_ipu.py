import datetime
import os

import torch

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

start_ipu = datetime.datetime.now()

EthanolSimulation(thermostats, log_files, torch.device("ipu"), 100000, True)

end_ipu = datetime.datetime.now()
print(f"The whole simulation took: {end_ipu - start_ipu} seconds")
#
#start_ipu_2 = datetime.datetime.now()
#
#EthanolSimulation(thermostats, log_files, torch.device("ipu"), 100000, False)
#
#end_ipu_2 = datetime.datetime.now()
#
#start_cpu = datetime.datetime.now()
#
#EthanolSimulation(thermostats, log_files, torch.device("cpu"), 100000, True)
#
#end_cpu = datetime.datetime.now()
#
#start_cpu_2 = datetime.datetime.now()
#
#EthanolSimulation(thermostats, log_files, torch.device("cou"), 100000, False)
#
#end_cpu_2 = datetime.datetime.now()
#
#
#result = f"--------------------" \
#         f"Dauer Cmd: 'EthanolSimulation(thermostats, log_files, \"cpu\", 100000, True)'" \
#         f"" \
#         f"{start_cpu - end_cpu}" \
#         f"" \
#         f"Dauer Cmd: 'EthanolSimulation(thermostats, log_files, \"ipu\", 100000, True)'" \
#         f"" \
#         f"{start_ipu - end_ipu}" \
#         f"" \
#         f"Dauer Cmd: 'EthanolSimulation(thermostats, log_files, \"cpu\", 100000, False)'" \
#         f"" \
#         f"{start_cpu_2 - end_cpu_2}" \
#         f"" \
#         f"Dauer Cmd: 'EthanolSimulation(thermostats, log_files, \"ipu\", 100000, False)'" \
#         f"" \
#         f"{start_ipu_2 - end_ipu_2}" \
#         f""
#