import os.path

import torch

from typing import List

from schnetpack.md.simulation_hooks import LangevinThermostat
from simulation import MDSimulations
from src.constants import WORKDIR


class EthanolSimulation:
    '''
    This class initializes and starts multiple simulations of the Ethanol-MD Example
    from the SchNetPack MD Tutorial with different configured Langevin-Thermostats.
    '''
    def __init__(self,
                 thermostats: List[LangevinThermostat],
                 log_files: List[str],
                 device: torch.device,
                 steps
                 ):

        simulation = MDSimulations(
            device,
            "/home/betti/masterarbeit/schnetpack/tests/testdata/md_ethanol.model",
            "/home/betti/masterarbeit/schnetpack/tests/testdata/md_ethanol.xyz",
            WORKDIR,
            1,
            300,
            0.5,
            5.0,
            2.0,
            torch.float32,
            thermostats,
            log_files
        )
        print("Start simulation")
        simulation.start_simulation(steps)
        print("finnish simulation")
