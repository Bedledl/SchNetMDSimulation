import torch

from typing import List, Type

from schnetpack.md.calculators import SchNetPackCalculator
from schnetpack.md.simulation_hooks import LangevinThermostat
from simulation import MDSimulations
from constants import WORKDIR


class EthanolSimulation:
    '''
    This class initializes and starts multiple simulations of the Ethanol-MD Example
    from the SchNetPack MD Tutorial with different configured Langevin-Thermostats.
    '''
    def __init__(self,
                 thermostats: List[LangevinThermostat],
                 log_files: List[str],
                 device: torch.device,
                 steps,
                 calculator_class: Type[SchNetPackCalculator],
                 ):


        simulation = MDSimulations(
            device,
            "../training/forcetut/best_inference_model",
            "../test/md_ethanol.xyz",
            WORKDIR,
            1,
            300,
            0.5,
            5,
            0,
            thermostats,
            log_files,
            calc_cls
        )
        print("Start simulation")
        simulation.start_simulation(steps)
        print("Finnish simulation")
