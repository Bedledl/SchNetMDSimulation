import os
from typing import List

import torch

from ase.io import read

from schnetpack.md import System, UniformInit, Simulator
from schnetpack.md.calculators import SchNetPackCalculator
from schnetpack.md.integrators import VelocityVerlet
from schnetpack.md.neighborlist_md import NeighborListMD
from schnetpack.md.simulation_hooks.thermostats import LangevinThermostat
from schnetpack.transform import ASENeighborList


class MDSimulations:
    '''
    This class initializes and starts multiple simulations with different thermostats.
    '''
    def __init__(self,
                 device: torch.device,
                 model_path: str,
                 molecule_path: str,
                 md_workdir: str,
                 replicas: int,
                 initial_temperature: int, # Kelvin
                 time_step: float, # femtoseconds z.B. 0.5
                 cutoff: float, # Angstrom z.B. 5.0
                 cutoff_shell: float, # Angstrom z.B. 2.0
                 simulation_precision: torch.dtype, # z.B. torch.float32
                 thermostats: List[LangevinThermostat]
                 ):
        self.__check_input(model_path, molecule_path, md_workdir, device)

        self.__md_simulations = []

        # this object provides the "get_neighbors" function to calculate the environments
        # of each atom. That calculation has to be done in every step and is part of the
        # Calculator.calculate -> generate_inputs routine
        # but first we must intialize the object
        md_neighborlist = NeighborListMD(
            cutoff,
            cutoff_shell,
            ASENeighborList,
        )

        # the task of the integrator is, to update momenta and atom positions
        md_integrator = VelocityVerlet(time_step)

        md_calculator = SchNetPackCalculator(
            model_path,  # path to stored model
            "forces",  # force key
            "kcal/mol",  # energy units
            "Angstrom",  # length units
            md_neighborlist,  # neighbor list
            energy_key="energy",  # name of potential energies
            required_properties=[],  # additional properties extracted from the model
        )

        molecule = read(molecule_path)

        for thermostat in thermostats:
            # The System instance stores the state (molecule positions, forces, momenta...) of the syste
            md_system = System()
            md_system.load_molecules(
                molecule,
                replicas,
                position_unit_input="Angstrom"
            )

            # Initializes the system momenta according to a uniform distribution
            # scaled to the given temperature.
            md_initializer = UniformInit(
                initial_temperature,
                remove_center_of_mass=True,
                remove_translation=True,
                remove_rotation=True,
            )
            md_initializer.initialize_system(md_system)

            # build simulator_hooks
            simulator_hooks = [
                thermostat
            ]

            # And now, we can create the Simulator Object which has pointers to all the
            # other components like Integrator or Calculator
            md_simulator = Simulator(
                md_system,
                md_integrator,
                md_calculator,
                simulator_hooks=simulator_hooks
            )

            md_simulator.to(device=device, dtype=simulation_precision)

            self.__md_simulations.append(md_simulator)

    def start_simulation(self, n_steps: int):
        # TODO
        # how do i run multiple simulations at once?
        self.__md_simulator.simulate(n_steps)


    def __check_input(self, model_path: str,
                      molecule_path: str,
                      md_workdir: str,
                      device: torch.device):
        if not os.path.isdir(md_workdir):
            raise ValueError(f"Path '{md_workdir}' does not exist.")

        if not os.path.isfile(model_path):
            raise ValueError(f"File '{model_path}' does not exist.")

        if not os.path.isfile(molecule_path):
            raise ValueError(f"File '{molecule_path}' does not exist.")

        torch.device(device)


class EthanolSimulation:
    '''
    This class initializes and starts multiple simulations of the Ethanol-MD Example
    from the SchNetPack MD Tutorial with different configured Langevin-Thermostats.
    '''
    def __init__(self,
                 thermostats: List[LangevinThermostat],
                 device: device
                 ):
