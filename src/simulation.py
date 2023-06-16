import os
from typing import List, Type

import torch

from ase.io import read

from schnetpack import properties
from schnetpack.md import System, UniformInit, Simulator
from schnetpack.md.calculators import SchNetPackCalculator
from schnetpack.md.integrators import VelocityVerlet
from schnetpack.md.neighborlist_md import NeighborListMD
from schnetpack.md.simulation_hooks import MoleculeStream, FileLogger, PropertyStream
from schnetpack.md.simulation_hooks.thermostats import LangevinThermostat
from schnetpack.transform import KNNNeighborList, ASENeighborList, CompleteNeighborList


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
                 thermostats: List[LangevinThermostat],
                 log_files: List[str],
                 calculator_class: Type[SchNetPackCalculator]
                 ):
        MDSimulations.__check_input(model_path, molecule_path, md_workdir, device, thermostats, log_files)

        self.__md_simulations = []

        # this object provides the "get_neighbors" function to calculate the environments
        # of each atom. That calculation has to be done in every step and is part of the
        # Calculator.calculate -> generate_inputs routine
        # but first we must intialize the object
        md_neighborlist = NeighborListMD(
            cutoff,
            cutoff_shell,
            KNNNeighborList,
        )

        # the task of the integrator is, to update momenta and atom positions
        md_integrator = VelocityVerlet(time_step)

        md_calculator = calculator_class(
            model_path,  # path to stored model
            "forces",  # force key
            "kcal/mol",  # energy units
            "Angstrom",  # length units
            md_neighborlist,  # neighbor list
            energy_key="energy",  # name of potential energies
            required_properties=[],  # additional properties extracted from the model
        )

        molecule = read(molecule_path)

        for thermostat, log_file in zip(thermostats, log_files):
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

            buffer_size = 100
            # Set up data streams to store positions, momenta and the energy
            data_streams = [
                MoleculeStream(store_velocities=True),
                PropertyStream(target_properties=[properties.energy]),
            ]

            file_logger = FileLogger(
                log_file,
                buffer_size,
                data_streams=data_streams,
                every_n_steps=1,  # logging frequency
                precision=32,  # floating point precision used in hdf5 database
            )

            # build simulator_hooks
            simulator_hooks = [
                thermostat,
                file_logger
            ]

            # And now, we can create the Simulator Object which has pointers to all the
            # other components like Integrator or Calculator
            md_simulator = Simulator(
                md_system,
                md_integrator,
                md_calculator,
                simulator_hooks=simulator_hooks
            )

            #md_simulator.float()
            #md_simulator._apply(lambda t: t.to(torch.int32) if t.long() else t)
            #md_simulator.to(device=device, dtype=simulation_precision)

            self.__md_simulations.append(md_simulator)

    def start_simulation(self, n_steps: int):
        # TODO
        # how do i run multiple simulations at once?
        for sim in self.__md_simulations:
            sim.simulate(n_steps)
            print(f"The IPU calculated {sim.calculator.steps} steps in {sim.calculator.d} seconds.")
            print(f"This is one step in {sim.calculator.d / sim.calculator.steps} seconds")
            print(f"The IPU needed  {sim.calculator.d_neighborlist} seconds for neighborlist calculations.")
            print(f"This is one neighborlist calculation in  {sim.calculator.d_neighborlist/ sim.calculator.steps} seconds")

    @staticmethod
    def __check_input(model_path: str,
                      molecule_path: str,
                      md_workdir: str,
                      device: torch.device,
                      thermostats: List[LangevinThermostat],
                      log_files: List[str]):
        if not os.path.isdir(md_workdir):
            raise ValueError(f"Path '{md_workdir}' does not exist.")

        if not os.path.isfile(model_path):
            raise ValueError(f"File '{model_path}' does not exist.")

        if not os.path.isfile(molecule_path):
            raise ValueError(f"File '{molecule_path}' does not exist.")

        torch.device(device)

        if len(thermostats) != len(log_files):
            raise ValueError(f"There must be passed a log_file for each thermostat."
                             f"Passed: {len(thermostats)} Thermostats and {len(log_files)} Logfiles")

