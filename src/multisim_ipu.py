import datetime
import os
from functools import partial
from multiprocessing import Pipe, Process, Pool

import schnetpack

from schnetpack.data.loader import _atoms_collate_fn

import torch

import poptorch
from schnetpack.md.simulation_hooks import LangevinThermostat
from constants import WORKDIR
from ethanol_simulation import EthanolSimulation

import concurrent.futures

from inputs_transformations import combine_inputs, split_inputs
from SchNetPackCalcIpu import MultiSimCalc

temperatures = [50, 300, 600, 1200]

thermostats = [
    LangevinThermostat(temp, 10) for temp in temperatures
]

date_obj = datetime.datetime.now()
date = date_obj.strftime("%d-%m-%Y_%H:%M")

log_files = [
    os.path.join(WORKDIR, f"log_thermostat_{temp}_{date}") for temp in temperatures
]

pipes = [
    Pipe() for _ in temperatures
]

steps = 10

simulation_objects = [
    EthanolSimulation([thermostat], [log_file], torch.device("ipu"), MultiSimCalc, pipe_endpoint=pipe[1])
    for thermostat, log_file, pipe in zip(thermostats, log_files, pipes)
]

model_file = "../training/forcetut/best_inference_model"
print("Loading model from {:s}".format(model_file))
# load model and keep it on CPU, device can be changed afterwards
model = torch.load(model_file, map_location="cpu").to(torch.float64)

# from schnetpack.md.calculators.schnetpack_calculator.SchNetPackCalculator._deactivate_postprocessing
if hasattr(model, "postprocessors"):
    for pp in model.postprocessors:
        if isinstance(pp, schnetpack.transform.AddOffsets):
            print("Found `AddOffsets` postprocessing module...")
            print(
                "Constant offset of {:20.11f} per atom  will be removed...".format(
                    pp.mean.detach().cpu().numpy()
                )
            )
model.do_postprocessing = False

model = model.eval()

ipu_executor = poptorch.inferenceModel(model)


with Pool(len(temperatures)) as pool:
    start_simulation_n_steps = partial(EthanolSimulation.start_simulation, steps=steps)
    pool.map(start_simulation_n_steps, simulation_objects)

    for _ in range(steps + 1):
        inputs_collected = []
        for index, pipe in enumerate(pipes):
            inputs = pipe[0].recv()
            inputs_collected.append(inputs)

        batch = combine_inputs(inputs_collected)
        output = ipu_executor(batch)
        outputs_collected = split_inputs(output, len(inputs_collected))
        for outputs, pipe in zip(outputs_collected, pipes):
            pipe[0].send(outputs)
