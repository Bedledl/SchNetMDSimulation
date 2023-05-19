import torch
import pickle

from poptorch_geometric.dataloader import DataLoader

model_file = "md_ethanol.model"
inputs_test_file = "inputs_test_obj.dat"

model = torch.load(model_file, map_location="cpu").to(torch.float64)
model = model.eval()

with open("inputs_test_obj.dat", "rb") as fl:
    inputs = pickle.load(fl)


class MoleculeListDataset(torch.utils.data.Dataset):
    def __init__(self, molecules):
        super(MoleculeListDataset, self,).__init__()

        self.molecules = molecules
        self.transform = lambda x: x  # some infer transform

    def __len__(self):
        return len(self.molecules)

    def __getitem__(self, idx):
        molecule = self.molecules[idx]

        return self.transform(molecule)

dataset = MoleculeListDataset(inputs)

in_data = DataLoader(dataset, batch_size=1)

ipu_executor = poptorch.inferenceModel(model)
results = []
for batch in in_data:
    result = ipu_executor(dataset)
    results.append(result)
