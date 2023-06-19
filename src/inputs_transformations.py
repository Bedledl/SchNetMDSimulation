from typing import List, Dict, Union

import schnetpack.properties as props
import torch

idx_keys = {props.idx_i, props.idx_j, props.idx_i_triples}
idx_triple_keys = {props.idx_j_triples, props.idx_k_triples}
integer_keys = {props.n_molecules}


def combine_inputs(inputs_list: List[Dict[str, Union[torch.Tensor, int]]])\
        -> Dict[str, Union[torch.Tensor, int]]:
    """
    Collect inputs to batch.
    Copied form src/schnetpack/data/loader.py: atoms_collate_fn from the schnetpack package
    :param input_a:
    :param input_b:
    :return:
    """
    coll_batch = {}

    keys = inputs_list[0].keys()

    for key in keys:
        if (key not in idx_keys) and (key not in idx_triple_keys) and (key not in integer_keys):
            #normal tensor attributes, that can be combined
            coll_batch[key] = torch.cat([inputs[key] for inputs in inputs_list], 0)
        elif key in idx_keys:
            # add for example idx_i_local to store the original neighbor list TODO why?
            coll_batch[key + "_local"] = torch.cat([inputs[key] for inputs in inputs_list], 0)
        elif key in integer_keys:
            coll_batch[key] = sum([inputs[key] for inputs in inputs_list])

            # n_atoms contains an entry for every molecule with the number of associated atoms as value
    # offsets contains for every molecule the offset that has to be added to the atom ids
    offsets_m = torch.cumsum(coll_batch[props.n_atoms], dim=0)
    offsets_m = torch.cat([torch.zeros((1,), dtype=offsets_m.dtype), offsets_m], dim=0)

    idx_m = torch.repeat_interleave(
        torch.arange(len(inputs_list)), repeats=coll_batch[props.n_atoms], dim=0
    )
    coll_batch[props.idx_m] = idx_m

    for key in idx_keys:
        if key in keys:
            coll_batch[key] = torch.cat(
                [d[key] + off for d, off in zip(inputs_list, offsets_m)], 0
            )

    # Shift the indices for the atom triples
    for key in idx_triple_keys:
        if key in keys:
            indices = []
            offset = 0
            for idx, d in enumerate(inputs_list):
                indices.append(d[key] + offset)
                offset += d[props.idx_j].shape[0]
            coll_batch[key] = torch.cat(indices, 0)

    return coll_batch


def split_inputs(batch: Dict[str, Union[torch.Tensor, int]], num_inputs: int)\
        -> List[Dict[str, Union[torch.Tensor, int]]]:
    #TODO vielleicht können hier noch viel mehr Werte übersprungen werden, z.B. die Nachbarlisten sollten nicht mehr wichtig sein
    inputs_list = [{} for _ in range(num_inputs)]

    def set_attribute(attr: str, values: List):
        for inputs, value in zip(inputs_list, values):
            inputs[attr] = value

    for key, value in batch.items():
        if (key not in idx_keys) and (key not in idx_triple_keys) and (key not in integer_keys):
            #normal tensor attributes, that can be chunked
            set_attribute(key, value.chunk(num_inputs))
        elif key in idx_keys:
                pass
        elif key in integer_keys:
            set_attribute(key, [value/num_inputs for _ in range(num_inputs)])
        elif key in idx_triple_keys:
            raise NotImplementedError

    return inputs_list

