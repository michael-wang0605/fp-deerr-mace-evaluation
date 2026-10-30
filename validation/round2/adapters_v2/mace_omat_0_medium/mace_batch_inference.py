# Generated isolated adapter; see adapter_manifest.json.
"""Use MACE's own ASE graph conversion and model arguments for force batches."""
import sys
sys.path.append('C:\\Users\\mwang\\Documents\\Codex\\2026-10-09\\what-does-prof-mo-in-my')
import numpy as np
import torch
from mace.tools import torch_geometric, torch_tools
from mace import data as mace_data

def predict(calculator, atoms_list, batch_size=16):
    assert calculator.num_models == 1 and (not calculator.use_compile)
    assert calculator.default_dtype == 'float64' and calculator.head == 'default'
    assert calculator.pad_num_atoms == calculator.pad_num_edges == 0
    assert not calculator.compute_atomic_stresses
    results = []
    for offset in range(0, len(atoms_list), batch_size):
        atoms = atoms_list[offset:offset + batch_size]
        calculator.arrays_keys.update({calculator.charges_key: 'charges'})
        keyspec = mace_data.KeySpecification(info_keys=calculator.info_keys, arrays_keys=calculator.arrays_keys)
        with torch_tools.default_dtype(calculator.default_dtype):
            graphs = [mace_data.AtomicData.from_config(mace_data.config_from_atoms(item, key_specification=keyspec, head_name=calculator.head), z_table=calculator.z_table, cutoff=calculator.r_max, heads=calculator.available_heads) for item in atoms]
        batch = torch_geometric.Batch.from_data_list(graphs).to(calculator.device)
        dtype = next(calculator.models[0].parameters()).dtype
        for key in batch.keys:
            value = batch[key]
            if torch.is_tensor(value) and torch.is_floating_point(value):
                batch[key] = value.to(dtype=dtype)
        output = calculator.models[0](batch.to_dict(), compute_stress=True, training=False, compute_edge_forces=False, compute_atomic_stresses=False)
        energy = output['energy'].detach().cpu().numpy() * calculator.energy_units_to_eV
        force = output['forces'].detach().cpu().numpy() * calculator.energy_units_to_eV / calculator.length_units_to_A
        stress = output['stress'].detach().cpu().numpy() * calculator.energy_units_to_eV / calculator.length_units_to_A ** 3
        limits = np.cumsum([0] + [len(item) for item in atoms])
        for (i, item) in enumerate(atoms):
            results.append({'energy_total': float(energy[i]), 'forces': force[limits[i]:limits[i + 1]].copy(), 'stress': stress[i].copy()})
        del output, batch, graphs
    return results
