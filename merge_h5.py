import h5py
import numpy as np
import os
import re

directory = "D_s-h5"  
cfg_pattern = r"n_cfg_(\d+)"
mass_pattern = r"Ds_(smpt|smsm)_([\d.]+)\.h5$"
cfg_ids = []
output_file = "D_s_final.h5"

with h5py.File(output_file, 'w') as f_out:
    ds_group = f_out.create_group("D_s")
    
    for filename in os.listdir(directory):
        if filename.endswith(".h5"):
            filepath = os.path.join(directory, filename)
            
            cfg_match = re.search(cfg_pattern, filename)
            if cfg_match:
                cfg_id = int(cfg_match.group(1))
                if cfg_id not in cfg_ids:
                    cfg_ids.append(cfg_id)
            else:
                print(f"Could not extract cfg id from {filename}")
                continue
            
            mass_match = re.search(mass_pattern, filename)
            if mass_match:
                smear_type = mass_match.group(1)  # 'smpt' or 'smsm'
                mass = mass_match.group(2)        # '0.15', '0.25', etc.
            else:
                print(f"Could not extract mass/smearing from {filename}")
                continue
            
            cfg_group = ds_group.require_group(f"cfg_{cfg_id}")
            mass_group = cfg_group.require_group(f"mass_{mass}")
            smear_group = mass_group.require_group(smear_type)
            
            with h5py.File(filepath, 'r') as f_in:
                for gamma_idx in range(16): 
                    gamma_key = f"D_s/gamma_{gamma_idx}/mom_0/correlator"
                    if gamma_key in f_in:
                        correlator = f_in[gamma_key][:] 
                        gamma_group = smear_group.require_group(f"gamma_{gamma_idx}")
                        mom_group = gamma_group.require_group("mom_0")
                        mom_group.create_dataset("correlator", data=correlator)
                    else:
                        print(f"Warning: {gamma_key} not found in {filename}")
            
            print(f"Processed {filename} with cfg id {cfg_id}, mass {mass}, smear {smear_type}")

    ds_group.attrs["cfg_ids"] = np.array(cfg_ids)

print(f"Combined data saved to {output_file}")
print(f"Total configurations processed: {len(cfg_ids)}")
print(f"Configuration IDs: {sorted(cfg_ids)}")

with h5py.File(output_file, 'r') as f:
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"{name}: {obj.shape}")
        elif isinstance(obj, h5py.Group):
            print(f"{name}/")
    f.visititems(print_structure)