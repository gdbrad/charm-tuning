import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
# import corr_lib

# corr_lib.Fitter(n_states=1)

def plot_and_save_correlators(hdf5_file_path, gamma_chroma,base_output_dir='plots'):
    try:
        file_name = Path(hdf5_file_path).stem
        
        with h5py.File(hdf5_file_path, 'r') as h5f:
            meson_group = h5f['D_s']
            
            if not meson_group.keys():
                print(f"No data found in HDF5 file: {hdf5_file_path}")
                return
            # if gamma_chroma:
            #     gamma_group = meson_group[gamma_key]
            #     gamma_value = int(gamma_key.split('_')[1])

            # else:
            for gamma_key in meson_group.keys():
                gamma_group = meson_group[gamma_key]
                gamma_value = int(gamma_key.split('_')[1])
                
                output_dir = os.path.join(base_output_dir, file_name, f'gamma_{gamma_value}')
                os.makedirs(output_dir, exist_ok=True)
                
                plt.figure(figsize=(10, 6))
                
                has_data = False
                
                for mom_key in gamma_group.keys():
                    mom_group = gamma_group[mom_key]
                    mom_num = int(mom_key.split('_')[1])
                    sink_mom = mom_group.attrs['sink_mom']
                    
                    correlator = mom_group['correlator'][:]
                    correlator_mag = np.abs(correlator)
                    
                    if len(correlator_mag) == 0:
                        print(f"No data for {file_name}: gamma={gamma_value}, mom={sink_mom}")
                        continue
                    
                    has_data = True
                    time = np.arange(len(correlator))
                    
                    plt.semilogy(time, correlator_mag, 
                               label=f'mom={sink_mom}',
                               marker='o', linestyle='-', markersize=4)
                
                if has_data:
                    plt.xlabel('Nt')
                    plt.ylabel('Correlator (log)')
                    plt.title(f'D_s Corr (γ={gamma_value}) - {file_name}')
                    plt.grid(True, which="both", ls="-", alpha=0.2)
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.tight_layout()
                    
                    output_file = os.path.join(output_dir, f'corr_gamma_{gamma_value}.png')
                    plt.savefig(output_file, dpi=300, bbox_inches='tight')
                    print(f"Saved plot: {output_file}")
                
                plt.close()

    except Exception as e:
        print(f"Error processing {hdf5_file_path}: {str(e)}")

def main():
    hdf5_dir = Path('D_s-h5')
    base_output_dir = 'plots'
    
    os.makedirs(base_output_dir, exist_ok=True)
    
    for hdf5_file in hdf5_dir.glob('*.h5'):
        print(f"\nProcessing {hdf5_file}")
        plot_and_save_correlators(hdf5_file, base_output_dir)

if __name__ == "__main__":
    main()