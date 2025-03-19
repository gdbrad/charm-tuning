import h5py
import numpy as np

class InputOutput:

    def __init__(self, project_path=None):
        super(InputOutput, self).__init__(project_path=project_path)

        self.file_h5 = self.project_path + '/data/pion_2pt_numvecs_32_tsrc_24.h5'


    def load_data(self):
        data = {}
        with h5py.File(self.file_h5) as f:
            for tsrc in f['/pion_contractions']:
                cfg_list = list(f[f'/pion_contractions/{tsrc}/'])
                t = int(tsrc[5:])

                # assuming all tsrc have the same cfg_list...
                data[('pion', t)] = np.array([f[f'/pion_contractions/{tsrc}/{cfg}'] 
                    for cfg in sorted(cfg_list, key=lambda cfg : int(cfg[4:]))][:])
                
                data[('pion', t)] = np.roll(data[('pion', t)], -t*4)

        return data