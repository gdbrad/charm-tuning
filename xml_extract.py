import xml.etree.ElementTree as ET
import h5py
import numpy as np
import argparse
import os
from pathlib import Path

def parse_xml_to_hdf5(xml_file_path, hdf5_file_path,smear):
    """Convert a single XML file to HDF5 format"""
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    print(tree,root)
    with h5py.File(hdf5_file_path, 'w') as h5f:
        meson_group = h5f.create_group('D_s')
        if smear == 'smpt':
            sm_path = './/Shell_Point_Wilson_Mesons/elem'
        elif smear == 'smsm':
            sm_path = './/Shell_Shell_Wilson_Mesons/elem'
            
        for elem in root.findall(sm_path):
            gamma_value = int(elem.find('gamma_value').text)
            gamma_group = meson_group.create_group(f'gamma_{gamma_value}')
            
            momenta = elem.find('momenta')
            for mom_elem in momenta.findall('elem'):
                sink_mom_num = int(mom_elem.find('sink_mom_num').text)
                sink_mom = mom_elem.find('sink_mom').text
                
                mom_group = gamma_group.create_group(f'mom_{sink_mom_num}')
                mom_group.attrs['sink_mom'] = sink_mom
                
                mesprop = mom_elem.find('mesprop')
                correlator_data = []
                
                for data_elem in mesprop.findall('elem'):
                    re = float(data_elem.find('re').text)
                    im = float(data_elem.find('im').text)
                    correlator_data.append(complex(re, im))
                
                correlator_array = np.array(correlator_data, dtype=np.complex128)
                mom_group.create_dataset('correlator', data=correlator_array)

def main(options):
    input_dir = Path(options.xml_dir)
    output_dir = Path('D_s-h5/')
    output_dir.mkdir(exist_ok=True)
    for xml_file in input_dir.glob('*.xml'):
        try:
            hdf5_file = output_dir / f"{xml_file.stem}.h5"
            if 'smsm' in xml_file.stem:
                parse_xml_to_hdf5(xml_file, hdf5_file,'smsm')
            elif 'smpt' in xml_file.stem:
                parse_xml_to_hdf5(xml_file, hdf5_file,'smpt')
            print(f"Successfully converted {xml_file} to {hdf5_file}")
            if options.verify:
                with h5py.File(hdf5_file, 'r') as h5f:
                    print(f"\nHDF5 file structure for {hdf5_file}:")
                    def print_structure(name, obj):
                        print(name)
                    h5f.visititems(print_structure)
        except Exception as e:
            print(f"Error processing {xml_file}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert XML files to HDF5 format')
    parser.add_argument('--xml_dir', type=str, required=True, 
                       help='Directory containing XML files to process')
    parser.add_argument('--verify', action='store_true',
                       help='Verify and print HDF5 structure after conversion')
    options = parser.parse_args()

    main(options)