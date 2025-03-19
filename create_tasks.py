
import argparse
import os
import jinja2
import yaml
import spec_xml

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
FDIR = os.path.dirname(os.path.realpath(__file__))
TEMPLATE = os.path.join(FDIR, 'templates')

class TaskHandler:
    def __init__(self, env):
        self.templates = {
            'spec': env.get_template('spec_Ds_cluster.xml.j2'),
            'chroma_spec': env.get_template('test_spec.sh.j2')
        }

        self.xml_classes = {
            'spec': spec_xml.Spec,
            'chroma_spec': spec_xml.Spec
        }

def main(options):
    with open(os.path.join(options.in_file)) as f:
        dataMap = yaml.safe_load(f)
    missing_values = [key for key in ['run_path', 'cfg_path'] if key not in dataMap]

    if missing_values:
        for key in missing_values:
            value = input(f"you forgot to include '{key}' in your infile dummy!: ")
            dataMap[key] = value

        # Rewrite the YAML file with added values
        with open(options.in_file, 'w') as f:
            yaml.safe_dump(dataMap, f)

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(TEMPLATE), undefined=jinja2.StrictUndefined)
    handler = TaskHandler(env)
    run_objects = []
    for task in options.list_tasks:
        if 'spec' in task:
            run_objects.append('spec')
            run_objects.append('chroma_spec')

    for cfg_id in range(options.cfg_i, options.cfg_f, options.cfg_step):
        print('Creating scripts for configuration', cfg_id)
        cfg_dir = os.path.realpath(os.path.join(options.ini_path, f'cnfg{cfg_id:02d}'))

        for obj in run_objects:
            obj_dir = os.path.join(cfg_dir)
            os.makedirs(obj_dir, exist_ok=True)
            if obj == 'chroma_spec':
                ini_out = f'{obj.split("_")[1]}_cfg{cfg_id:02d}.sh'
            elif obj == 'spec':
                ini_out = f'{obj}_cfg{cfg_id:02d}.ini.xml'

            ini_out_path = os.path.join(obj_dir, ini_out)
            with open(options.in_file) as f:
                dataMap = yaml.safe_load(f)

            if obj in options.list_tasks:
                base = handler.xml_classes[obj]
                expected_keys = base.model_fields.keys()
                filtered_data = {k: v for k, v in dataMap.items() if k in expected_keys}
                filtered_data['cfg_id'] = f'{cfg_id:02d}'
                ens_props = spec_xml.parse_ensemble(short_tag=filtered_data['ens_short'])
                filtered_data.update(ens_props)
                output_xml = handler.templates[obj].render(filtered_data)
                
                print(f"Writing file {ini_out_path} for object {obj}")
                
                with open(ini_out_path, 'w') as f:
                    f.write(output_xml)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, required=True)
    parser.add_argument('--cfg_i', type=int, required=True)
    parser.add_argument('--cfg_f', type=int, required=True)
    parser.add_argument('--cfg_step', type=int, nargs='?', default=10, help='default: %(default)s')
    parser.add_argument('--ini-path', type=str, required=True)
    parser.add_argument('-l', '--list_tasks', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument('--overwrite', type=bool, nargs='?', default=False, help='if true, overwrite existing xml and sh scripts for chroma task of interest')
    parser.add_argument('--cfg_path', default='/p/project1/exotichadrons/pederiva/6stout/beta_3.70/ms_0.000/mud_-0.022/s32t96/cnfg/', help='default: %(default)s')
    parser.add_argument('--code_dir', default='/p/scratch/exotichadrons/chroma-distillation', help='default: %(default)s')
    parser.add_argument('--run_dir', default='', help='default: %(default)s')
    options = parser.parse_args()

    main(options)
