#!/bin/bash
conf_start=11
conf_step=40
conf_end=1971
basedir="ini-multi"

for i in $(seq ${conf_start} ${conf_step} ${conf_end} ); do
  j=$(printf %02d $i)

  cfg_dir="${basedir}/cnfg${j}"
  
  if [ -d "${cfg_dir}" ]; then
    cd ${cfg_dir}
    
    echo "Starting config ${i}"
    spec="spec_cfg${j}.sh"
    
    if [ -f "${spec}" ]; then
      sbatch ${spec}
    else
      echo "Shell script ${spec} not found in ${cfg_dir}"
    fi
    
    cd - > /dev/null
  else
    echo "Configuration directory ${cfg_dir} does not exist"
  fi
done
