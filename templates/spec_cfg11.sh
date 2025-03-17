#!/bin/bash
# template for computing D_s mesons with chroma 

#SBATCH --nodes=4
#SBATCH --partition=dc-gpu
#SBATCH --gpu-bind=none
#SBATCH --account=exotichadrons
#SBATCH -t 06:00:00
#SBATCH --gres=gpu:4

export OPENBLAS_NUM_THREADS=16
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=0,1,2,3

CODE_DIR=/p/scratch/exotichadrons/chroma-superb
myenv=$CODE_DIR/env-new-jureca-gpu.sh
formenv=$CODE_DIR/env_extra.sh
source $myenv 
source $formenv

chroma=$CODE_DIR/install/chroma-quda-qdp-jit-double-nd4-cmake-superbblas-cuda/bin/chroma
BASE_DIR=/p/scratch/exotichadrons/charm-tuning
log=$BASE_DIR/test-spec-cfg11.log
in=$BASE_DIR/ini-spec/cnfg11/spec_cfg11.ini.xml
out=$BASE_DIR/test_spec_11.out.xml
rm $out
stdout="$BASE_DIR/test_spec_11.out"

export OPTS=" -geom 1 2 2 4"
echo "START  "$(date "+%Y-%m-%dT%H:%M")
srun -n 16 -c 16 $chroma $OPTS -i $in -o $out -l $log > $stdout 2>&1
echo "FINISH JOB "$(date "+%Y-%m-%dT%H:%M")
