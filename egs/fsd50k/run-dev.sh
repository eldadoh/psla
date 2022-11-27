#!/bin/bash
#SBATCH -p sm
#SBATCH -x sls-sm-1,sls-2080-[1,3],sls-1080-3,sls-sm-5
#SBATCH -p gpu
#SBATCH -x sls-titan-[0-2,9]
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=48000
#SBATCH --job-name="psla_fsd"
#SBATCH --output=./log_%j.txt

set -x
source ../../pslavenv/bin/activate
export TORCH_HOME=./

att_head=4
model=efficientnet
psla=True
eff_b=2
batch_size=16

if [ $psla == True ]; then
  impretrain=True
  freqm=48
  timem=192
  mixup=0.5
  bal=True
else
  impretrain=False
  freqm=0
  timem=0
  mixup=0
  bal=False
fi

p=none
if [ $p == none ]; then
  trpath=/home/ubuntu/psla/data/datafiles/fsd50k_tr_full.json
else
  trpath=./datafiles/fsd50k_tr_full_type1_2_${p}.json
fi

data_val=/home/ubuntu/psla/data/datafiles/fsd50k_val_full.json
data_eval=/home/ubuntu/psla/data/datafiles/fsd50k_eval_full.json
label_csv=/home/ubuntu/psla/egs/fsd50k/class_labels_indices.csv
num_workers=8

epoch=1
lr=5e-4
weight_averaging=False
wa_start=21
wa_end=40
lrscheduler_start=10
#exp_dir=./exp/demo-${model}-${eff_b}-${lr}-fsd50k-impretrain-${impretrain}-fm${freqm}-tm${timem}-mix${mixup}-bal-${bal}-b${batch_size}-le${p}-2
exp_dir=/home/ubuntu/psla/egs/fsd50k/exp/demo-${model}-${eff_b}-${lr}-fsd50k-impretrain-${impretrain}-fm${freqm}-tm${timem}-mix${mixup}-bal-${bal}-b${batch_size}-le${p}-2
mkdir -p $exp_dir

CUDA_CACHE_DISABLE=1 python3 ./src/run.py --data-train $trpath \
  --data-val $data_val --data-eval $data_eval \
  --label-csv $label_csv \
  --exp-dir $exp_dir --n-print-steps 1000 --save_model True --num-workers $num_workers \
  --n_class 200 --n-epochs ${epoch} --batch-size ${batch_size} --lr $lr \
  --model ${model} --eff_b $eff_b --impretrain ${impretrain} --att_head ${att_head} \
  --freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} --lr_patience 2 \
  --dataset_mean -4.6476 --dataset_std 4.5699 --target_length 3000 --noise False \
  --metrics mAP --warmup True --loss BCE --lrscheduler_start ${lrscheduler_start} --lrscheduler_decay 0.5 \
  --wa $weight_averaging --wa_start ${wa_start} --wa_end ${wa_end}
