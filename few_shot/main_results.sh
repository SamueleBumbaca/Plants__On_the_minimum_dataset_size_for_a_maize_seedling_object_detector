#!/bin/bash
export PYTHONPATH=$(pwd):$PYTHONPATH
echo $PYTHONPATH
datalist=(
#"ArTaxOr"
#"clipart1k"
# "DIOR"
# "UODD"
# "NEUDET"
# "FISH"
# "1"
# "2"
"3"
)
shot_list=(
# 1
5
10
30
50
)
model_list=(
# "s"
# "b"
"l"
)
for model in "${model_list[@]}"; do
  echo "model: ${model}"
  for dataset in "${datalist[@]}"; do
    echo "dataset: ${dataset}"
    for shot in "${shot_list[@]}"; do
      echo "shot: ${shot}"
      CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --num-gpus 1 --config-file configs/${dataset}/vit${model}_shot${shot}_${dataset}_finetune.yaml MODEL.WEIGHTS weights/trained/few-shot/vit${model}_0089999.pth DE.OFFLINE_RPN_CONFIG configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml OUTPUT_DIR output/vit${model}/${dataset}_${shot}shot/
    done
  done
done
