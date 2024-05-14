#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py --base configs/stable-diffusion/v1-finetune_Pheno.yaml \
                                      -t \
                                      --actual_resume /netscratch/naeem/cocostuff/freestyle-sd-v1-4-coco.ckpt \
                                      --logdir /netscratch/naeem/freestyle_logs/ \
                                      -n phenobench_global_lighting_20k_syn_val \
                                      --gpus 0, \
                                      --data_root /netscratch/naeem/cocostuff/phenobench_cocostuff/ \
                                      --train_txt_file /netscratch/naeem/cocostuff/phenobench_cocostuff/pheno_train.txt \
                                      --val_txt_file /netscratch/naeem/cocostuff/phenobench_synthetic/pheno_test.txt

# --actual_resume /netscratch/naeem/cocostuff/freestyle-sd-v1-4-coco.ckpt \
# /netscratch/naeem/freestyle_logs/2024-02-01T17-19-28_phenobench_coco_2/checkpoints/epoch=000009.ckpt
