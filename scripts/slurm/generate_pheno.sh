#!/usr/bin/env bash
# ln -s /netscratch/naeem/cocostuff/stable-diffusion/ /home/iqbal/FreestyleNet/models/ldm/
/opt/conda/envs/freestyle/bin/python scripts/LIS.py \
                                    --batch_size 2 \
                                    --config configs/stable-diffusion/v1-finetune_Pheno.yaml \
                                    --ckpt /netscratch/naeem/freestyle_logs/2024-05-12T01-42-35_phenobench_global_lighting_20k/checkpoints/last.ckpt \
                                    --dataset Phenobench \
                                    --outdir /netscratch/naeem/sugarbeet_syn_v2/images_lis_global/ \
                                    --txt_file /netscratch/naeem/sugarbeet_syn_v2/pheno_syn_v2.txt \
                                    --data_root /netscratch/naeem/sugarbeet_syn_v2/ \
                                    --W 1024 \
                                    --H 1024 \
                                    --plms
