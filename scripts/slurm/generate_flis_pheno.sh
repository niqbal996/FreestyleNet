#!/usr/bin/env bash
/opt/conda/envs/freestyle/bin/python scripts/FLIS.py \
                                    --batch_size 1 \
                                    --config configs/stable-diffusion/v1-finetune_Pheno.yaml \
                                    --ckpt /netscratch/naeem/freestyle_logs/2024-05-12T01-42-35_phenobench_global_lighting_20k/checkpoints/last.ckpt \
                                    --json /netscratch/naeem/sugarbeet_syn_v2/flis_images/layout_morning.json \
                                    --outdir /netscratch/naeem/sugarbeet_syn_v2/flis_images/ \
                                    --W 1024 \
                                    --H 1024 \
                                    --plms
