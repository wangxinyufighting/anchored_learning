#!/usr/bin/env bash                                                                  
set -euo pipefail                                                                    
                                                                                     
model_name=Qwen3-4B-Instruct-2507                                                    
data_name=medcalc_train                                                              
lr=1e-5                                                                              
ratio=0.5                                                                            
num_stages=10                                                                        
epochs_per_stage=5                                                                   
                                                                                     
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 torchrun --nproc_per_node=6 al_og.py \              
    --model_name "${model_name}" \                                                   
    --teacher_model_path "/czsun/zhi/xywang/LlamaFactory/saves/${model_name}_${data_name}_1e-5" \    
    --student_model_path /czsun/models/${model_name} \                               
    --ref_model_path /czsun/models/${model_name} \                                   
    --data_path "../LlamaFactory/data/${data_name}.json" \                           
    --data_name "${data_name}" \                                                     
    --epochs_per_stage "${epochs_per_stage}" \                                       
    --num_stages "${num_stages}" \                                                   
    --mixing_ratio "${ratio}" \                                                      
    --lr "${lr}" \                                                                   
    --per_device_train_batch_size 2 \                                                
    --gradient_checkpointing \                                                       
    --save_steps 2000 \                                                             
    --save_total_limit 4 \                                                           
    --max_length 8192 \                                                              
    --exp_name "e${epochs_per_stage}s${num_stages}"     