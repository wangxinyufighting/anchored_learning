model_name=Qwen3-4B
data_name=medcalc_train
lr=1e-5
ratio=0.5
num_stages=10
epochs_per_stage=5

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 testv3_iter_og.py \
    --teacher_model_path /root/autodl-tmp/LlamaFactory/saves/${model_name}_${data_name}_1e-6/checkpoint-4410\
    --student_model_path /root/autodl-tmp/models/${model_name} \
    --ref_model_path /root/autodl-tmp/models/${model_name} \
    --data_path /root/autodl-tmp/LlamaFactory/data/${data_name}.json\
    --data_name ${data_name} \
    --epochs_per_stage ${epochs_per_stage}\
    --num_stages ${num_stages} \
    --mixing_ratio ${ratio} \
    --lr ${lr} \
    --temperature 0.9 \
    --lmbda 1.0 \
    --per_device_train_batch_size 2 \
    --exp_name e${epochs_per_stage}s${num_stages}_on_policy