# anchored_learning

## 环境

```
conda create -n anchor python=3.11.0

conda activate anchor

cd LlamaFactory
pip install -e .
pip install -r requirements/metrics.txt

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 trl
pip install flash-attn --no-build-isolation
pip3 install deepspeed
```


## 训练

sh train_pipeline.sh

会自动执行 sft ，再执行 anchored learning

需要修改：

1. 下载相关模型：Qwen3-4B
2. 修改 base_model_path
3. sh train_pipeline.sh启动训练
4. 请根据实际情况调整：
   1. sft_batch_size
   2. al_batch_size
