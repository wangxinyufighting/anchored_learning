# anchored_learning

## 环境

```
conda create -n anchored python=3.12.0

conda activate anchored

pip install torch==2.9.0 trl
pip install flash-attn --no-build-isolation

cd LlamaFactory
pip install -e .
pip install -r requirements/metrics.txt
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
