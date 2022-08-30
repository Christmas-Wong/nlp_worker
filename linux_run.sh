config="/data/wangfei2/code/nlp_worker/configs/mll_bert.yml"

# 定位conda
source /data/miniconda3/etc/profile.d/conda.sh

# 切换环境
conda activate wangfei

# 注册wandb
wandb login --host=https://api.wandb.ai --relogin c3a8aec3a536dde837f1822cef6325e8025aebd7

# 执行python训练脚本
CUDA_VISIBLE_DEVICES=4 python3 main.py ${config}