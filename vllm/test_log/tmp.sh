sbatch --gres=gpu:1 -J vllm_serve \
    vllm_serve_sbatch.sh \
    s3://datafrontier/leihaodong/Qwen/Qwen2-0.5B/ \
    Qwen2-0.5B \
    1 \
    8101 \