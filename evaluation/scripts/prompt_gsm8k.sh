export PYTHONPATH=$(pwd)

python -u ./evaluation/prompt_optimization.py \
--task GSM8K_DSPy \
--engine llama-3_1 \
--optimizer_version v1 \
--run_validation \
--num_threads 10 \

python -u ./evaluation/prompt_optimization.py \
--task GSM8K_DSPy \
--engine llama-3_1 \
--optimizer_version v2 \
--run_validation \
--num_threads 5 \
