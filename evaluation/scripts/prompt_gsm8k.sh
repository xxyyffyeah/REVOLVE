export PYTHONPATH=$(pwd)

python -u ./evaluation/prompt_optimization.py \
--task GSM8K_DSPy \
--engine meta-llama/Meta-Llama-3.1-70B-Instruct \
--optimizer_version v1 \
--run_validation \
--num_threads 10 \

python -u ./evaluation/prompt_optimization.py \
--task GSM8K_DSPy \
--engine meta-llama/Meta-Llama-3.1-70B-Instruct \
--optimizer_version v2 \
--run_validation \
--num_threads 10 \

python -u ./evaluation/gsm8k_dspy.py \
--model meta-llama/Meta-Llama-3.1-70B-Instruct
