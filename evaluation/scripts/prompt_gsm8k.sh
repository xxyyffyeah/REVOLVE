export PYTHONPATH=$(pwd)

python -u ./evaluation/prompt_optimization.py \
--task GSM8K_DSPy \
--engine azure-gpt4o \
--optimizer_version v1 \
--run_validation \
--num_threads 20 \

python -u ./evaluation/prompt_optimization.py \
--task GSM8K_DSPy \
--engine azure-gpt4o \
--optimizer_version v1_momentum \
--run_validation \
--num_threads 20 \

python -u ./evaluation/prompt_optimization.py \
--task GSM8K_DSPy \
--engine azure-gpt4o \
--optimizer_version v2 \
--run_validation \
--num_threads 20 \

#python -u ./evaluation/gsm8k_dspy.py \
#--model meta-llama/Meta-Llama-3.1-70B-Instruct
