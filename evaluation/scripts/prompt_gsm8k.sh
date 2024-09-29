export PYTHONPATH=$(pwd)

# Run optimization for llama3.1 8B on Object Counting
python -u ./evaluation/prompt_optimization.py \
--task BBH_object_counting \
--tg_engine azure-gpt4o \
--model meta-llama/Meta-Llama-3.1-8B-Instruct \
--optimizer_version v1 \
--run_validation \
--num_threads 20 \

python -u ./evaluation/prompt_optimization.py \
--task BBH_object_counting \
--tg_engine azure-gpt4o \
--model meta-llama/Meta-Llama-3.1-8B-Instruct \
--optimizer_version v1_momentum \
--run_validation \
--num_threads 20 \

python -u ./evaluation/prompt_optimization.py \
--task BBH_object_counting \
--tg_engine azure-gpt4o \
--model meta-llama/Meta-Llama-3.1-8B-Instruct \
--optimizer_version v2 \
--run_validation \
--num_threads 20 \

# Run optimization for llama3.1 8B on GSM8K
python -u ./evaluation/prompt_optimization.py \
--task GSM8K_DSPy \
--tg_engine azure-gpt4o \
--model meta-llama/Meta-Llama-3.1-8B-Instruct \
--optimizer_version v1 \
--run_validation \
--num_threads 20 \

python -u ./evaluation/prompt_optimization.py \
--task GSM8K_DSPy \
--tg_engine azure-gpt4o \
--model meta-llama/Meta-Llama-3.1-8B-Instruct \
--optimizer_version v1_momentum \
--run_validation \
--num_threads 20 \

python -u ./evaluation/prompt_optimization.py \
--task GSM8K_DSPy \
--tg_engine azure-gpt4o \
--model meta-llama/Meta-Llama-3.1-8B-Instruct \
--optimizer_version v2 \
--run_validation \
--num_threads 20 \

# Run optimization for gemini 1.5 pro on Object Counting
python -u ./evaluation/prompt_optimization.py \
--task BBH_object_counting \
--tg_engine azure-gpt4o \
--model gemini-1.5-pro \
--optimizer_version v1 \
--run_validation \
--num_threads 20 \

python -u ./evaluation/prompt_optimization.py \
--task BBH_object_counting \
--tg_engine azure-gpt4o \
--model gemini-1.5-pro \
--optimizer_version v1_momentum \
--run_validation \
--num_threads 20 \

python -u ./evaluation/prompt_optimization.py \
--task BBH_object_counting \
--tg_engine azure-gpt4o \
--model gemini-1.5-pro \
--optimizer_version v2 \
--run_validation \
--num_threads 20 \

# Run optimization for gemini 1.5 pro on GSM8K
python -u ./evaluation/prompt_optimization.py \
--task GSM8K_DSPy \
--tg_engine azure-gpt4o \
--model gemini-1.5-pro \
--optimizer_version v1 \
--run_validation \
--num_threads 20 \

python -u ./evaluation/prompt_optimization.py \
--task GSM8K_DSPy \
--tg_engine azure-gpt4o \
--model gemini-1.5-pro \
--optimizer_version v1_momentum \
--run_validation \
--num_threads 20 \

python -u ./evaluation/prompt_optimization.py \
--task GSM8K_DSPy \
--tg_engine azure-gpt4o \
--model gemini-1.5-pro \
--optimizer_version v2 \
--run_validation \
--num_threads 20 \
