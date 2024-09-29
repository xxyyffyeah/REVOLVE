export PYTHONPATH=$(pwd)

# Run optimization for llama3.1 8B on Object Counting using llama3.1 70B as the optimization engine
python -u ./evaluation/prompt_optimization.py \
--task BBH_object_counting \
--tg_engine meta-llama/Llama-3.1-70B-Instruct \
--model meta-llama/Meta-Llama-3.1-8B-Instruct \
--optimizer_version v1 \
--run_validation \
--num_threads 20 \

python -u ./evaluation/prompt_optimization.py \
--task BBH_object_counting \
--tg_engine meta-llama/Llama-3.1-70B-Instruct \
--model meta-llama/Meta-Llama-3.1-8B-Instruct \
--optimizer_version v1_momentum \
--run_validation \
--num_threads 20 \

python -u ./evaluation/prompt_optimization.py \
--task BBH_object_counting \
--tg_engine meta-llama/Llama-3.1-70B-Instruct \
--model meta-llama/Meta-Llama-3.1-8B-Instruct \
--optimizer_version v2 \
--run_validation \
--num_threads 20 \

# Run optimization for llama3.1 8B on GSM8K using llama3.1 70B as the optimization engine
python -u ./evaluation/prompt_optimization.py \
--task GSM8K_DSPy \
--tg_engine meta-llama/Llama-3.1-70B-Instruct \
--model meta-llama/Meta-Llama-3.1-8B-Instruct \
--optimizer_version v1 \
--run_validation \
--num_threads 20 \

python -u ./evaluation/prompt_optimization.py \
--task GSM8K_DSPy \
--tg_engine meta-llama/Llama-3.1-70B-Instruct \
--model meta-llama/Meta-Llama-3.1-8B-Instruct \
--optimizer_version v1_momentum \
--run_validation \
--num_threads 20 \

python -u ./evaluation/prompt_optimization.py \
--task GSM8K_DSPy \
--tg_engine meta-llama/Llama-3.1-70B-Instruct \
--model meta-llama/Meta-Llama-3.1-8B-Instruct \
--optimizer_version v2 \
--run_validation \
--num_threads 20 \
