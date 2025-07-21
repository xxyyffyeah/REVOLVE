# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

REVOLVE is an optimization framework that enhances the stability and efficiency of AI system optimization by tracking the evolution of model responses across iterations. Building on textual feedback from LLMs, REVOLVE simulates higher-order optimization effects, ensuring adjustments are guided not only by immediate feedback but also by the model's performance trajectory.

**Key Features:**
- Built upon TextGrad foundation with enhanced response evolution tracking
- Supports prompt optimization, solution refinement, and code optimization
- Provides intuitive API for custom optimization tasks and loss functions
- More stable and efficient than traditional derivative-based methods

**Core Components:**
- **revolve/**: Core REVOLVE framework (autograd, engines, optimizers, tasks, variables)
- **dsp/** and **dspy/**: DSPy framework integration for advanced prompt engineering
- **evaluation/**: Comprehensive evaluation scripts and benchmarks

## Installation

**For Users:**
```bash
pip install revolve
```

**For Development:**
```bash
pip install -r requirements.txt
pip install -e .
export PYTHONPATH=$(pwd)
```

## Core Evaluation Commands

### Solution Optimization
Tests the optimizer's ability to improve solution quality across different datasets:

```bash
# GPQA Diamond dataset
python evaluation/solution_optimization.py --task GPQA_diamond --engine gpt-4o --num_threads 10 --optimizer_version v2

# MMLU Machine Learning  
python evaluation/solution_optimization.py --task MMLU_machine_learning --engine gpt-4o --num_threads 10 --optimizer_version v2

# MMLU College Physics
python evaluation/solution_optimization.py --task MMLU_college_physics --engine gpt-4o --num_threads 10 --optimizer_version v2
```

### Prompt Optimization
Requires two LLM specifications:
- `--backbone_engine`: LLM used by REVOLVE for optimization process
- `--model`: Target LLM being optimized

```bash
# BBH Object Counting
python evaluation/prompt_optimization.py --task BBH_object_counting --backbone_engine gpt-4o --model gpt-3.5-turbo --num_threads 10 --optimizer_version v2

# GSM8K with DSPy
python evaluation/prompt_optimization.py --task GSM8K_DSPy --backbone_engine gpt-4o --model gpt-3.5-turbo --num_threads 10 --optimizer_version v2
```

### Code Optimization
Requires external leetcode-hard-gym dependency:

```bash
# Setup (one-time)
git clone https://github.com/GammaTauAI/leetcode-hard-gym.git && cd leetcode-hard-gym
python -m pip install -e .
cd ..

# Run evaluation
python ./evaluation/code_optimization/leetcode_testtime_with_supervision.py --engine meta-llama/Meta-Llama-3.1-70B-Instruct --optimizer_version v2 --size 200
```

## Optimizer Versions

- **v1**: Original TextGrad - optimizes based on textual feedback
- **v1_momentum**: Momentum-TextGrad - adjusts optimization using feedback trends across iterations  
- **v2**: REVOLVE method - tracks response evolution over time for more stable and efficient optimization

Use `--optimizer_version` flag to select the desired method.

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_basics.py -v
```

## Batch Evaluation Scripts

```bash
# Comprehensive prompt optimization evaluations
bash evaluation/scripts/prompt_opt.sh
bash evaluation/scripts/prompt_opt_llama70b.sh
```

## Architecture Details

### REVOLVE Core (`revolve/`)
- **autograd/**: Automatic differentiation for textual gradients with multimodal support
- **engine/**: Language model backends (OpenAI, Anthropic, Cohere, Gemini, local models via vLLM/Together)
- **optimizer/**: TextualGradientDescent variants including momentum and evolution tracking
- **tasks/**: Evaluation benchmarks (BigBench Hard, GPQA, GSM8K, LeetCode, MMLU, multimodal tasks)
- **variable.py**: Core Variable class for computational graph tracking

### DSPy Integration
- Advanced prompt engineering and program synthesis capabilities
- Retrieval-augmented generation modules
- Teleprompt optimization techniques

## Environment Configuration

Set appropriate API keys for LLM providers:
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"  
export GOOGLE_API_KEY="your-key"
export COHERE_API_KEY="your-key"
```

## Related Work References

This project builds upon:
- **TextGrad**: Foundation for LLM-based gradient pipelines
- **DSPy**: Pioneering framework for leveraging LMs in diverse applications
- **ProTeGi**: Inspiration for textual gradients concept
- **Reflexion**: Self-reflection framework demonstrating text-based optimization

## Important Notes

- Always set `PYTHONPATH=$(pwd)` when running evaluation scripts from repository root
- Logs are written to `./logs/` directory in JSON format
- Use `--num_threads` parameter to control evaluation parallelization
- Code optimization evaluation requires leetcode-hard-gym to be cloned separately
- CUDA 11.2+ recommended for optimal performance
- Python 3.9+ required
- All codes should be written in English 

EOF < /dev/null