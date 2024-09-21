import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.evaluate import Evaluate
import numpy as np
import random

# Set up the LM.
turbo = dspy.HFModel(model='meta-llama/Meta-Llama-3.1-8B-Instruct')
dspy.settings.configure(lm=turbo)

# Load math questions from the GSM8K dataset.
gsm8k = GSM8K()
gsm8k_trainset, gsm8k_devset, gsm8k_testset = gsm8k.train, gsm8k.dev, gsm8k.test
indices = np.arange(len(gsm8k_trainset))
np.random.seed(42)
random.seed(42)
np.random.shuffle(indices)
gsm8k_trainset = [gsm8k_trainset[i] for i in indices[:36]]


class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.prog(question=question)


# Set up the optimizer: we want to "bootstrap" (i.e., self-generate) 4-shot examples of our CoT program.
config = dict(max_bootstrapped_demos=8, num_candidate_programs=10)

# Optimize! Use the `gsm8k_metric` here. In general, the metric is going to tell the optimizer how well it's doing.
teleprompter = BootstrapFewShotWithRandomSearch(metric=gsm8k_metric, **config)
optimized_cot = teleprompter.compile(CoT(), trainset=gsm8k_trainset)

# Set up the evaluator, which can be used multiple times.
evaluate = Evaluate(devset=gsm8k_testset, metric=gsm8k_metric, num_threads=4, display_progress=True, display_table=0)

# Evaluate our `optimized_cot` program.
evaluate(optimized_cot)
