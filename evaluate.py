import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM


class Evaluator:
    def __init__(self, config, tokenizer) -> None:
        
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name)
        