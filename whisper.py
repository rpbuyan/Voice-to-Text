import torch
from transformers import BitsAndBytesConfig, pipeline

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)