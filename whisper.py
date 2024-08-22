import torch
from transformers import BitsAndBytesConfig, pipeline

import whisper
import gradio as gr
import time
import warnings
import os
from gtts import gTTS
from PIL import Image

import nltk
from nltk import sent_tokenize

nltk.download('punkt')

class ModelConfig:
    def __init__(self):
        self.model_id = "llava-hf/llava-1.5-7b-hf"
        if torch.cuda.is_available():
            self.quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            self.pipe = pipeline(
                "image-to-text",
                model=self.model_id,
                model_kwargs={"quantization_config": self.quant_config}
            )
        else:
            self.pipe = pipeline("image-to-text", model=self.model_id)
        self.image = Image.open("ngolokante.png")
        
    def generate_text(self):
        return self.pipe(self.image, padding=True)

class nltkConfig:
    def __init__(self, image, pipe):
        self.max_new_tokens = 250
        self.prompt_instructions = """
        Describe the image thorougly and using as much detail as possible. You are a helpful AI assistant who is able to answer
        questions about the image. Now generate the helpful answer: what is the image all about?
        """
        self.prompt = "User: <image>\n" + self.prompt_instructions + "\nAssistant:"
        self.outputs = pipe(image, prompt=self.prompt, generate_kwargs={"max_new_tokens": self.max_new_tokens})

    def get_text(self):
        for sent in sent_tokenize(self.outputs[0]["generated_text"]):
            print(sent)

if __name__ == "__main__":
    modelconfig = ModelConfig()
    nltkconfig = nltkConfig(modelconfig.image, modelconfig.pipe)
    nltkconfig.get_text()