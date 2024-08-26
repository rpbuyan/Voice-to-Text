import torch
from transformers import BitsAndBytesConfig, pipeline

import whisper
import gradio as gr
import time
import datetime
import warnings
import os
from gtts import gTTS
from PIL import Image

import nltk
from nltk import sent_tokenize

import numpy as np

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
    
    def cuda_config(self):
        torch.cuda.is_available()
        local_gpu = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("medium", device=local_gpu)

    def history_log(self):
        timestamp = str(datetime.datetime.now()).replace(" ", "-")
        logfile = f"log_{timestamp}.txt"
        with open(logfile, "a", encodings='utf-8') as f:
            f.write("Log file created at: " + timestamp + "\n")
            f.write("Model: " + self.pipe + "\n")
            f.write("Outputs: " + self.pipe(self.image, padding=True) + "\n")
            f.close()
            

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

class gradioConfig:
    def __init__(self, modelconfig):
        self.modelconfig = modelconfig
        self.image = gr.inputs.Image()
        self.outputs = gr.outputs.Textbox()
        self.title = "Whisper"
        self.description = "An AI assistant that generates a helpful answer about an image."
        self.examples = [
            ["ngolokante.png"]
        ]
        self.app = gr.Interface(
            modelconfig.generate_text,
            self.image,
            self.outputs,
            title=self.title,
            description=self.description,
            examples=self.examples
        )
        self.app.launch()

if __name__ == "__main__":
    modelconfig = ModelConfig()
    nltkconfig = nltkConfig(modelconfig.image, modelconfig.pipe)
    nltkconfig.get_text()
    warnings.filterwarnings("ignore")  # ignore built-in warnings
    gradioconfig = gradioConfig(modelconfig)
