import asyncio
import io
import os
import time
from taskiq_redis import ListQueueBroker, RedisAsyncResultBackend
import base64
from diffusers import (
    DiffusionPipeline,
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
    DPMSolverMultistepScheduler,
)
from PIL import Image
import torch
import ImageReward as reward
import random
import redis
import argparse
from transformers import CLIPImageProcessor
import pathlib, sys


redis_async_result = RedisAsyncResultBackend(
    redis_url="redis://localhost:6379",
)

# Or you can use PubSubBroker if you need broadcasting
broker = ListQueueBroker(
    url="redis://localhost:6379",
    result_backend=redis_async_result,
)
prompts = [
    "A serene forest with ancient trees and a carpet of bluebells.",    
]
result = []
def hash_function(input_string: str):
    """
    A simple hash function that converts a string input into an integer hash value.
    
    Args:
        input_string (str): The input string to be hashed.
    
    Returns:
        int: The hash value of the input string.
    """
    hash_value = 0
    for char in input_string:
        hash_value = (hash_value * 31 + ord(char)) % 2**32
    return hash_value

model_path_list = ["stabilityai/stable-diffusion-xl-base-1.0", "RunDiffusion/Juggernaut-XL-v9"]
lora_path_list = ["checkpoint-5000" ,"Xrunner/dpo-juggernautxl"]
t2i_model = DiffusionPipeline.from_pretrained(model_path_list[1], torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
print("SDXL model loaded")
t2i_model.load_lora_weights(lora_path_list[1], weight_name="pytorch_lora_weights.safetensors", adapter_name="imagereward-lora")
t2i_model.set_adapters(["imagereward-lora"], adapter_weights=[0.9])
print("Lora model loaded successfully.")
scoring_model = reward.load("ImageReward-v1.0")


def pil_image_to_base64(image: Image.Image, format="JPEG") -> str:
    if format not in ["JPEG", "PNG"]:
        format = "JPEG"
    image_stream = io.BytesIO()
    image = image.convert("RGB")
    image.save(image_stream, format=format)
    base64_image = base64.b64encode(image_stream.getvalue()).decode("utf-8")
    return base64_image

@broker.task
async def generate_image(prompt: str, guidance_scale: float, num_inference_steps: int):
    start_time = time.time()
    t2i_model.to("cuda")
    
    global result
    # t2i_model.to("cuda")
    """Solve all problems in the world."""
    
    print(f"-------------prompt in broker: {prompt}-------------------")
    print(f"-------------Guidance_scale in broker: {guidance_scale}-------------------")
    _prompt = prompt.strip(" .") + " in cartoony anime-like style."
    # _prompt = prompt
    images = t2i_model(prompt=_prompt, num_inference_steps=35, guidance_scale=7.5).images
    
    
    end_time = time.time()

    print(f"Successfully generated images in {end_time-start_time} seconds.")
    
    score = scoring_model.score(prompt, images)
    # images[0].save(f"{score}-{prompt}.png")
    # Note: encode <class 'PIL.Image.Image'>
    base64_image = pil_image_to_base64(images[0])
    print("All problems are solved!")
    # return images, score
    return {"prompt": prompt, "score": score, "image": base64_image}

if __name__ == "__main__":
    # Add the base repository to the path so the miner can access it
    file_path = str(
        pathlib.Path(__file__).parent.parent.parent.parent.resolve(),
    )
    if file_path not in sys.path:
        sys.path.append(file_path)

    # Import StableMiner after fixing path
    from miner import StableMiner
    broker.startup()
    # Start the miner
    StableMiner()
    
