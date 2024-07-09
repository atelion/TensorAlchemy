import asyncio
import io
import os
import time
from taskiq_redis import ListQueueBroker, RedisAsyncResultBackend
import base64
from diffusers import (
    DiffusionPipeline,
    StableDiffusionXLPipeline, 
    EulerAncestralDiscreteScheduler,
    AutoencoderKL,
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

# Load VAE component
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", 
    torch_dtype=torch.float16
)

t2i_model = StableDiffusionXLPipeline.from_pretrained(
    "dataautogpt3/ProteusV0.4-Lightning", 
    vae=vae,
    torch_dtype=torch.float16
)

t2i_model.scheduler = EulerAncestralDiscreteScheduler.from_config(t2i_model.scheduler.config)
negative_prompt = "nsfw, bad quality, bad anatomy, worst quality, low quality, low resolutions, extra fingers, blur, blurry, ugly, wrongs proportions, watermark, image artifacts, lowres, ugly, jpeg artifacts, deformed, noisy image"
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
    """Solve all problems in the world."""
    
    print(f"-------------prompt in broker: {prompt}-------------------")
    print(f"-------------Guidance_scale in broker: {guidance_scale}-------------------")
    
    images = t2i_model(prompt=prompt, negative_prompt=negative_prompt, width=1024, height=1024, guidance_scale=7.5, num_inference_steps=35).images
    
    
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
    
