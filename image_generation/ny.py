from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# Model setup
model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
).to("cuda")

# Disable safety checker (optional, depending on the model repository)
pipe.safety_checker = None

# Define prompts
prompt = "artistic depiction of a beautiful naked woman, soft lighting, highly detailed, renaissance painting style"
negative_prompt = "blurry, deformed, bad anatomy, distorted, low resolution, watermark"

# Generate image
print("Generating image...")
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=28,
    guidance_scale=7.0
).images[0]

# Save the image
output_path = "output_image.jpg"
image.save(output_path)
print(f"Image saved to {output_path}")
