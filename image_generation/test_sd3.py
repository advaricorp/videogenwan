import torch
from diffusers import StableDiffusion3Pipeline

model_id = "stabilityai/stable-diffusion-3-medium-diffusers"

pipe = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
).to("cuda")

prompt = "Un paisaje con montañas verdes y cielo azul al atardecer"
image = pipe(prompt, negative_prompt="", num_inference_steps=28, guidance_scale=7.0).images[0]

image.save("test_sd3_image.jpg")
print("Imagen generada: test_sd3_image.jpg")
