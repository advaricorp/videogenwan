from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import StableDiffusion3Pipeline
import torch
from io import BytesIO
from fastapi.responses import Response

# Carga del modelo (hazlo una sola vez al inicio)
model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
pipe = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
).to("cuda")

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""

@app.post("/generate")
async def generate_image(req: PromptRequest):
    image = pipe(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        num_inference_steps=28,
        guidance_scale=7.0
    ).images[0]

    buf = BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/jpeg")
