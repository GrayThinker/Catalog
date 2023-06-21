from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import torch

def run():
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        revision="fp16",
        torch_dtype=torch.float16,
    )

    input_image = Image.open("C:\Users\Joseph\Pictures\catalog\chairs.jpg")
    mask_image = Image.open("C:\Users\Joseph\Pictures\catalog\out.jpg")
    prompt = "Nice summer afternoon, charis on porch, realistic, beach"
    #image and mask_image should be PIL images.
    #The mask structure is white for inpainting and black for keeping as is
    pipe.to("cuda")
    image = pipe(prompt=prompt, image=input_image, mask_image=mask_image).images[0]
    image.save("./yellow_cat_on_park_bench.png")

print(torch.cuda.is_available())
run()