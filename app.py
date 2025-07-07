import gradio as gr
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.hub.load(
    "bryandlee/animegan2-pytorch:main",
    "generator",
    pretrained="face_paint_512_v2",
    device=device
).eval()

def face2anime(image):
    image = image.resize((512, 512))
    tensor = to_tensor(image).unsqueeze(0).to(device) * 2 - 1
    out = model(tensor)[0].cpu()
    out = (out * 0.5 + 0.5).clamp(0, 1)
    return to_pil_image(out)

iface = gr.Interface(
    fn=face2anime,
    inputs=gr.Image(type="pil", label="Upload Your Photo"),
    outputs=gr.Image(type="pil", label="Anime Output"),
    title="AnimeGANv2 — Turn Your Photos into Anime!",
    theme="dark"
)

iface.launch()
