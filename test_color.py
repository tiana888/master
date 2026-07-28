import torch
from PIL import Image
from pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from ip_adapter import IPAdapterPlusXL

# --- 配置路徑 ---
base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "models/image_encoder"
ip_ckpt = "sdxl_models/ip-adapter-plus_sdxl_vit-h.bin"
device = "cuda"

# 1. 載入模型
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    add_watermarker=False,
).to(device)

ip_model = IPAdapterPlusXL(
    pipe, image_encoder_path, ip_ckpt, device, num_tokens=16
)

# 2. 準備測試影像 (選一張鮮豔的彩色圖)
test_color_img = Image.open("assets/1.png").convert("RGB")

# 3. 執行「純色彩注入」測試
# 我們將所有解耦係數設為 0，subscale 設為 0
dummy_img = Image.new("RGB", (224, 224), (128, 128, 128)) # 中性灰

images = ip_model.generate(
    prompt="a house with trees, vibrant rich colors, high saturation",
    clr_ref_img=test_color_img,
    clr_texture_ref_img=test_color_img, # 暫時用同一張
    texture_ref_img=dummy_img,        # 給它一張假圖
    geometry_ref_img=dummy_img,       # 給它一張假圖
    color_scale=2.0,
    substract_scale=0.0,              # ⚠️ 先設為 0，看有沒有顏色
    geometry_scale=0.0,        # 不注入幾何
    texture_scale=0.0,         # 不注入紋理
    num_inference_steps=30,
    seed=42,
    guidance_scale=8.5,
)

images[0].save("test_pure_color_result.png")
print("純色彩注入測試完成，請檢查 test_pure_color_result.png")