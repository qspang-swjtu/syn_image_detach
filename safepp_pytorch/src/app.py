import io
import os
import torch
import uuid
import hashlib
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from contextlib import asynccontextmanager
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
# 导入你原有的模块
from data.transforms import build_val_transform, five_crop_tensor_views
from models.safepp import build_model
from utils.common import load_yaml

# 用于在全局存储模型和配置，以便在处理请求时复用
ml_components = {}

def resolve_device() -> torch.device:
    """自动选择设备"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 生命周期管理：在服务启动时加载模型，服务关闭时清理资源。
    这避免了每次请求都重新加载模型，大大提高响应速度。
    """
    os.makedirs("collected_data/real", exist_ok=True)
    os.makedirs("collected_data/fake", exist_ok=True)
    # ⚠️ 请在这里配置你的 yaml 和权重文件路径，或者后续改为从环境变量读取
    config_path = "/ai_paas_jf/pangqs/AIGC/safepp_pytorch/configs/stage3.yaml" 
    ckpt_path = "/ai_paas_jf/pangqs/AIGC/safepp_pytorch/model_wild_DF/stage2_merger_dedup1_v3_30epoch/best.pt"
    
    device = resolve_device()
    print(f"Loading config from {config_path}...")
    cfg = load_yaml(config_path)
    
    print(f"Loading model from {ckpt_path} to {device}...")
    model = build_model(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt['ema'] if 'ema' in ckpt else ckpt['model']
    model.load_state_dict(state, strict=True)
    model.eval()
    
    # 保存到全局字典中供路由使用
    ml_components["model"] = model
    ml_components["cfg"] = cfg
    ml_components["device"] = device
    
    print("Model loaded successfully! Ready to serve.")
    yield
    # 退出服务时的清理工作
    ml_components.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 创建 FastAPI 实例
app = FastAPI(title="SAFE++ Image Inference API", lifespan=lifespan)
# 跨域处理
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有前端地址访问（测试阶段最方便）
    allow_credentials=False,
    allow_methods=["*"],  # 允许所有请求方法 (POST, GET 等)
    allow_headers=["*"],  # 允许所有请求头
)
def predict_image_tensor(img: Image.Image, tta: int) -> float:
    """处理单张图片的推理逻辑（剥离了文件读取部分）"""
    model = ml_components["model"]
    cfg = ml_components["cfg"]
    device = ml_components["device"]
    
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available() and device.type == 'cuda'):
        if tta == 5:
            views = five_crop_tensor_views(img, cfg)
            x = torch.stack(views, dim=0).to(device)
            prob = torch.sigmoid(model(x)).max().item()
        else:
            transform = build_val_transform(cfg)
            x = transform(img).unsqueeze(0).to(device)
            prob = torch.sigmoid(model(x)).item()
    return float(prob)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    修改点 1：去除了 TTA 参数，后端现在默认使用 TTA = 5
    """
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file format. Error: {str(e)}")
        
    # 强制使用精准模式 (TTA 5)
    prob = predict_image_tensor(img, 5)
    pred = 1 if prob >= 0.5 else 0

    return {
        "filename": file.filename,
        "probability": prob,
        "prediction": pred,
        "prediction_name": "fake" if pred == 1 else "real",
        "tta": 5, 
        "device": str(ml_components["device"])
    }

@app.post("/feedback")
async def handle_feedback(
    file: UploadFile = File(...),
    correct_label: str = Form(..., description="图片的真实标签：'real' 或 'fake'")
):
    if correct_label not in ["real", "fake"]:
        raise HTTPException(status_code=400, detail="Invalid label.")

    try:
        # 1. 一次性读取图片的所有字节数据
        file_bytes = await file.read()
        
        # 2. 计算这些字节的 SHA-256 哈希值，作为该图片的唯一“数字指纹”
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        
        # 3. 提取文件后缀名
        _, ext = os.path.splitext(file.filename)
        if not ext:
            ext = ".png" # 默认后缀
            
        # 4. 使用哈希值作为文件名 (例如: abc123def456...789.png)
        unique_filename = f"{file_hash}{ext}"
        save_path = os.path.join("collected_data", correct_label, unique_filename)

        # 5. 关键判断：如果文件已经存在，说明之前已经保存过了，直接返回成功，不再重复写入
        if os.path.exists(save_path):
            return {
                "status": "success", 
                "message": "样本已存在，感谢您的反馈！(已忽略重复保存)", 
                "saved_to": save_path
            }

        # 6. 如果不存在，才将文件真正写入磁盘
        with open(save_path, "wb") as f:
            f.write(file_bytes)
            
        return {
            "status": "success", 
            "message": "Feedback collected successfully.", 
            "saved_to": save_path
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process feedback. {str(e)}")