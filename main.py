import torch
from ultralytics import YOLO

def train_model():
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_available else "cpu")
    if cuda_available:
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not available. Using CPU.")

    model = YOLO("yolo12n.pt")
    model.model.to(device)

    results = model.train(
        data="cfg/balloon_yolo_config.yaml",
        epochs=1000,
        imgsz=640,
        project="backup",
        name="balloon_training",
        device=device,
        batch=16,
        workers=0
    )
    return results

if __name__ == "__main__":
    train_model()