import torch
import argparse
from ultralytics import YOLO

def train_model(model_name, data_config, epochs, batch_size, img_size, project_name, run_name, cache):
    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"

    if cuda_available:
        num_gpus = torch.cuda.device_count()
        print(f"🚀 Training on {num_gpus} GPU(s): {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")
    else:
        print("⚠️ WARNING: CUDA not available. Training on CPU.")

    model = YOLO(model_name)

    results = model.train(
        data=data_config,
        epochs=epochs,
        imgsz=img_size,
        project=project_name,
        name=run_name,
        device=device,
        batch=batch_size,
        workers=min(batch_size, 8),
        amp=True,
        cache=cache,
        verbose=True
    )

    print("\n✅ Training completed!")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO Model")
    parser.add_argument("--model", type=str, default="yolo12n.pt", help="YOLO model file (e.g., yolo12n.pt)")
    parser.add_argument("--data", type=str, default="cfg/balloon_yolo_config.yaml", help="Path to data configuration YAML file")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for training")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    parser.add_argument("--project", type=str, default="backup", help="Project name for saving training results")
    parser.add_argument("--name", type=str, default="balloon_training", help="Run name for this training session")
    parser.add_argument("--cache", type=str, default="ram", choices=["ram", "disk", "none"], help="Cache method (ram, disk, or none)")

    args = parser.parse_args()
    
    train_model(
        model_name=args.model,
        data_config=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        project_name=args.project,
        run_name=args.name,
        cache=args.cache
    )