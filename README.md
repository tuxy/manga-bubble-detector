# 🎈 Manga/Comics Speech Balloon Detection using YOLOv12  

## 📌 Overview  

This project is designed to train a **YOLOv12** model to detect speech balloons in **manga and comic images**. It includes:  
✅ Pre-processing scripts for dataset organization and label formatting.  
✅ A **training script** for model development.  
✅ An **inference script** for applying the trained model to new images.  

## 📂 Directory Structure  

```
backup\
└── balloon_training\
    └── weights\
        ├── best.pt
        └── last.pt
cfg\
└── balloon_yolo_config.yaml
dataset\
├── images\
│   ├── train\
│   ├── valid\
│   └── test\
└── labels\
    ├── train\
    ├── valid\
    └── test\
scripts\
├── inference.py
├── round_labels.py
└── split_dataset.py
README.md
train.py
requirements.txt
```

## 📄 File Descriptions  

### 🛠 `cfg/balloon_yolo_config.yaml`  

This **YAML configuration file** defines the dataset paths and class names for training. It includes:  
🔹 Relative dataset paths.  
🔹 Directories for **training, validation, and testing** images.  
🔹 The label class name.  

```yaml
path: "../dataset" # If you face issues, replace this with the absolute path to your dataset folder.
train: images/train
val: images/valid
test: images/test

names:
  0: balloon
```

---

### 🔍 `scripts/inference.py`  

**Purpose:**  
Runs inference on a folder of images using a trained YOLO model.  

**Usage:**  
```bash
python scripts/inference.py --weight <path_to_weights> --img_folder <path_to_images> --output_folder <output_folder_name>
```

**How It Works:**  
✅ Loads the YOLO model from the specified weights.  
✅ Iterates through images in the given folder.  
✅ Resizes images **based on height** and runs inference.  
✅ Saves output images **with bounding boxes** in the output folder.  

---

### 🏷 `scripts/round_labels.py`  

**Purpose:**  
Ensures label consistency by rounding numerical values to **four decimal places**.  

**Usage:**  
```bash
python scripts/round_labels.py <label_directory>
```

**How It Works:**  
✅ Scans the specified folder for **.txt** label files.  
✅ Rounds each **numerical value** (except class index) to **four decimal places**.  
✅ Overwrites the original files with rounded values.  

---

### 📂 `scripts/split_dataset.py`  

**Purpose:**  
Splits the dataset into **training, validation, and test** sets based on defined percentages.  

**Usage:**  
```bash
python scripts/split_dataset.py <dataset_path> --train_pct 70 --valid_pct 20 --test_pct 10
```

**How It Works:**  
✅ Checks that percentages **add up to 100**.  
✅ Creates the required **subdirectories** for images and labels.  
✅ Randomly shuffles images and assigns them to **train, validation, and test** sets.  
✅ Moves **corresponding label files** along with the images.  

---

### 🎯 `train.py`  

**Purpose:**  
The main script for training the YOLO model on the **speech balloon dataset**.  

**Usage:**  
```bash
python train.py --model <model_file> --data cfg/balloon_yolo_config.yaml --epochs 1000 --batch 16 --imgsz 640 --project backup --name balloon_training --cache ram
```

**How It Works:**  
✅ **Detects GPU availability** for optimized training.  
✅ Loads the YOLO model with pre-trained weights.  
✅ Trains using specified **epochs, batch size, image size, and caching method**.  
✅ Saves the best and latest weights in the **backup directory**.  

---

## 🛠 Installation & Requirements  

This project uses `requirements.txt` for dependency management. Install all required packages with:  

```bash
pip install -r requirements.txt
```

**Key dependencies:**  
- `torch`  
- `ultralytics`  
- `opencv-python`  
- `pillow`  

---

## 🔄 Usage Workflow  

### **1️⃣ Dataset Preparation**  
Use `scripts/split_dataset.py` to split your dataset into **training, validation, and test sets**.  

### **2️⃣ Label Processing**  
Run `scripts/round_labels.py` to ensure that all label values are **rounded** for consistency.  

### **3️⃣ Model Training**  
Execute `train.py` with the appropriate arguments to train your **YOLO model**.  

### **4️⃣ Run Inference**  
After training, apply `scripts/inference.py` to detect speech balloons in **new images**.  

---

## ℹ️ Need Help?  
If you need more details about **project setup, script functionalities, or troubleshooting**, feel free to ask! 🚀