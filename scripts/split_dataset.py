import os
import shutil
import random
import argparse

def split_dataset(dataset_path, train_pct=70, valid_pct=20, test_pct=10):
    if train_pct + valid_pct + test_pct != 100:
        raise ValueError("Train, validation, and test percentages must sum to 100.")

    images_path = os.path.join(dataset_path, "images")
    train_path = os.path.join(images_path, "train")
    valid_path = os.path.join(images_path, "valid")
    test_path = os.path.join(images_path, "test")

    for folder in [train_path, valid_path, test_path]:
        os.makedirs(folder, exist_ok=True)

    all_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(all_files)

    total_files = len(all_files)
    train_count = int(total_files * (train_pct / 100))
    valid_count = int(total_files * (valid_pct / 100))
    test_count = total_files - train_count - valid_count

    train_files = all_files[:train_count]
    valid_files = all_files[train_count:train_count + valid_count]
    test_files = all_files[train_count + valid_count:]

    def move_files(file_list, destination):
        for file in file_list:
            file_path = os.path.join(images_path, file)
            label_path = os.path.join(images_path, file.replace(file.split('.')[-1], 'txt'))
            
            shutil.move(file_path, os.path.join(destination, file))
            if os.path.exists(label_path):
                shutil.move(label_path, os.path.join(destination, os.path.basename(label_path)))

    move_files(train_files, train_path)
    move_files(valid_files, valid_path)
    move_files(test_files, test_path)

    print(f"Split completed! {train_count} images in train, {valid_count} in valid, {test_count} in test.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a dataset of images into train, valid, and test sets.")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset directory.")
    parser.add_argument("--train_pct", type=int, default=70, help="Percentage for training set (default: 70).")
    parser.add_argument("--valid_pct", type=int, default=20, help="Percentage for validation set (default: 20).")
    parser.add_argument("--test_pct", type=int, default=10, help="Percentage for test set (default: 10).")

    args = parser.parse_args()
    split_dataset(args.dataset_path, args.train_pct, args.valid_pct, args.test_pct)
