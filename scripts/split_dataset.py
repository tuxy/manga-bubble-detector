import os
import shutil
import random
import argparse

def split_dataset(dataset_path, train_pct=70, valid_pct=20, test_pct=10):
    if train_pct + valid_pct + test_pct != 100:
        raise ValueError("Train, validation, and test percentages must sum to 100.")

    # Define paths
    images_path = os.path.join(dataset_path, "images")
    labels_path = os.path.join(dataset_path, "labels")

    for split in ["train", "valid", "test"]:
        os.makedirs(os.path.join(images_path, split), exist_ok=True)
        os.makedirs(os.path.join(labels_path, split), exist_ok=True)

    all_images = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(all_images)

    total_images = len(all_images)
    train_count = int(total_images * (train_pct / 100))
    valid_count = int(total_images * (valid_pct / 100))
    test_count = total_images - train_count - valid_count

    train_files = all_images[:train_count]
    valid_files = all_images[train_count:train_count + valid_count]
    test_files = all_images[train_count + valid_count:]

    def move_files(file_list, split):
        for file in file_list:
            image_src = os.path.join(images_path, file)
            image_dest = os.path.join(images_path, split, file)

            label_file = file.rsplit(".", 1)[0] + ".txt"
            label_src = os.path.join(labels_path, label_file)
            label_dest = os.path.join(labels_path, split, label_file)

            shutil.move(image_src, image_dest)
            if os.path.exists(label_src):
                shutil.move(label_src, label_dest)

    move_files(train_files, "train")
    move_files(valid_files, "valid")
    move_files(test_files, "test")

    print(f"✅ Split completed! {train_count} images in train, {valid_count} in valid, {test_count} in test.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a dataset of images and labels into train, valid, and test sets.")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset directory.")
    parser.add_argument("--train_pct", type=int, default=70, help="Percentage for training set (default: 70).")
    parser.add_argument("--valid_pct", type=int, default=20, help="Percentage for validation set (default: 20).")
    parser.add_argument("--test_pct", type=int, default=10, help="Percentage for test set (default: 10).")

    args = parser.parse_args()
    split_dataset(args.dataset_path, args.train_pct, args.valid_pct, args.test_pct)
