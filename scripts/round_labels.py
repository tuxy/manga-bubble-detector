import os
import argparse

def round_labels(label_dir):
    if not os.path.exists(label_dir):
        raise ValueError("The specified directory does not exist.")

    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

    if not label_files:
        print("No label files found.")
        return

    for file in label_files:
        file_path = os.path.join(label_dir, file)
        
        with open(file_path, "r") as f:
            lines = f.readlines()

        rounded_lines = []
        for line in lines:
            parts = line.split()
            if len(parts) > 1:
                parts[1:] = [f"{float(num):.4f}" for num in parts[1:]]
            rounded_lines.append(" ".join(parts))

        with open(file_path, "w") as f:
            f.write("\n".join(rounded_lines) + "\n")

    print(f"Processed {len(label_files)} label files with four decimal places.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Round label file values to four decimal places.")
    parser.add_argument("label_dir", type=str, help="Path to the directory containing label files.")

    args = parser.parse_args()
    round_labels(args.label_dir)
