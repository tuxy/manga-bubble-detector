import os
import random

# Diretório onde as imagens estão
dataset_path = "img_mangas/dataset/images"

# Obtém todas as imagens
images = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(".jpg")]

# Embaralha os dados para evitar viés
random.shuffle(images)

# Define a proporção (80% treino, 20% validação)
split = int(0.8 * len(images))
train_images = images[:split]
valid_images = images[split:]

# Salva os arquivos
with open("img_mangas/dataset/train.txt", "w") as f:
    f.write("\n".join(train_images))

with open("img_mangas/dataset/valid.txt", "w") as f:
    f.write("\n".join(valid_images))

print("Arquivos train.txt e valid.txt gerados com sucesso!")
