import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# ... [A te eredeti kódod eleje marad változatlan] ...

# Feltételezem, hogy a következő változóid vannak:
# - train_images: numpy tömb alakú, mérete [képek száma, 128, 128, 3]
# - train_image_ids: az egyes képekhez tartozó ID-k listája
# - data_array: numpy tömb, amely az ID-ket és a címkéket tartalmazza

# 1. Adat augmentációs transzformációk definiálása
data_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
])

# 2. Képek augmentálása és adathalmaz bővítése
N = 5  # Minden képből 5 augmentált változatot készítünk

# Képek, címkék és ID-k listáinak inicializálása
images_list = []
labels_list = []
IDs_list = []

# ID-k és címkék összerendelése
id_to_label = {row[0]: row[1] for row in data_array}

# Eredeti képek és címkék hozzáadása a listákhoz
for img, img_id in zip(train_images, train_image_ids):
    label = id_to_label[img_id]
    images_list.append(img)
    labels_list.append(label)
    IDs_list.append(img_id)

# Augmentáció alkalmazása minden képre
for img, label, img_id in zip(train_images, labels_list, IDs_list):
    # Kép átalakítása PIL formátumra
    img_uint8 = (img * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    for _ in range(N):
        # Augmentáció alkalmazása
        augmented_img = data_augmentation(pil_img)
        # Visszaalakítás numpy tömbbé és normalizálás 0-1 közé
        augmented_img_np = np.array(augmented_img).astype(np.float32) # / 255.0
        images_list.append(augmented_img_np)
        labels_list.append(label)
        IDs_list.append(img_id)  # Ugyanazt az ID-t adjuk hozzá

# Az augmentált adathalmaz konvertálása numpy tömbbé
train_images_augmented = np.array(images_list)
labels_array = np.array(labels_list)
train_image_ids_augmented = IDs_list

print(f"Eredeti képek száma: {len(train_images)}")
print(f"Képek száma augmentáció után: {len(train_images_augmented)}")

# 3. Átlag és szórás újraszámítása az augmentált adathalmazra
train_mean = train_images_augmented.mean(axis=(0, 1, 2))
train_std = train_images_augmented.std(axis=(0, 1, 2))

print(f'Calculated mean for train images: {train_mean}')
print(f'Calculated std for train images: {train_std}')

# 4. Transzformációk frissítése
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=train_mean.tolist(), std=train_std.tolist())
])

# 5. Egyedi Dataset osztály módosítása
class CustomImageDataset(Dataset):
    def __init__(self, images, labels, ids, transform=None):
        self.images = images
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        img_id = self.ids[idx]

        # Kép átalakítása PIL formátumra
        img_uint8 = (img * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)

        if self.transform:
            img = self.transform(pil_img)

        return img, label  # Az img_id-t is visszaadhatjuk, ha szükséges

# 6. Adathalmaz és DataLoader létrehozása
dataset = CustomImageDataset(images=train_images_augmented, labels=labels_array, ids=train_image_ids_augmented, transform=transform_train)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# 7. Modell betöltése és tréning
# A modell betöltése és a tréning ciklus ugyanaz marad

# ... [A tréning kódod változatlan marad] ...

# 8. Validációs adathalmaz létrehozása (ha szükséges)
# Ha szeretnél validációs adathalmazt, szétválaszthatod az adataidat tréning és validációs halmazra

# 9. Kiíratás és modell értékelése
# Az értékelés során használhatod az eredeti teszt adataidat

# ... [A kiíratási kódod változatlan marad] ...
