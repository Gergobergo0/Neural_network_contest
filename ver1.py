

# KEPEK + EXCEL beolvasas


# Képek importálása és átalakítása -----------------------------------------
# Elsődleges csomagok importálása
import torch
import numpy as np
import matplotlib.image as mpimg
from skimage.transform import resize
import glob
import os
from torchvision import transforms

# Seed beállítása a reprodukálhatósághoz
SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)

# Kép fájlok keresése
train_image_files = glob.glob('../Neural_network_contest/train_data/*.png')  # Cseréld ki a mappát, ahol a képek vannak
test_image_files = glob.glob('../Neural_network_contest/test_data/*.png')    # Test képek


# Tárolók az adatokhoz
train_data_dict = {}
test_data_dict = {}


# Fájlok beolvasása és csoportosítása ID és típus alapján
for image_path in train_image_files:
    file_name = os.path.splitext(os.path.basename(image_path))[0]

    if "__" in file_name:
        id_part = file_name.rsplit('_', 1)[0]  # itt tárolja az ID-t
        type_part = file_name.split('_')[-1]  # itt tárolja a típusát : maszk, fázis , amplitúdó

        if type_part in ["amp", "mask", "phase"]:
            if id_part not in train_data_dict:
                train_data_dict[id_part] = {'amp': None, 'mask': None, 'phase': None}

            if train_data_dict[id_part][type_part] is None:
                img = mpimg.imread(image_path)

                if img.shape[:2] != (128, 128):
                    img = resize(img, (128, 128), anti_aliasing=True)

                train_data_dict[id_part][type_part] = img

# Test képek beolvasása és csoportosítása ID és típus alapján
for image_path in test_image_files:
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    if "__" in file_name:
        test_id_part = file_name.rsplit('_', 1)[0]  # ID-t tárolja
        test_type_part = file_name.split('_')[-1]   # Típust tárolja: maszk, fázis, amplitúdó

        if test_type_part in ["amp", "mask", "phase"]:
            if test_id_part not in test_data_dict:
                test_data_dict[test_id_part] = {'amp': None, 'mask': None, 'phase': None}

            if test_data_dict[test_id_part][test_type_part] is None:
                img = mpimg.imread(image_path)
                if img.shape[:2] != (128, 128):
                    img = resize(img, (128, 128), anti_aliasing=True)
                test_data_dict[test_id_part][test_type_part] = img


# Adatok numpy tömbbe konvertálása
train_image_list = []
train_image_ids = []
for id_key, img_types in train_data_dict.items():
    if img_types['amp'] is not None and img_types['mask'] is not None and img_types['phase'] is not None:
        image_stack = np.stack([img_types['amp'], img_types['mask'], img_types['phase']], axis=-1)
        train_image_list.append(image_stack)
        train_image_ids.append(id_key)

test_image_list = []
test_image_ids = []
for test_id_part, test_type_part in test_data_dict.items():
    if test_type_part['amp'] is not None and test_type_part['mask'] is not None and test_type_part['phase'] is not None:
        image_stack = np.stack([test_type_part['amp'], test_type_part['mask'], test_type_part['phase']], axis=-1)
        test_image_list.append(image_stack)
        test_image_ids.append(test_id_part)



# Végső 4D tömb (képek száma, 128, 128, 3)
train_images = np.array(train_image_list)
test_images = np.array(test_image_list)

# Számláló a törölt elemekhez ---> KITÖRLI AMIBEN NINCS : phase + amp + mask  !!!!!!!!!!!!!!!!!
deleted_count_train = 0
deleted_count_test = 0

# Az ID-k másolata az iterációhoz, hogy ne módosítsuk közben az eredeti dictionary-t
for id_key in list(train_data_dict.keys()):
    img_types = train_data_dict[id_key]

    if img_types['amp'] is None or img_types['mask'] is None or img_types['phase'] is None:
        del train_data_dict[id_key]
        deleted_count_train += 1

for test_id_part in list(test_data_dict.keys()):
    img_types = test_data_dict[test_id_part]

    if test_type_part['amp'] is None or test_type_part['mask'] is None or test_type_part['phase'] is None:
        del test_data_dict[test_id_part]
        deleted_count_test += 1
print(f"Törölt elemek száma a train-ben: {deleted_count_train}")
print(f"Törölt elemek száma a test-ben: {deleted_count_test}")
print(f"Maradék teljes elemek száma a train-ben: {len(train_data_dict)}")
print(f"Maradék teljes elemek száma a test-ben: {len(test_data_dict)}")

# Kép normalizálás: skálázás 0-1 közé
# train_images = train_images / 255.0
# test_images = test_images / 255.0

# Átlag és szórás kiszámolása mindkét halmazra
train_mean = train_images.mean()
train_std = train_images.std()
test_mean = test_images.mean()
test_std = test_images.std()

print(f'Calculated mean for train images: {train_mean}')
print(f'Calculated std for train images: {train_std}')
print(f'Calculated mean for test images: {test_mean}')
print(f'Calculated std for test images: {test_std}')


# --------------------------------------   Data augmentation   ---------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
from PIL import Image








# Alkalmazzuk a transzformációkat: Normalizálás mindkét adathalmazra
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[train_mean], std=[train_std])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[test_mean], std=[test_std])
])

train_image_tensors = []
for img in train_images:
    transformed_img = transform_train(img)
    train_image_tensors.append(transformed_img)
train_image_tensors = torch.stack(train_image_tensors)
print(f"train_image_tensors : {train_image_tensors.shape}")  # [db, type, x, y]

test_image_tensors = []
for img in test_images:
    transformed_img = transform_test(img)
    test_image_tensors.append(transformed_img)
test_image_tensors = torch.stack(test_image_tensors)
print(f"test_image_tensors : {test_image_tensors.shape}")  # [db, type, x, y]


# EXCEL BEOLVASÁS --------------------------------------------------------------
import pandas as pd
import numpy as np
file_path = 'data_labels_train.csv'
df = pd.read_csv(file_path)
# Csak a szükséges oszlopok kiválasztása
selected_data = df[['filename_id', 'defocus_label']]
# Átalakítás numpy tömbbé
data_array = selected_data.to_numpy()
# Ellenőrzés
print(f"data_array.shape : {data_array.shape}")
# EXCEL BEOLVASÁS --------------------------------------------------------------
file_path = 'example_solutions.csv'
df = pd.read_csv(file_path)
selected_data = df[['Id', 'Expected']]
example_solutions = selected_data.to_numpy()
print(f"example_solutions.shape : {example_solutions.shape}")
















# --------------------------------------   ALAP BEALLÍTÁS  -------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------import torch




print("Eredeti címke:", data_array[:, 1])
# erre azért van szükség, mert CrossEntrophy 0-vmennyi számokat vár
unique_labels = np.unique(data_array[:, 1])
label_map = {label: idx for idx, label in enumerate(unique_labels)}
mapped_labels = np.array([label_map[label] for label in data_array[:, 1]])
# data_array címkék frissítése a mapped_labels segítségével
data_array[:, 1] = mapped_labels
print("Átalakított címke:", data_array[:, 1])











# --------------------------------------   BETANITAS  ------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


# Egyedi dataset osztály
class CustomImageDataset(Dataset):
    def __init__(self, images, image_ids, data_array, transform=None):
        self.images = images
        self.image_ids = image_ids
        self.data_dict = {row[0]: row[1] for row in data_array}  # ID-k és címkék
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img = self.images[idx]
        img_id = self.image_ids[idx]
        label = self.data_dict[img_id]  # címke hozzárendelése az ID alapján

        if self.transform:
            img = self.transform(img)

        return img, label


# Dataset betöltés
dataset = CustomImageDataset(images=train_image_tensors, image_ids=train_image_ids, data_array=data_array)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Modell betöltése
from model import MobileNetV2Custom

num_classes = len(np.unique(data_array[:, 1]))
model = MobileNetV2Custom(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(dev)

# Betanítás validációval
num_epochs = 30
for epoch in range(num_epochs):
    # Tréning szakasz
    model.train()
    train_loss = 0.0
    for train_images, labels in train_loader:
        train_images, labels = train_images.to(dev), labels.to(dev).long()

        optimizer.zero_grad()
        outputs = model(train_images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_loss /= len(train_loader)

    # Validáció szakasz
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for img_id in train_image_ids:
            img_idx = train_image_ids.index(img_id)
            img = train_image_tensors[img_idx].unsqueeze(0).to(dev)
            label = data_array[data_array[:, 0] == img_id, 1][0]  # helyes címke

            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            if predicted.item() == label:
                correct += 1
            total += 1

    val_accuracy = correct / total
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    scheduler.step()











# --------------------------------------   KIIRATAS  -------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
from evaluate_and_export import evaluate_model
output_file = 'solution.csv'
evaluate_model(model, test_image_tensors, test_image_ids, label_map, output_file, dev)

