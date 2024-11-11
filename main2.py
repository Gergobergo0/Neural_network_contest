


#   Itt tárolom azokat a kiegészítő elemeket ami validációhoz és data augtationhoz kell
#       Ezek még nem működőképesek....




num_epochs = 5
tran_batch_size = 16


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
train_images = train_images / 255.0
test_images = test_images / 255.0














# --------------------------------------   AUGMENTÁCIÓ  ----------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
from PIL import Image
data_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
])
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
        augmented_img_np = np.array(augmented_img).astype(np.float32)  / 255.0
        images_list.append(augmented_img_np)
        labels_list.append(label)
        IDs_list.append(img_id)  # Ugyanazt az ID-t adjuk hozzá
# Az augmentált adathalmaz konvertálása numpy tömbbé
train_images_augmented = np.array(images_list)
labels_array = np.array(labels_list)
train_image_ids_augmented = IDs_list
print(f"Eredeti képek száma: {len(train_images)}")
print(f"Képek száma augmentáció után: {len(train_images_augmented)}")






















# Átlag és szórás kiszámolása mindkét halmazra
train_mean = train_images.mean()
train_std = train_images.std()
test_mean = test_images.mean()
test_std = test_images.std()
#train_mean = train_images.mean(axis=(0, 1, 2))  # AUG
#train_std = train_images.std(axis=(0, 1, 2))
#test_mean = test_images.mean(axis=(0, 1, 2))
#test_std = test_images.std(axis=(0, 1, 2))
print(f'Calculated mean for train images: {train_mean}')
print(f'Calculated std for train images: {train_std}')
print(f'Calculated mean for test images: {test_mean}')
print(f'Calculated std for test images: {test_std}')


# --------------------------------------   Data augmentation   ---------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
from PIL import Image








# Alkalmazzuk a transzformációkat: Normalizálás mindkét adathalmazra
"""
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[train_mean], std=[train_std])
])
"""

# AUG
 #4. Transzformációk frissítése
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=train_mean.tolist(), std=train_std.tolist())
])


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[test_mean], std=[test_std])
])




train_image_tensors = []
for img in train_images_augmented:
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
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image



# Data augmentation transzformációk
augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Normalize(mean=train_mean.tolist(), std=train_std.tolist())
])



# Augmentált adatokat tartalmazó osztály
class AugmentedImageDataset(Dataset):
    def __init__(self, images, image_ids, data_array, transform=None, augmentations=None):
        self.images = images
        self.image_ids = image_ids
        self.data_dict = {row[0]: row[1] for row in data_array}  # ID és címkék
        self.transform = transform
        self.augmentations = augmentations
    def __len__(self):
        return len(self.image_ids)
    def __getitem__(self, idx):
        img = self.images[idx]
        img_id = self.image_ids[idx]
        label = self.data_dict[img_id]
        # If img is not a tensor, apply the initial transform (e.g., ToTensor)
        if not isinstance(img, torch.Tensor) and self.transform:
            img = self.transform(img)
        # Apply augmentations
        if self.augmentations:
            img = self.augmentations(img)
        return img, label, img_id
# Augmentáció alkalmazása a betanító adatokra
dataset = AugmentedImageDataset(images=train_image_tensors, image_ids=train_image_ids, data_array=data_array, transform=None, augmentations=augmentations)



# Modell betöltése
from model import MobileNetV2Custom
num_classes = len(np.unique(data_array[:, 1]))
model = MobileNetV2Custom(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(dev)




import torch
import numpy as np
from torch.utils.data import DataLoader, random_split

# Validációs adathalmaz kiválasztása (20%-os minta a train adatból)
validation_split_ratio = 0.2  # 20%-os validáció
train_size = int((1 - validation_split_ratio) * len(train_image_ids))
val_size = len(train_image_ids) - train_size

# Train-val felosztás
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Dataloader-ek létrehozása
train_loader = DataLoader(train_dataset, batch_size=tran_batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)






def validate_model(model, val_loader, criterion, device, label_map):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    reverse_label_map = {idx: label for label, idx in label_map.items()}  # Fordított címkék, hogy ID-ként értelmezhessük

    with torch.no_grad():
        for images, labels, ids in val_loader:
            images, labels = images.to(device), labels.to(device).long()
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            # Predikciók
            _, predicted = torch.max(outputs, 1)
            predicted_labels = [reverse_label_map[p.item()] for p in predicted]

            # Egyezések ellenőrzése az ID-kal
            correct += sum(1 for pred_id, true_id in zip(predicted_labels, ids) if pred_id == true_id)
            total += len(ids)

    val_loss /= len(val_loader)
    val_accuracy = correct / total
    return val_loss, val_accuracy, correct, total  # Hozzáadva a helyes találatok és összes mintaszám visszatérítéséhez






# Betanítás módosítása validációval
# Betanítás (validáció nélkül minden epoch után)
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for images, labels, _ in train_loader:
        images, labels = images.to(dev), labels.to(dev).long()

        optimizer.zero_grad()
        outputs = model(images)
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

    # print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}")
    scheduler.step()







# Evaluate on the validation set and count matching predicted vs. actual IDs
def evaluate_on_val_set(model, val_loader, device, label_map):
    model.eval()  # Set model to evaluation mode
    reverse_label_map = {idx: label for label, idx in label_map.items()}  # Reverse label mapping for interpretation

    correct_count = 0  # Count of correct predictions
    total_count = 0  # Total number of validation samples
    results = []  # To store test_ids and predicted labels for analysis

    with torch.no_grad():  # No need to calculate gradients during evaluation
        for images, labels, ids in val_loader:  # Loop through validation loader
            images = images.to(device)

            # Model predictions
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability

            # Interpret and format the predictions
            for i, prediction in enumerate(predicted):
                predicted_label = reverse_label_map[prediction.item()]  # Map index to actual label
                predicted_label = int(predicted_label)  # Ensure integer format

                # Append results for analysis
                results.append([ids[i], predicted_label])

                # Check if the predicted label matches the actual label (ID)
                if predicted_label == ids[i]:
                    correct_count += 1
                total_count += 1

    # Calculate accuracy
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    print(f"Validation Accuracy: {accuracy:.4f}")

    return results, accuracy  # Returns results list and validation accuracy

results, val_accuracy = evaluate_on_val_set(model, val_loader, dev, label_map)
output_file = 'log.csv'
import csv
file_exists = os.path.isfile(output_file)
# Eredmények mentése CSV fájlba
with open(output_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(['Epochs', 'Batch_size', 'Percent'])
    writer.writerow([num_epochs, tran_batch_size, val_accuracy])
print(f"Validation Accuracy: {val_accuracy:.4f}")





# --------------------------------------   KIIRATAS  -------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
from evaluate_and_export import evaluate_model
output_file = f'results/solution_epochs{num_epochs}_batch{tran_batch_size}_accuracy{val_accuracy:.2f}.csv'
output_file = 'solution.csv'
evaluate_model(model, test_image_tensors, test_image_ids, label_map, output_file, dev)

