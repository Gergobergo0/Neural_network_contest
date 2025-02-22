
# ITT lehet változtatni, hogy milyen konfigurációkkal fusson le egymás után.
# Ez azért sexy, mert teljesen autómata, mind a log, mind a file létrehozása a result mappába.

configurations = [
    (50, 8, 1, "MobileNetV2Custom"),
    (55, 8, 1, "MobileNetV2Custom"),
    (50, 8, 0, "MobileNetV2Custom"),
]

# num_epochs = 1
# train_batch_size = 8
# fel_le_kerekit = 1  # le=1 , fel=0... lefele magasabb pontot ad
# model_neve = "MobileNetV2Custom" ---> ez logolás miatt kell



# --------------------------------------   INICIALIZÁLÁS   -------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import glob
import os
import csv

from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize
from torchvision import transforms
from model import MobileNetV2Custom
from evaluate_and_export import evaluate_model



# Seed beállítása a reprodukálhatósághoz
SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)

train_image_files = glob.glob('../Neural_network_contest/train_data/*.png')
test_image_files = glob.glob('../Neural_network_contest/test_data/*.png')

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

test_images = np.array(test_image_list)
train_images = np.array(train_image_list)


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
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Kép normalizálás: skálázás 0-1 közé ---> ezt meg kell nézni, h kell-e. Mert amúgy elég erős lehet.
# train_images = train_images / 255.0
# test_images = test_images / 255.0
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Átlag és szórás kiszámolása mindkét halmazra
train_mean = train_images.mean()
train_std = train_images.std()
test_mean = test_images.mean()
test_std = test_images.std()




# Alkalmazzuk a transzformációkat: Normalizálás mindkét adathalmazra
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[train_mean], std=[train_std]) ])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[test_mean], std=[test_std])   ])

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
file_path = 'data_labels_train.csv'
df = pd.read_csv(file_path)
selected_data = df[['filename_id', 'defocus_label']]
data_array = selected_data.to_numpy()
print(f"data_array.shape : {data_array.shape}")






# --------------------------------------   Data augmentation   ---------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------



                #   :(


# --------------------------------------   ALAP BEALLÍTÁS  -------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
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




"""
from torch.utils.data import DataLoader, random_split
validation_split_ratio = 0.2  # 10%-os validáció
train_size = int((1 - validation_split_ratio) * len(train_image_ids))
val_size = len(train_image_ids) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
val_loader = DataLoader(val_dataset, batch_size=train_batch_size, shuffle=False)
"""
# Dataset betöltés


# Betanítás validációval
for num_epochs, train_batch_size, fel_le_kerekit, model_neve in configurations:
    dataset = CustomImageDataset(images=train_image_tensors, image_ids=train_image_ids, data_array=data_array)
    train_loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)

    # Modell betöltése

    num_classes = len(np.unique(data_array[:, 1]))
    model = MobileNetV2Custom(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)


    # train_batch_size
    t_loss_min = 99
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

        """
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
        """
        if train_loss < t_loss_min : t_loss_min = train_loss
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}")
        scheduler.step()









    # --------------------------------------   LOGOLÁS  --------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
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

    # results, val_accuracy = evaluate_on_val_set(model, val_loader, dev, label_map)
    val_accuracy = 0.0
    output_file = 'log.csv'

    file_exists = os.path.isfile(output_file)
    # Eredmények mentése CSV fájlba
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Epochs', 'Batch_size', 'Percent'])
        writer.writerow([num_epochs, train_batch_size, val_accuracy])
    print(f"Validation Accuracy: {val_accuracy:.4f}")





    # --------------------------------------   KIIRATAS  -------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    evaluate_model(model, test_image_tensors, test_image_ids, label_map, dev,num_epochs,train_batch_size,val_accuracy,fel_le_kerekit,model_neve,t_loss_min)
