# Augmentáció

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

# Data augmentation transzformációk
augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),  # A ToTensor átalakítás az augmentáció után történik
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

        # Transzformációk alkalmazása
        if self.transform:
            img = self.transform(img)

        # Augmentáció alkalmazása (ID változatlan marad)
        if self.augmentations:
            augmented_img = self.augmentations(img)
            return augmented_img, label, img_id  # Augmentált adat ID-val
        else:
            return img, label, img_id

# Augmentáció alkalmazása a betanító adatokra
dataset = AugmentedImageDataset(images=train_image_tensors, image_ids=train_image_ids, data_array=data_array, transform=None, augmentations=augmentations)

# Adatok felosztása tréning és validációs készletre
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)





# Validáció
# Validációs ciklus az epoch során
def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images, labels = images.to(device), labels.to(device).long()
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(val_loader)
    val_accuracy = correct / total
    return val_loss, val_accuracy

# Betanítás módosítása validációval
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

    # Validáció futtatása
    val_loss, val_accuracy = validate_model(model, val_loader, criterion, dev)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    scheduler.step()
