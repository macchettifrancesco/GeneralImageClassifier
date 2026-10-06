import os
import copy
import json
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

from sklearn.model_selection import StratifiedKFold


# Configurazione
DATASET_DIR = "dataset"
MODEL_OUTPUT_PATH = "model.pth"
CLASS_NAMES_PATH = "class_names.json"

BATCH_SIZE = 16
NUM_EPOCHS = 40
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224
EARLY_STOPPING_PATIENCE = 7
K_FOLDS = 5
RANDOM_STATE = 42

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# Data transform
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Caricamento dati
def load_datasets(dataset_dir: str) -> Tuple[
    Dict[str, datasets.ImageFolder],
    list
]:
    image_datasets = {
        "train": datasets.ImageFolder(dataset_dir, transform=train_transforms),
        "val": datasets.ImageFolder(dataset_dir, transform=val_transforms),
    }

    class_names = image_datasets["train"].classes

    return image_datasets, class_names


def build_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)

    # Congela tutti i parametri inizialmente
    for param in model.parameters():
        param.requires_grad = False

    # Scongela l'ultimo blocco convoluzionale
    for param in model.features[-1].parameters():
        param.requires_grad = True

    # Sostituisce la testa con un classificatore più robusto
    in_features = model.classifier[-1].in_features

    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
    )

    return model.to(DEVICE)


# Ottimizzatore con LR differenziato
def build_optimizer(model: nn.Module) -> optim.Optimizer:
    return optim.Adam(
        [
            {"params": model.features[-1].parameters(), "lr": LEARNING_RATE * 0.1},
            {"params": model.classifier.parameters(), "lr": LEARNING_RATE},
        ],
        weight_decay=1e-4,
    )


# Addestramento
def train_model(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    dataset_sizes: Dict[str, int],
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
) -> Tuple[nn.Module, float]:

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f"\nEpoca {epoch + 1}/{num_epochs}")

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print(f"  {phase.upper():5s} — Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            # Early stopping
            if phase == "val":
                scheduler.step()

                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    print(
                        f"  [Early stopping] nessun miglioramento da "
                        f"{epochs_no_improve}/{EARLY_STOPPING_PATIENCE} epoche"
                    )

                    if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                        print(f"\nEarly stopping attivato all'epoca {epoch + 1}.")
                        model.load_state_dict(best_model_wts)
                        return model, best_val_acc

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    model.load_state_dict(best_model_wts)

    return model, best_val_acc


# K-Fold Cross Validation
def train_k_fold(
    image_datasets: Dict[str, datasets.ImageFolder],
    class_names: list,
) -> nn.Module:

    num_classes = len(class_names)

    labels = image_datasets["train"].targets

    skf = StratifiedKFold(
        n_splits=K_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    best_model_wts = None
    best_fold_acc = 0.0
    fold_accuracies = []

    for fold, (train_indices, val_indices) in enumerate(
        skf.split(range(len(labels)), labels),
        start=1
    ):
        print("\n" + "=" * 50)
        print(f"Fold {fold}/{K_FOLDS}")
        print("=" * 50)

        train_subset = Subset(image_datasets["train"], train_indices)
        val_subset = Subset(image_datasets["val"], val_indices)

        dataloaders = {
            "train": DataLoader(
                train_subset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=2
            ),
            "val": DataLoader(
                val_subset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=2
            ),
        }

        dataset_sizes = {
            "train": len(train_subset),
            "val": len(val_subset),
        }

        model = build_model(num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = build_optimizer(model)

        model, val_acc = train_model(
            model=model,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=NUM_EPOCHS,
        )

        fold_accuracies.append(val_acc)

        print(f"\nFold {fold} completata — Best Val Acc: {val_acc:.4f}")

        if val_acc > best_fold_acc:
            best_fold_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print("\n" + "=" * 50)
    print("Risultati K-Fold Cross Validation")
    print("=" * 50)

    for i, acc in enumerate(fold_accuracies, start=1):
        print(f"Fold {i}: {acc:.4f}")

    mean_acc = sum(fold_accuracies) / len(fold_accuracies)

    print(f"\nAccuracy media: {mean_acc:.4f}")
    print(f"Migliore accuracy: {best_fold_acc:.4f}")

    final_model = build_model(num_classes)
    final_model.load_state_dict(best_model_wts)

    return final_model


# Salvataggio
def save_model(
    model: nn.Module,
    class_names: list,
    model_path: str,
    class_names_path: str,
) -> None:
    torch.save(model.state_dict(), model_path)

    with open(class_names_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    print(f"\nModello salvato in: {model_path}")
    print(f"Classi salvate in:  {class_names_path}")


# Main
def main() -> None:
    start_time = time.time()
    print(f"Device in uso: {DEVICE}\n")

    image_datasets, class_names = load_datasets(DATASET_DIR)

    print("Classi trovate:", class_names)

    model = train_k_fold(
        image_datasets=image_datasets,
        class_names=class_names,
    )

    save_model(
        model=model,
        class_names=class_names,
        model_path=MODEL_OUTPUT_PATH,
        class_names_path=CLASS_NAMES_PATH,
    )

    elapsed = time.time() - start_time
    print(f"\nTraining completato in {elapsed:.2f} secondi")


if __name__ == "__main__":
    main()