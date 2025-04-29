---
layout: post
title:  "Part 1: MRI Brain Cancer Classification with fine-tuned Vision Transformer"
date:   2025-04-12 15:33:15 +0200
categories: jekyll update
---
# MRI Brain Cancer Classification with fine-tuned Vision Transformer

Medical imaging plays a pivotal role in the early detection and diagnosis of brain tumors. Among the many types of medical images, Magnetic Resonance Imaging (MRI) offers detailed structural views of the brain.   
In this blog post, I describe how to use Vision Transformers (ViTs) to classify brain tumors from MRI images. I will also compare the performance to a vanilla CNN and MobileNet V2.  
I use the the following [dataset](https://www.kaggle.com/datasets/orvile/brain-cancer-mri-dataset/data) from Kaggle. The dataset includes a total of 6056 images, uniformly resized to 512x512 pixels.

The repository can be found [here](https://github.com/alext90/brain_cancer_kaggle_vit)

### Vision Transformers

![ViT architecture](/assets/img/visionTransformer.png)

The [Vision Transformer](https://arxiv.org/pdf/2010.11929) (ViT) is a deep learning architecture introduced by Google that brings the success of transformer models in NLP into the computer vision space. Unlike Convolutional Neural Networks (CNNs), which rely on local filters to extract spatial features, ViTs process images as sequences of patches and use self-attention to model global relationships across the entire image.  

**Overview**:

1. **Image to Patches**: The input image is divided into fixed-size patches (e.g., 16×16 pixels), each of which is flattened into a vector.  
2. **Linear Embedding**: Each patch is linearly projected into an embedding space. A special class token is also prepended to the sequence.  
3. **Position Encoding**: Since transformers have no built-in notion of spatial order, positional embeddings are added to the patch embeddings.  
4. **Transformer Encoder**: The sequence is passed through a standard transformer encoder consisting of multi-head self-attention and feedforward layers.  
5. **Classification Head**: The final hidden state of the class token is fed into a classification head (e.g., a fully connected layer) to predict the class label.  

Vision Transformers have shown very good performance on image classification benchmarks, especially when fine-tuned on domain-specific tasks, where spatial context and global understanding are critical.

### Loading and Preparing the Data

Our dataset consists of MRI images labeled by brain tumor types, and we start by organizing the data into three sets: **training**, **validation**, and **test**. This allows the model to learn, tune its parameters, and finally be evaluated on unseen data.

1. **Image Transformations**:  
   We apply a series of transformations including resizing, normalization, and data augmentation (like random flips and rotations) using `torchvision.transforms`. These ensure that the model is robust to slight variations in input images.

2. **Dataset Class**:  
   We define a custom `Dataset` class that loads images and labels, applying transformations on the fly.

3. **Data Splits**:  
   The dataset is split into train, validation, and test sets using stratified sampling (to ensure balanced class distribution), and then wrapped into `DataLoader`s for efficient batching and shuffling.

```python
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class BCC_Dataloader:
    def __init__(
        self, data_dir: str, device: str, batch_size: int = 16, num_workers: int = 7
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers
        self.transform = {
            "train": transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            ),
        }
        self.train_loader, self.val_loader, self.test_loader = self.load_data()

    def load_data(self) -> tuple:
        """
        Load the dataset and split it into training, validation, and test sets.
        Returns:
            tuple: A tuple containing the training, validation, and test data loaders.
        """
        full_data = datasets.ImageFolder(
            root=self.data_dir, transform=self.transform["train"]
        )
        
        # Calculate split
        train_size = int(0.7 * len(full_data))
        val_size = int(0.2 * len(full_data))
        test_size = len(full_data) - train_size - val_size
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            full_data, [train_size, val_size, test_size]
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        
        return train_loader, val_loader, test_loader

```

### Simple CNN as Baseline

Before fine-tuning the Vision Transformer mentioned above I wanted to train a simple 2D CNN as a baseline. Nothing special here, so here just architecture and forward pass. Loss and optimizer is the same as in the ViT later:

```python
class SimpleCNN(pl.LightningModule):
    def __init__(self, num_classes, lr=1e-4):
        super(SimpleCNN, self).__init__()
        self.lr = lr

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

        ...
```


### Training and Evaluating the Vision Transformer

Once the data is prepared and loaded, the next step is to fine-tune a Vision Transformer (ViT) on our MRI dataset. I use a pretrained ViT model as base and adapt it to the classification task. Fine-tuning allows us to benefit from rich, general-purpose visual features learned on large datasets (like ImageNet), while adapting the model to the specific patterns found in brain MRI images.

#### Model Definition with PyTorch Lightning

We define a `LightningModule` that wraps a Vision Transformer backbone, adds a classification head for the 3 classes, and handles training, validation, and testing. The MobileNet V2 version looks pretty similar. Just the the classification head change is slightly different.

```python
import pytorch_lightning as pl
import torch
from torchvision import models
import torch.nn as nn


class BCC_Model(pl.LightningModule):
    def __init__(self, base_model, num_classes, lr=1e-4):
        super().__init__()

        base_model = models.vit_b_16(pretrained=True)

        # Freeze lazer
        for param in base_model.parameters():
            param.requires_grad = False

        # Replace classifier head for num_classes
        base_model.heads.head = nn.Linear(in_features=base_model.heads.head.in_features, out_features=num_classes)

        # Unfreeze classifier
        for param in base_model.heads.head.parameters():
            param.requires_grad = True

        self.model = base_model
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

```

---

#### Training and evaluating the Model

Now we can simply initialize our model and start training with the `Trainer` class. This automatically handles logging, checkpointing, and even GPU usage if available.

```python
import torch
from torchvision import models
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping


from src.data_utils import BCC_Dataloader
from src.model import BCC_Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data/"

BATCH_SIZE = 16
NUM_WORKERS = 7

NUM_CLASSES = 3
MAX_EPOCHS = 10
LEARNING_RATE = 1e-4


if __name__ == "__main__":
    BCC_dataloader = BCC_Dataloader(
        DATA_DIR, DEVICE, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )
    train_loader, val_loader, test_loader = BCC_dataloader.load_data()

    base_model = models.vgg19(pretrained=True)
    model = BCC_Model(base_model=base_model, num_classes=NUM_CLASSES, lr=LEARNING_RATE)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min",
    )

    trainer = pl.Trainer(
        accelerator="mps",
        max_epochs=MAX_EPOCHS,
        logger=CSVLogger("logs", name="BCC_Model"),
        callbacks=[early_stopping],
    )
    trainer.fit(model, train_loader, val_loader)

```

For evaluation I will use scikit-learn to calculate confusion matrix, accuracy and f1 score:

```python
def evaluate_model(model, test_loader):
    all_preds = []
    all_labels = []

    model.eval()

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            y_hat = model(x)
            preds = torch.argmax(y_hat, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    f1 = f1_score(all_labels, all_preds, average="weighted")
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    print(f"Multilabel F1 Score: {f1}")
    print(f"Accuracy: {acc}")
    print("Confusion Matrix:")
    print(cm)
```

---

#### Final Evaluation

When running the training for 10 epochs we get the following results on the test set. ViT gives us a small improvement of :  

```
CNN:
-------
acc                0.866
weighted_f1        0.864

Confusion Matrix:
[183   8   3]
[ 28 150  25]
[  8   9 192]


MobileNet V2:
-------
acc               0.857
weighted_f1       0.858

Confusion Matrix:
[179  13   2]
[ 17 158  28]
[ 15  11 183]


ViT:
------        
acc               0.905
weighted_f1       0.893

Confusion Matrix:
[181  10   2]
[ 15 163  26]
[ 13   9 185]


```

ViT outperforms both the simple CNN and the fine-tuned MobileNet V2 in this use-case. To be honest it is a very small dataset and seems to be a pretty simple task since the metrics are already pretty good without a lot of fine-tuning.