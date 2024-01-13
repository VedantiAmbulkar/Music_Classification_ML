
# Packages Used
import math
import numpy as np
from sklearn.metrics import f1_score
from glob import glob
import librosa.display
import pandas as pd
import torch
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import Subset
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as trans
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split,SubsetRandomSampler
from torchvision.transforms import ToTensor
import librosa
import time
import os
import matplotlib.pyplot as plt


# The below function is used to get the audio files and convert them to Mel Spectrograms
import matplotlib.image
dataset_path = "/kaggle/input/music-genre-data/Data/genres_original"
def convert_to_image():
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        if dirpath is not dataset_path:

            # save genre label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("/")[-1]

            if not os.path.exists('/kaggle/working/'+semantic_label):
                os.mkdir('/kaggle/working/'+semantic_label)

            # process all audio files in genre sub-dir

            for i,f in enumerate(filenames):

                file_path = os.path.join(dirpath, f)
                #print(file_path)
                signal, _ = librosa.load(path=file_path)
                S = librosa.feature.melspectrogram(y=signal, sr=22050)
                S_DB = librosa.power_to_db(S, ref=np.max)
                matplotlib.image.imsave('/kaggle/working/'+semantic_label+"1"+'/'+str(i)+'image.png', S_DB)
            print(semantic_label+"completed")

convert_to_image()

#The below code gets the images from the directory and converts them into a 224 by 224 normalized tensors.
dataset = ImageFolder('/kaggle/input/music-genre-data/Data/images_original', trans.Compose([
        trans.Resize((224, 224)),
        trans.ToTensor(),
        trans.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
]))

val_pct = 0.01
val_size = int(val_pct * len(dataset))
train_ds, valid_ds = random_split(dataset, [len(dataset) - val_size, val_size])

# define the validation and training batch sizes and load them into dataloaders
train_batch_size = 250
valid_batch_size = 32
trainloader = DataLoader(train_ds, batch_size=train_batch_size)
validloader = DataLoader(valid_ds, batch_size=valid_batch_size)

# The below code gets the pretrained weights and loads them into an object
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ft = models.vgg16(pretrained=True)

total_trainable_params = sum(
    p.numel() for p in model_ft.parameters() if p.requires_grad)

total_trainable_params

# the below code feezes the parameters of the model since we do not need all parameters for training purpose
for param in model_ft.parameters():
    param.requires_grad = False

# This defines the architectural changes made to our VGG 16 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ft = models.vgg16(pretrained=True)

num_features = model_ft.classifier[6].in_features
features = list(model_ft.classifier.children())[:-1]
features.extend([nn.Linear(num_features,2048),
                 nn.ReLU(inplace=True),
                 nn.Linear(2048,512),
                 nn.ReLU(inplace=True),
                 nn.Linear(512,256),
                 nn.ReLU(inplace=True),
                 nn.Linear(256,128),
                 nn.ReLU(inplace=True),
                 nn.Linear(128,64),
                 nn.ReLU(inplace=True),
                 nn.Linear(64,10),
                      nn.Softmax()])
model_ft.classifier = torch.nn.Sequential(*features)
print(model_ft)

# Below lines define the loss function and the optimizer to be used during training.
criterion = torch.nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)

# below function counts the total parameters we are going to train our model on.
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(count_parameters(model_ft))

# Below function defines the main training loop
# It takes as parameters the model , loss, optimizer and number of epochs.
def train_model(model, criterion, optimizer, num_epochs):
    since = time.time()
    model.train()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        running_loss = 0
        running_corrects = 0

        for i,data in enumerate(trainloader):

            optimizer.zero_grad()
            inputs,labels=data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(trainloader.dataset)
        print(f"Train Loss: {epoch_loss:.4f}")

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

    return model

vgg_based = train_model(model_ft, criterion, optimizer_ft, num_epochs=3)

# below function calculates the accuracy on the test dataset.
def test_accuracy(model,dataloader):
    correct = 0
    total = 0
    model.eval()
    y_preds=[]
    y_true=[]
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, dim=1)
            y_preds.extend(predicted)
            y_true.extend(labels)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the {total} test Music Samples: {accuracy:.2f} %')
    return y_preds,y_true

y_preds,y_true=test_accuracy(vgg_based,validloader)

classes=['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop','reggae', 'rock']
def confusionmatrix(y_true,y_pred):
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                         columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sns.heatmap(df_cm, annot=True)
    # plt.savefig('output.png')

torch.save(vgg_based.state_dict(), '/kaggle/working/model.pth')


confusionmatrix(y_true,y_preds)

