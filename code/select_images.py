from genericpath import isdir
import torchvision
import torchvision.transforms as transforms
from instance_selection import select_instances
from torchvision.datasets import ImageFolder
import os

dataset_name = "Fer2013_merge"
train_df_path = f"../datasets/{dataset_name}"
new_path = f"../datasets/{dataset_name}_selected"

# images are expected to be in range [-1, 1]
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                     std=[0.5, 0.5, 0.5])])

# replace CIFAR10 with your own dataset 
dataset = ImageFolder(root=train_df_path, transform=transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                     transforms.ToTensor()]))
classes = dataset.classes
instance_selected_dataset = select_instances(dataset, retention_ratio=50, num_workers=0)

for c in classes:
    os.makedirs(f"{new_path}/{c}", exist_ok=True)

for i,data in enumerate(instance_selected_dataset):
    torchvision.utils.save_image(data[0],f"{new_path}/{classes[data[1]]}/{i}.png")