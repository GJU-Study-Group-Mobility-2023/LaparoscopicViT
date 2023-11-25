import os

from dataset import videos_to_ImageFolder
from train import trainViT

def laparoscopicViT(mode, augmentation):
    # Specify the path to the tool annotations folder
    '/home/malbanna/Desktop/R.Janini_Work/laparoscopicViT/dataset/train'
    annotations_folder = "/home/malbanna/Desktop/R.Janini_Work/laparoscopicViT/dataset/cholec80/tool_annotations"
    # Specify the path to the videos folder
    videos_folder = "/home/malbanna/Desktop/R.Janini_Work/laparoscopicViT/dataset/cholec80/videos"
    # Specify the path to the output image folders
    images_dir = "/home/malbanna/Desktop/R.Janini_Work/laparoscopicViT/dataset/cholec80ImageFolder"

    if (os.path.isdir(images_dir)):
        print("The folder already exists")
    else:
        os.makedirs(images_dir, exist_ok=True)
        videos_to_ImageFolder(annotations_folder, videos_folder, images_dir)


    if mode == 'train':
        trainViT(model_name, images_dir, augmentation)
    else:
        return
        
    
if __name__ == "__main__":
    # mode = input("Enter mode (train/test): ")
    # augmentation = input('Use augmentaion (y/n): ')
    # model_name = input("Enter model name: ")

    model_name = 'google/vit-base-patch16-224-in21k'
    mode = 'train'
    augmentation = False
    laparoscopicViT(mode, augmentation, model_name)