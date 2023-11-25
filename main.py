import os

from dataset import videos_to_ImageFolder

# Specify the path to the tool annotations folder
annotations_folder = "C:\\Users\\moham\\Desktop\\GJU\\dataset\\cholec80\\tool_annotations"
# Specify the path to the videos folder
videos_folder = "C:\\Users\\moham\\Desktop\\GJU\\dataset\\cholec80\\videos"
# Specify the path to the output image folders
output_folder = "C:\\Users\\moham\\Desktop\\GJU\\dataset\\cholec80ImageFolder"

if (os.path.isdir(output_folder)):
    print("The folder already exists")
else:
    os.makedirs(output_folder, exist_ok=True)
    videos_to_ImageFolder(annotations_folder, videos_folder, output_folder)