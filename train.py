import os
import json
import torch
import evaluate
import requests
import numpy as np
import pandas as pd
import pyarrow as pa
from sklearn.metrics import precision_score

from PIL import Image
from io import BytesIO
from typing import Tuple
from torchvision.transforms import Compose, ColorJitter, ToTensor, Normalize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
from datasets import load_dataset, load_from_disk, Dataset, Features, Array3D
from transformers import AutoProcessor, ViTFeatureExtractor, ViTForImageClassification, Trainer, TrainingArguments, default_data_collator



def split_dataset(
    dataset: Dataset,
    val_size: float=0.2,
    test_size: float=0.1
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Returns a tuple with three random train, validation and test subsets by splitting the passed dataset.
    Size of the validation and test sets defined as a fraction of 1 with the `val_size` and `test_size` arguments.
    """

    print("Splitting dataset into train, validation and test sets...")

    # Split dataset into train and (val + test) sets
    split_size = round(val_size + test_size, 3)
    dataset = dataset.train_test_split(shuffle=True, test_size=split_size)

    # Split (val + test) into val and test sets
    split_ratio = round(test_size / (test_size + val_size), 3)
    val_test_sets = dataset['test'].train_test_split(shuffle=True, test_size=split_ratio)

    train_dataset = dataset["train"]
    val_dataset = val_test_sets["train"]
    test_dataset = val_test_sets["test"]
    return train_dataset, val_dataset, test_dataset


def process_examples_augmentation(examples, image_processor):   
    # Get batch of images
    images = examples['image']
    augment  = Compose([
        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation(degrees=15),
        ToTensor()
    ])
    
    # Preprocess
    inputs = image_processor(images=images)
    # Add pixel_values
    examples['pixel_values'] = inputs['pixel_values']
    examples["pixel_values"] = [augment(image.convert("RGB")) for image in examples["image"]]
    examples["pixel_values"] = torch.stack(examples["pixel_values"])

    return examples

def process_examples(examples, image_processor):   
    # Get batch of images
    images = examples['image']
    # Preprocess
    inputs = image_processor(images=images)
    # Add pixel_values
    examples['pixel_values'] = inputs['pixel_values']

    return examples

def apply_processing(
    model_name: str,
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    augmentation: bool = False
) -> Tuple[Dataset, Dataset, Dataset]:

    # Extend the features 
    features = Features({
        **train_dataset.features,
        'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
    })

    # Instantiate image_processor
    image_processor = AutoProcessor.from_pretrained(model_name)
    
    # Preprocess images
    if (not augmentation):
        train_dataset = train_dataset.map(process_examples, batched=True, features=features, fn_kwargs={"image_processor": image_processor})
        val_dataset = val_dataset.map(process_examples, batched=True, features=features, fn_kwargs={"image_processor": image_processor})
        test_dataset = test_dataset.map(process_examples, batched=True, features=features, fn_kwargs={"image_processor": image_processor})
    else:
        train_dataset = train_dataset.map(process_examples_augmentation, batched=True, features=features, fn_kwargs={"image_processor": image_processor})
        val_dataset = val_dataset.map(process_examples_augmentation, batched=True, features=features, fn_kwargs={"image_processor": image_processor})
        test_dataset = test_dataset.map(process_examples_augmentation, batched=True, features=features, fn_kwargs={"image_processor": image_processor})

    # Set to torch format for training
    train_dataset.set_format('torch', columns=['pixel_values', 'label'])
    val_dataset.set_format('torch', columns=['pixel_values', 'label'])
    test_dataset.set_format('torch', columns=['pixel_values', 'label'])
    
    # Remove unused column
    train_dataset = train_dataset.remove_columns("image")
    val_dataset = val_dataset.remove_columns("image")
    test_dataset = test_dataset.remove_columns("image")
    
    return train_dataset, val_dataset, test_dataset



# Split dataset into train and test sets
def trainViT(model_name, images_dir, augmentation):
    train_save_path = '/home/malbanna/Desktop/R.Janini_Work/laparoscopicViT/dataset/train'
    val_save_path = '/home/malbanna/Desktop/R.Janini_Work/laparoscopicViT/dataset/val'
    test_save_path = '/home/malbanna/Desktop/R.Janini_Work/laparoscopicViT/dataset/test'

    model_dir = "/home/malbanna/Desktop/R.Janini_Work/laparoscopicViT/model"
    output_data_dir = "/home/malbanna/Desktop/R.Janini_Work/laparoscopicViT/outputs"

    val_size = 0.2
    test_size = 0.1
    k_for_top_acc = 3

    # Total number of training epochs to perform
    num_train_epochs = 15
    # The batch size per GPU/TPU core/CPU for training
    per_device_train_batch_size = 32
    # The batch size per GPU/TPU core/CPU for evaluation
    per_device_eval_batch_size = 64
    # The initial learning rate for AdamW optimizer
    learning_rate = 2e-5
    # Number of steps used for a linear warmup from 0 to learning_rate
    warmup_steps = 500
    # The weight decay to apply to all layers except all bias and LayerNorm weights in AdamW optimizer
    weight_decay = 0.01

    main_metric_for_evaluation = "mean_precision"


    dataset = dataset.load_dataset(images_dir, val_size, test_size)
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, val_size, test_size)

    print(f"Train images: {train_dataset.num_rows}")
    print(f"Val images: {val_dataset.num_rows}")
    print(f"Test images: {test_dataset.num_rows}")

    # Apply AutoProcessor
    train_dataset, val_dataset, test_dataset = apply_processing(model_name, train_dataset, val_dataset, test_dataset, augmentation)

    # Save train, validation and test preprocessed datasets
    train_dataset.save_to_disk(train_save_path, num_shards=1)
    val_dataset.save_to_disk(val_save_path, num_shards=1)
    test_dataset.save_to_disk(test_save_path, num_shards=1)

    train_dataset = load_from_disk(train_save_path)
    val_dataset = load_from_disk(val_save_path)
    
    num_classes = train_dataset.features["label"].num_classes
    print(f"Number of classes: {num_classes}")
    labels = train_dataset.features["label"]
    print(f"Labels: {labels}")

    # Download model from model hub
    model = ViTForImageClassification.from_pretrained(model_name, num_labels=num_classes)

    # Download feature extractor from hub
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

    # Compute metrics function for binary classification
    acc_metric = evaluate.load("accuracy", module_type="metric")

    def compute_metrics(eval_pred):
        predicted_probs, labels = eval_pred
        # Accuracy
        predicted_labels = np.argmax(predicted_probs, axis=1)
        acc = acc_metric.compute(predictions=predicted_labels, references=labels)
        # Top-K Accuracy
        top_k_indexes = [np.argpartition(row, -k_for_top_acc)[-k_for_top_acc:] for row in predicted_probs]
        top_k_classes = [top_k_indexes[i][np.argsort(row[top_k_indexes[i]])] for i, row in enumerate(predicted_probs)]
        top_k_classes = np.flip(np.array(top_k_classes), 1)
        acc_k = {
            f"accuracy_k" : sum([label in predictions for predictions, label in zip(top_k_classes, labels)]) / len(labels)
        }
        
        # Precision for each class
        predicted_classes = np.argmax(predicted_probs, axis=1)
        precision_per_class = precision_score(labels, predicted_classes, average=None)
        
        # Mean Precision
        mean_precision = np.mean(precision_per_class)
        
        # Merge metrics
        acc.update(acc_k)
        acc.update({"precision_per_class": precision_per_class.tolist()})
        acc.update({"mean_precision": mean_precision})
        return acc


    # Change labels
    id2label = {key:train_dataset.features["label"].names[index] for index,key in enumerate(model.config.id2label.keys())}
    label2id = {train_dataset.features["label"].names[index]:value for index,value in enumerate(model.config.label2id.values())}
    model.config.id2label = id2label
    model.config.label2id = label2id

    # Define training args
    training_args = TrainingArguments(
        output_dir = model_dir,
        num_train_epochs = num_train_epochs,
        per_device_train_batch_size = per_device_train_batch_size,
        per_device_eval_batch_size = per_device_eval_batch_size,
        warmup_steps = warmup_steps,
        weight_decay = weight_decay,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        logging_strategy = "epoch",
        logging_dir = f"{output_data_dir}/logs",
        learning_rate = float(learning_rate),
        load_best_model_at_end = True,
        metric_for_best_model = main_metric_for_evaluation,
    )

    # Create Trainer instance
    trainer = Trainer(
        model = model,
        args = training_args,
        compute_metrics = compute_metrics,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        data_collator = default_data_collator,
        tokenizer = feature_extractor
    )

    trainer.train()
    trainer.save_model(model_dir)   
    results = trainer.evaluate(eval_dataset=val_dataset)
    print(results)
