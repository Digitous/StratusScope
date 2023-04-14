import os
import json
import torch
from transformers import AutoModel, AutoConfig, logging
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

logging.set_verbosity_warning()
logging.set_verbosity_error()

def select_folder():
    Tk().withdraw()
    folder = filedialog.askdirectory()
    return folder

def load_sharded_model(folder):
    files = os.listdir(folder)
    model_files = sorted([f for f in files if f.startswith('pytorch_model-') and f.endswith('.bin')])

    num_shards = len(model_files)

    model_state_dict = {}
    for model_file in model_files:
        shard = torch.load(os.path.join(folder, model_file), map_location=torch.device('cpu'))
        model_state_dict.update(shard)

    config = AutoConfig.from_pretrained(folder)
    model = AutoModel.from_pretrained(folder, config=config, state_dict=model_state_dict)
    return model

def get_layer_number(name):
    parts = name.split('.')
    for part in parts:
        if part.isdigit():
            return int(part)
    return None

def compare_layers(model1, model2):
    layer_diffs = []
    layer_diff = 0
    current_layer = None

    model1_parameters = list(model1.named_parameters())
    num_parameters = len(model1_parameters)

    for i, (n1, p1) in enumerate(model1_parameters):
        layer_number = get_layer_number(n1)

        # If this is a new layer or the last parameter, store the previous layer's difference and reset
        if current_layer is None:
            current_layer = layer_number
        elif current_layer != layer_number or i == num_parameters - 1:
            layer_diffs.append(layer_diff)
            layer_diff = 0
            current_layer = layer_number

        p2 = model2.state_dict()[n1]
        diff = torch.abs(p1 - p2).sum().item()
        layer_diff += diff

    return layer_diffs

def plot_layer_diff(layer_diffs, model1_name, model2_name):
    plt.figure(figsize=(20, 6))
    num_layers = len(layer_diffs)
    layer_indices = range(num_layers)
    plt.bar(layer_indices, layer_diffs)
    plt.xticks(layer_indices)
    plt.xlabel('Layer')
    plt.ylabel('Difference')
    plt.title(f"{model1_name} vs {model2_name} Layer Difference")
    plt.ylim(bottom=0)
    print("Script completed, close graph to unload models and return to commandline.")    
    plt.show()

def main():
    print("Select model1 folder:")
    model1_folder = select_folder()
    model1_name = os.path.basename(model1_folder)
    print("Select model2 folder:")
    model2_folder = select_folder()
    model2_name = os.path.basename(model2_folder)

    print("Loading Models...")
    model1 = load_sharded_model(model1_folder)
    model2 = load_sharded_model(model2_folder)

    print("Examining Models...")
    layer_diffs = compare_layers(model1, model2)

    plot_layer_diff(layer_diffs, model1_name, model2_name)

    del model1
    del model2
    torch.cuda.empty_cache()
    import gc
    gc.collect()

if __name__ == "__main__":
    main()