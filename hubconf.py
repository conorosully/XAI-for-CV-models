dependencies = ['torch','huggingface_hub']
import network
import torch
from huggingface_hub import hf_hub_download
import json
import os  # adjust to your model class



MODEL_FILES = {
    "car_single_room": "models/car_single_room/model.pth",
    "pot_plant_classifier": "models/pot_plant_classifier/model.pth",
    "pot_plant_classifier_gap": "models/pot_plant_classifier_gap/model.pth",
}

def car_single_room():
    return get_model("car_single_room", num_classes=1, input_dim=224)

def pot_plant_classifier():
    return get_model("pot_plant_classifier")

def pot_plant_classifier_gap():
    return get_model("pot_plant_classifier_gap")

def get_model(model_name,num_classes=4,input_dim=256):

    # Check if names match
    available_models = ['car_single_room', 'pot_plant_classifier', 'pot_plant_classifier_gap']
    if model_name not in available_models:
        raise ValueError(f"Model '{model_name}' not found. Available models: {available_models}")
    
    model_path = hf_hub_download(repo_id="a-data-odyssey/XAI-for-CV-models", 
                             filename=MODEL_FILES[model_name],)


    if model_name == "pot_plant_classifier_gap":
        model = network.CNNWithGAP(num_classes=num_classes)
    else:
        model = network.CNN(num_classes=num_classes, input_dim=input_dim)

    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    return model
