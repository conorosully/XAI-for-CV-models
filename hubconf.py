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


def get_model(model_name):

    # Check if names match
    available_models = ['car_single_room', 'pot_plant_classifier', 'pot_plant_classifier_gap']
    if model_name not in available_models:
        raise ValueError(f"Model '{model_name}' not found. Available models: {available_models}")
    
    model_path = hf_hub_download(repo_id="a-data-odyssey/XAI-for-CV-models", 
                             filename=MODEL_FILES[model_name],)

    
    config_path = os.path.join(os.path.dirname(__file__), f"models/{model_name}/config.json")

    with open(config_path, 'r') as f:
        config = json.load(f)

    num_classes = config['classes']
    input_dim = config['input_dim']

    if model_name == "pot_plant_classifier_gap":
        model = network.CNNWithGAP(num_classes=num_classes)
    else:
        model = network.CNN(num_classes=num_classes, input_dim=input_dim)

    try:
        # Try the new default (safe) way
        state_dict = torch.load(model_path, map_location='cpu')
    except Exception as e:
        print(f"Warning: Failed to load with weights_only=True ({e}). Retrying with weights_only=False...")
        # Fallback to legacy behavior if safe to do so
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)

    model.load_state_dict(state_dict)
    model.eval()

    return model
