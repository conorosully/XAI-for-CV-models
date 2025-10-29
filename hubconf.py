dependencies = ['torch', 'torchvision']
import network
import torch
import json
import os

def get_model(model_name):

    # Check if names match
    available_models = ['car_single_room', 'pot_plant_classifier', 'pot_plant_classifier_gap']
    if model_name not in available_models:
        raise ValueError(f"Model '{model_name}' not found. Available models: {available_models}")
    
    model_path = "models/{}/model.pth".format(model_name)
    config_path = "models/{}/config.json".format(model_name)

    config = json.load(open(os.path.join(os.path.dirname(__file__), config_path)))
    num_classes = config['classes']
    input_dim = config['input_dim']

    if model_name == "pot_plant_classifier_gap":
        model = network.CNNWithGAP(num_classes=num_classes)
    else:
        model = network.CNN(num_classes=num_classes, input_dim=input_dim)

    state_dict_loaded = torch.load(os.path.join(os.path.dirname(__file__), model_path), map_location='cpu')
    model.load_state_dict(state_dict_loaded)
    model.eval()



    return model