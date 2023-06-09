import torch
import sys
import safetensors.torch


def load_model(path):
    if path.endswith(".safetensors"):
        model_file = safetensors.torch.load_file(path, device="cpu")
    else:
        model_file = torch.load(path, map_location="cpu")

    state_dict = model_file["state_dict"] if "state_dict" in model_file else model_file
    return state_dict

def check_tensors(model_path: str):
    wrong_index = []
    if model_path == "":
        print("Please provide a model path")
        return wrong_index

    checkpoint = load_model(model_path)

    if "cond_stage_model.transformer.text_model.embeddings.position_ids" in checkpoint:
        check_tensor = checkpoint["cond_stage_model.transformer.text_model.embeddings.position_ids"]
    elif "cond_stage_model.transformer.embeddings.position_ids" in checkpoint:
        check_tensor = checkpoint["cond_stage_model.transformer.embeddings.position_ids"]
    else:
        print("Invalid checkpoint file or checkpoint in SDv2 format version")
        return wrong_index

    for i in range(torch.numel(check_tensor)):
        tensor_value = check_tensor.data[0, i]
        value_error = tensor_value-i
        if abs(value_error)>0.0001:
            wrong_index.append(i)
            print(f"Wrong index: {i}, Value: {tensor_value:.5f}, Deviation: {value_error:.5f}")
    return wrong_index

# Usage
model_path = sys.argv[1]  # get the model path from command line argument
wrong_indexes = check_tensors(model_path)

