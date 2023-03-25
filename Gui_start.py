import gradio as gr
import os
import time
import json
from gradio.themes.base import Base



# Define the filename to save the input values
SAVE_FILE = "re-basin_config.json"

# Define a function to save the input values to a file
def save_input_values(inputs):
    with open(SAVE_FILE, "w") as f:
        json.dump(inputs, f)

# Define a function to load the input values from a file
def load_input_values():
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE, "r") as f:
            return json.load(f)
    else:
        return {}

# Load saved input values (if any)
saved_inputs = load_input_values()

demo = gr.Blocks(title="Re-basin Merger", theme='gradio/monochrome', width="100%")
with demo:
    error_box = gr.Textbox(label="Error", visible=False)

    with gr.Row():
        modelA_box = gr.Textbox(label="Model A path", value=saved_inputs.get("modelA_box", ""))
        modelB_box = gr.Textbox(label="Model B path", value=saved_inputs.get("modelB_box", ""))
        
    output_box = gr.Textbox(label="Filename for saving merged model or pruning", value=saved_inputs.get("output_box", "merged_model"))

    with gr.Row():
        iterations_box = gr.Number(label="Iterations", value=saved_inputs.get("iterations_box", 100))
        alpha_box = gr.Slider(value=saved_inputs.get("alpha_box", 0.5), minimum=0, maximum=1, step=0.001, label="Alpha")

    with gr.Row():
        usefp16_box = gr.Checkbox(label="Use fp16", value=saved_inputs.get("usefp16_box", True))
        cuda_box = gr.Checkbox(label="GPU", value=saved_inputs.get("cuda_box", False))
        fast_box = gr.Checkbox(label="Fast", value=saved_inputs.get("fast_box", True))

    with gr.Row():
        merge_layers_radio = gr.Radio(["All", "Convolutional layers", "Fully connected layers"], label="Layers to be merged", value=saved_inputs.get("merge_layers_radio", "All"), interactive=True)

    with gr.Row():
        run_btn = gr.Button("Run re-basin")
        prune_btn = gr.Button("Prune model")

    output = gr.Textbox(label="Output", visible=False)

    with gr.Column(visible=False) as output_col:
        diagnosis_box = gr.Textbox(label="Diagnosis")
        patient_summary_box = gr.Textbox(label="Patient Summary")

    def run_rebasin(modelA, modelB, output, iterations, alpha, usefp16, device, merge_layers, fast):
        inputs = {
            "modelA_box": modelA,
            "modelB_box": modelB,
            "output_box": output,
            "iterations_box": iterations,
            "alpha_box": alpha,
            "usefp16_box": usefp16,
            "cuda_box": device,
            "merge_layers_radio": merge_layers,
            "fast_box": fast
        }
        save_input_values(inputs)
        
        if len(modelA) == 0:
            return {error_box: gr.update(value="Enter path to model A", visible=True)}
        if len(modelB) == 0:
            return {error_box: gr.update(value="Enter path to model B", visible=True)}

        device_type = "cuda" if device else "cpu"
        usefp16_type = " --usefp16 " if usefp16 else ""
        go_fast = " --fast " if fast else ""
        
        if merge_layers == "All":
            layers = "all"
        elif merge_layers == "Convolutional layers":
            layers = "convolutional"
        elif merge_layers == "Fully connected layers":
            layers = "fully_connected"
        

        iterations = int(iterations)
        if os.name == 'posix':
            rebasin_cmd = "python " + os.path.dirname(__file__) + "/SD_rebasin_merge.py --model_a \"" + modelA + "\" --model_b \"" + modelB + "\"  --layers " + layers + " --output " + output +  str(usefp16_type) + " " +  str(go_fast) + " --alpha " + str(alpha) + " --iterations " + str(iterations) + " --device " + device_type
        if os.name == 'nt': 
            rebasin_cmd = "python " + os.path.dirname(__file__) + "\\SD_rebasin_merge.py --model_a \"" + modelA + "\" --model_b \"" + modelB + "\"  --layers " + layers + " --output " + output +  str(usefp16_type) + " " +  str(go_fast) + " --alpha " + str(alpha) + " --iterations " + str(iterations) + " --device " + device_type

        return {
            os.system(rebasin_cmd)
        }

    def run_prune(output, usefp16):
        usefp16_type = " --usefp16 " if usefp16 else ""
        prune_cmd = "python " + os.path.dirname(__file__) + "/prune.py " + str(usefp16_type) + output + ".safetensors " + output + "_pruned.ckpt" 

        return {
            os.system(prune_cmd)
        }

    run_btn.click(run_rebasin,[modelA_box, modelB_box, output_box, iterations_box, alpha_box, usefp16_box, cuda_box, merge_layers_radio, fast_box])
    prune_btn.click(run_prune,[output_box, usefp16_box])

if __name__ == "__main__":
    demo.launch()

