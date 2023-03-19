import gradio as gr
import os
import time

demo = gr.Blocks(title="Re-basin Merger")
with demo:
    error_box = gr.Textbox(label="Error", visible=False)

    with gr.Row():
        modelA_box = gr.Textbox(label="Model A path")
        modelB_box = gr.Textbox(label="Model B path")
        
    output_box = gr.Textbox(label="Filename for saving merged model", value="merged_output")

    with gr.Row():
        iterations_box = gr.Number(label="Iterations", value=100)
        alpha_box = gr.Slider(value=0.5, minimum=0, maximum=1, step=0.1, label="Alpha")

    with gr.Row():
        usefp16_box = gr.Checkbox(label="Use fp16", value=True)
        cuda_box = gr.Checkbox(label="Cuda", value=False)

    with gr.Row():
        install_btn = gr.Button("install re-basin requirements")
        run_btn = gr.Button("Run re-basin")

    output = gr.Textbox(label="Output", visible=False)

    with gr.Column(visible=False) as output_col:
        diagnosis_box = gr.Textbox(label="Diagnosis")
        patient_summary_box = gr.Textbox(label="Patient Summary")

    def run(modelA, modelB, output, iterations, alpha, usefp16, device ):
        if len(modelA) == 0:
            return {error_box: gr.update(value="Enter path to model A", visible=True)}
        if len(modelB) == 0:
            return {error_box: gr.update(value="Enter path to model B", visible=True)}

        device_type = "cuda" if device else "cpu"
        usefp16_type = " --usefp16 " if usefp16 else ""

        iterations = int(iterations)
        if os.name == 'posix':
            cmd = "python " + os.path.dirname(__file__) + "/SD_rebasin_merge.py --model_a " + modelA + " --model_b " + modelB + " --output " + output +  str(usefp16_type) + " --alpha " + str(alpha) + " --iterations " + str(iterations) + " --device " + device_type
        if os.name == 'nt': 
            cmd = "python " + os.path.dirname(__file__) + "\\SD_rebasin_merge.py --model_a " + modelA + " --model_b " + modelB + " --output " + output +  str(usefp16_type) + " --alpha " + str(alpha) + " --iterations " + str(iterations) + " --device " + device_type

        return {
            
            os.system(cmd)

            

            #output_col: gr.update(visible=True),
            #diagnosis_box: "covid" if "Cough" in symptoms else "flu",
            #patient_summary_box: f"{name}, {age} y/o"
        }
    def install():
        install_cmd = "pip install -r" + os.path.dirname(__file__) + "\\requirements.txt"
        install_cmd2 = "pip install -r" + os.path.dirname(__file__) + "\\requirements2.txt -i https://download.pytorch.org/whl/cu113 "

        return {
            os.system(install_cmd),
            os.system(install_cmd2),
            "done"
        }

    run_btn.click(run,[modelA_box, modelB_box, output_box, iterations_box, alpha_box, usefp16_box, cuda_box])
    install_btn.click(install)


if __name__ == "__main__":
    demo.launch()

