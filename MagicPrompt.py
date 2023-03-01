#!/usr/bin/env python
# coding: utf-8

import os
import html
import torch

import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM
from omegaconf import OmegaConf

conf = OmegaConf.load('models.yaml')
base_dir = os.path.dirname(os.path.realpath(__file__))
models_dir = os.path.join(base_dir, conf.models_dir)


class Model:
    name = None
    model = None
    tokenizer = None


models = {}
current = Model()


def model_selection_changed(model_name):
    if model_name == "None":
        current.tokenizer = None
        current.model = None
        current.name = None
    else:
        current.tokenizer = models[model_name].tokenizer
        current.model = models[model_name].model
        current.name = models[model_name].name


def loading_models():
    for item in conf.models:
        _model = Model()
        _model.name = item["Name"]
        _model.model = item["Model"]
        _model.tokenizer = item["Tokenizer"]
        models[_model.name] = _model

    model_selection_changed(list(models.keys())[0])


def get_model_path(name):
    dirname = os.path.join(models_dir, name)
    if not os.path.isdir(dirname):
        return name

    return dirname


def check_model():
    model_name = current.name
    path = get_model_path(model_name)
    model_chk = os.path.exists(models_dir+model_name+'/pytorch_model.bin')
    if model_chk is False:
        print('model not found, start cloning from Hugging Face')
        dir_chk = os.path.exists(models_dir)
        if dir_chk is False:
            os.makedirs(models_dir)

        if model_name != 'None':
            path = os.path.join(models_dir, model_name)
            dir_chk = os.path.exists(path)
            if dir_chk is False:
                os.makedirs(path)

            try:
                tokenizer_dl = GPT2Tokenizer.from_pretrained(current.tokenizer)
                model_dl = GPT2LMHeadModel.from_pretrained(current.model)
                tokenizer_dl.save_pretrained(path)
                model_dl.save_pretrained(path)
                print('model cloned from Hugging Face')
            except Exception as e:
                print(
                    f"Exception encountered while attempting to install tokenizer: {e}")
                return gr.update(), f"Error: {e}"


def generate(model, prompt, temperature, top_k, min_length, max_length, repetition_penalty, num_return_sequences):
    try:
        if current.name != model:
            current.tokenizer = None
            current.model = None
            current.name = None
        if model != 'None':
            check_model()
            path = get_model_path(model)
            current.tokenizer = AutoTokenizer.from_pretrained(
                models[model].tokenizer)
            current.model = AutoModelForCausalLM.from_pretrained(
                models[model].model)
            current.name = current.name

        assert current.model, 'No model available'
        assert current.tokenizer, 'No tokenizer available'
    except Exception as e:
        print(f"Exception: {e}")

    try:
        print(f"Generate new prompt from: \"{prompt}\" with {current.name}")
        input_ids = current.tokenizer(prompt, return_tensors='pt').input_ids
        if input_ids.shape[1] == 0:
            input_ids = torch.asarray(
                [[current.tokenizer.bos_token_id]], dtype=torch.long)
        output = current.model.generate(input_ids,
                                        do_sample=True,
                                        temperature=max(
                                            float(temperature), 1e-6),
                                        top_k=round(top_k),
                                        max_length=max_length,
                                        num_return_sequences=num_return_sequences,
                                        repetition_penalty=float(
                                            repetition_penalty),
                                        pad_token_id=current.tokenizer.pad_token_id or current.tokenizer.eos_token_id
                                        )
        print("Generation complete!")
        texts = current.tokenizer.batch_decode(
            output, skip_special_tokens=True)
        index = 0
        markup = ''
        for generated_text in texts:
            index += 1
            markup += f"""
    <div class="box" style="margin-bottom: var(--size-3);border: 1px solid var(--color-border-primary);border-radius: var(--radius-lg);background: var(--color-background-tertiary);color: var(--color-text-body);">
        <p id='prompt_res_{index}' style="font-size:var(--scale-0);padding:var(--size-2-5) var(--size-3)">{html.escape(generated_text)}</p>
    </div>
    """

        return markup
    except Exception as e:
        print(
            f"Exception encountered while attempting to generate prompt: {e}")
        return gr.update(), f"Error: {e}"


with gr.Blocks(analytics_enabled=0, title="MagicPrompt Generator") as magicprompt:

    gr.HTML("<h1 style='text-align:center;'>MagicPrompt Generator</h1>")
    with gr.Tab("Generator"):
        with gr.Row():
            with gr.Column(scale=80):
                text_input = gr.Textbox(
                    lines=2, show_label=False, value="",  placeholder="Enter your prompt...")
            with gr.Column(scale=10):
                submit = gr.Button('Generate', variant='primary')
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    temp_slider = gr.Slider(
                        elem_id="temp_slider", label="Temperature", interactive=True, minimum=0, maximum=4, value=1)
                    top_k_slider = gr.Slider(
                        elem_id="top_k_slider", label="Top K", value=12, minimum=1, maximum=50, step=1, interactive=True)
                    repetition_penalty = gr.Slider(
                        label="Repetition penalty", elem_id="repetition_penalty", value=1, minimum=1, maximum=4, step=0.01)
                with gr.Row():
                    min_length_slider = gr.Slider(
                        elem_id="min_length_slider", label="Min Length", interactive=True, minimum=1, maximum=400, step=1, value=20)
                    max_length_slider = gr.Slider(
                        elem_id="max_length_slider", label="Max Length", interactive=True, minimum=1, maximum=400, step=1, value=90)
                    num_return_sequences_slider = gr.Slider(
                        elem_id="num_return_sequences_slider", label="How Many To Generate", value=5, minimum=1, maximum=20, interactive=True, step=1)
                with gr.Row():
                    loading_models()
                    models_list = list(models.keys())
                    model_selection = gr.Dropdown(
                        label="Model", elem_id="prompt_model", interactive=True, value=models_list[0], choices=["None"] + models_list)
            with gr.Column():
                with gr.Row():
                    res = gr.HTML()

    submit.click(
        fn=generate,
        inputs=[model_selection, text_input, temp_slider, top_k_slider, min_length_slider,
                max_length_slider, repetition_penalty, num_return_sequences_slider],
        outputs=[res]
    )
    model_selection.change(
        fn=model_selection_changed,
        inputs=[model_selection],
        outputs=[],
    )

if __name__ == "__main__":
    magicprompt.queue(concurrency_count=20).launch(server_name="0.0.0.0", server_port=8090, show_api=False, debug=True)
