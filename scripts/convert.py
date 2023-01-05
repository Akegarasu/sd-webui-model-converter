import html
import os.path
import tqdm
import torch
import safetensors.torch
import gradio as gr
from torch import Tensor
from modules import script_callbacks, shared
from modules import sd_models
from modules.ui import create_refresh_button


def conv_fp16(t: Tensor):
    if not isinstance(t, Tensor):
        return t
    return t.half()


def conv_bf16(t: Tensor):
    if not isinstance(t, Tensor):
        return t
    return t.bfloat16()


def conv_full(t):
    return t


_g_precision_func = {
    "full": conv_full,
    "fp32": conv_full,
    "fp16": conv_fp16,
    "bf16": conv_bf16,
}


def add_tab():
    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                gr.HTML(value="<p>Converted checkpoints will be saved in your <b>checkpoint</b> directory.</p>")

                with gr.Row():
                    model_name = gr.Dropdown(sd_models.checkpoint_tiles(),
                                             elem_id="model_converter_model_name",
                                             label="Model")
                    create_refresh_button(model_name, sd_models.list_models,
                                          lambda: {"choices": sd_models.checkpoint_tiles()},
                                          "refresh_checkpoint_Z")

                custom_name = gr.Textbox(label="Custom Name (Optional)", elem_id="model_converter_custom_name")

                with gr.Row():
                    checkpoint_format = gr.Radio(choices=["ckpt", "safetensors"], value="ckpt",
                                                 label="Checkpoint format", elem_id="checkpoint_format")
                    precision = gr.Radio(choices=["fp32", "fp16", "bf16"], value="fp32",
                                         label="Precision", elem_id="checkpoint_precision")
                with gr.Row():
                    m_type = gr.Radio(choices=["all", "no-ema", "ema-only"], value="all",
                                      label="Model type")

                model_converter_convert = gr.Button(elem_id="model_converter_convert", label="Convert",
                                                    variant='primary')

            with gr.Column(variant='panel'):
                submit_result = gr.Textbox(elem_id="model_converter_result", show_label=False)

            model_converter_convert.click(
                fn=do_convert,
                inputs=[model_name, checkpoint_format, precision, m_type, custom_name],
                outputs=[submit_result]
            )

    return [(ui, "model convert", "model_convert")]


def load_model(path):
    if path.endswith(".safetensors"):
        m = safetensors.torch.load_file(path, device="cpu")
    else:
        m = torch.load(path, map_location="cpu")
    state_dict = m["state_dict"] if "state_dict" in m else m
    return state_dict


def do_convert(model, checkpoint_format, precision: str, conv_type: str, custom_name):
    shared.state.begin()
    shared.state.job = 'model-convert'

    model_info = sd_models.checkpoints_list[model]
    shared.state.textinfo = f"Loading {model_info.filename}..."
    print(f"Loading {model_info.filename}...")
    state_dict = load_model(model_info.filename)

    ok = {}  # {"state_dict": {}}
    _hf = _g_precision_func[precision]

    print("Converting model...")

    if conv_type == "ema-only" or conv_type == "prune":
        for k in tqdm.tqdm(state_dict):
            ema_k = "___"
            try:
                ema_k = "model_ema." + k[6:].replace(".", "")
            except:
                pass
            if ema_k in state_dict:
                ok[k] = _hf(state_dict[ema_k])
                print("ema: " + ema_k + " > " + k)
            elif not k.startswith("model_ema.") or k in ["model_ema.num_updates", "model_ema.decay"]:
                ok[k] = _hf(state_dict[k])
                print(k)
            else:
                print("skipped: " + k)
    elif conv_type == "no-ema":
        for k, v in tqdm.tqdm(state_dict.items()):
            if "model_ema" not in k:
                ok[k] = _hf(v)
    else:
        for k, v in tqdm.tqdm(state_dict.items()):
            ok[k] = _hf(v)

    ckpt_dir = shared.cmd_opts.ckpt_dir or sd_models.model_path
    save_name = custom_name if custom_name != "" else f"{model_info.model_name}-{precision}-{conv_type}"
    save_name += ".safetensors" if checkpoint_format == "safetensors" else ".ckpt"

    save_path = os.path.join(ckpt_dir, save_name)
    print(f"Saving to {save_path}...")

    if checkpoint_format == "safetensors":
        safetensors.torch.save_file(ok, save_path)
    else:
        torch.save({"state_dict": ok}, save_path)

    shared.state.end()
    return "Checkpoint saved to " + save_path


script_callbacks.on_ui_tabs(add_tab)
