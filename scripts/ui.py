import gradio as gr
from modules import script_callbacks
from modules import sd_models, sd_vae
from modules.ui import create_refresh_button
from scripts import convert


def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}


def add_tab():
    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Row(equal_height=True):
            with gr.Column(variant='panel'):
                gr.HTML(value="<p>Converted checkpoints will be saved in your <b>checkpoint</b> directory.</p>")
                with gr.Tabs():
                    with gr.TabItem(label='Single process'):
                        with gr.Row():
                            model_name = gr.Dropdown(sd_models.checkpoint_tiles(),
                                                     elem_id="model_converter_model_name",
                                                     label="Model")
                            create_refresh_button(model_name, sd_models.list_models,
                                                  lambda: {"choices": sd_models.checkpoint_tiles()},
                                                  "refresh_checkpoint_Z")
                        custom_name = gr.Textbox(label="Custom Name (Optional)")

                    with gr.TabItem(label='Input file path'):
                        with gr.Row():
                            model_path = gr.Textbox(label="model path")

                    with gr.TabItem(label='Batch from directory'):
                        with gr.Row():
                            input_directory = gr.Textbox(label="Input Directory")

                with gr.Row():
                    precision = gr.Radio(choices=["fp32", "fp16", "bf16"], value="fp16", label="Precision")
                    m_type = gr.Radio(choices=["disabled", "no-ema", "ema-only"], value="disabled", label="Pruning Methods")

                with gr.Row():
                    checkpoint_formats = gr.CheckboxGroup(choices=["ckpt", "safetensors"], value=["safetensors"], label="Checkpoint Format")
                    show_extra_options = gr.Checkbox(label="Show extra options", value=False)

                with gr.Row():
                    bake_in_vae = gr.Dropdown(choices=["None"] + list(sd_vae.vae_dict), value="None", label="Bake in VAE")
                    create_refresh_button(bake_in_vae, sd_vae.refresh_vae_list, lambda: {"choices": ["None"] + list(sd_vae.vae_dict)}, "model_converter_refresh_bake_in_vae")

                with gr.Row():
                    force_position_id = gr.Checkbox(label="Force CLIP position_id to int64 before convert", value=True)
                    fix_clip = gr.Checkbox(label="Fix clip", value=False)
                    delete_known_junk_data = gr.Checkbox(label="Delete known junk data", value=False)

                with gr.Row(visible=False) as extra_options:
                    specific_part_conv = ["copy", "convert", "delete"]
                    unet_conv = gr.Dropdown(specific_part_conv, value="convert", label="unet")
                    text_encoder_conv = gr.Dropdown(specific_part_conv, value="convert", label="text encoder")
                    vae_conv = gr.Dropdown(specific_part_conv, value="convert", label="vae")
                    others_conv = gr.Dropdown(specific_part_conv, value="convert", label="others")

                model_converter_convert = gr.Button(elem_id="model_converter_convert", label="Convert",
                                                    variant='primary')

            with gr.Column(variant='panel'):
                submit_result = gr.Textbox(elem_id="model_converter_result", show_label=False)

            show_extra_options.change(
                fn=lambda x: gr_show(x),
                inputs=[show_extra_options],
                outputs=[extra_options],
            )

            model_converter_convert.click(
                fn=convert.convert_warp,
                inputs=[
                    model_name,
                    model_path,
                    input_directory,
                    checkpoint_formats,
                    precision, m_type, custom_name,
                    bake_in_vae,
                    unet_conv,
                    text_encoder_conv,
                    vae_conv,
                    others_conv,
                    fix_clip,
                    force_position_id,
                    delete_known_junk_data
                ],
                outputs=[submit_result]
            )

    return [(ui, "Model Converter", "model_converter")]


script_callbacks.on_ui_tabs(add_tab)
