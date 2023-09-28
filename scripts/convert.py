import os
import tqdm
import torch
import safetensors.torch
from torch import Tensor
from modules import shared
from modules import sd_models, sd_vae

# position_ids in clip is int64. model_ema.num_updates is int32
dtypes_to_fp16 = {torch.float32, torch.float64, torch.bfloat16}
dtypes_to_bf16 = {torch.float32, torch.float64, torch.float16}


class MockModelInfo:
    def __init__(self, model_path: str) -> None:
        self.filepath = model_path
        self.filename: str = os.path.basename(model_path)
        self.model_name: str = self.filename.split(".")[0]


def conv_fp16(t: Tensor):
    return t.half() if t.dtype in dtypes_to_fp16 else t


def conv_bf16(t: Tensor):
    return t.bfloat16() if t.dtype in dtypes_to_bf16 else t


def conv_full(t):
    return t


_g_precision_func = {
    "full": conv_full,
    "fp32": conv_full,
    "fp16": conv_fp16,
    "bf16": conv_bf16,
}


def check_weight_type(k: str) -> str:
    if k.startswith("model.diffusion_model"):
        return "unet"
    elif k.startswith("first_stage_model"):
        return "vae"
    elif k.startswith("cond_stage_model"):
        return "clip"
    return "other"


def load_model(path):
    if path.endswith(".safetensors"):
        m = safetensors.torch.load_file(path, device="cpu")
    else:
        m = torch.load(path, map_location="cpu")
    state_dict = m["state_dict"] if "state_dict" in m else m
    return state_dict


def fix_model(model, fix_clip=False, force_position_id=False):
    # code from model-toolkit
    nai_keys = {
        'cond_stage_model.transformer.embeddings.': 'cond_stage_model.transformer.text_model.embeddings.',
        'cond_stage_model.transformer.encoder.': 'cond_stage_model.transformer.text_model.encoder.',
        'cond_stage_model.transformer.final_layer_norm.': 'cond_stage_model.transformer.text_model.final_layer_norm.'
    }
    position_id_key = "cond_stage_model.transformer.text_model.embeddings.position_ids"
    for k in list(model.keys()):
        for r in nai_keys:
            if type(k) == str and k.startswith(r):
                new_key = k.replace(r, nai_keys[r])
                model[new_key] = model[k]
                del model[k]
                print(f"[Converter] Fixed novelai error key {k}")
                break

    if force_position_id and position_id_key in model:
        model[position_id_key] = model[position_id_key].to(torch.int64)

    if fix_clip:
        if position_id_key in model:
            correct = torch.Tensor([list(range(77))]).to(torch.int64)
            now = model[position_id_key].to(torch.int64)

            broken = correct.ne(now)
            broken = [i for i in range(77) if broken[0][i]]
            if len(broken) != 0:
                model[position_id_key] = correct
                print(f"[Converter] Fixed broken clip\n{broken}")
            else:
                print("[Converter] Clip in this model is fine, skip fixing...")
        else:
            print("[Converter] Missing position id in model, try fixing...")
            model[position_id_key] = torch.Tensor([list(range(77))]).to(torch.int64)

    return model


def convert_warp(
        model_name, model_path, directory,
        *args
):
    if sum(map(bool, [model_name, model_path, directory])) != 1:
        print("[Converter] Check your inputs. Multiple input was set or missing input")
        return

    if directory != "":
        if not os.path.exists(directory) or not os.path.isdir(directory):
            return "Error: path not exists or not dir"

        files = [f for f in os.listdir(directory) if f.endswith(".ckpt") or f.endswith(".safetensors")]

        if len(files) == 0:
            return "Error: cant found model in directory"

        # remove custom filename in batch processing
        _args = list(args)
        _args[3] = ""

        for m in files:
            do_convert(MockModelInfo(os.path.join(directory, m)), *_args)

    elif model_path != "":
        if os.path.exists(model_path):
            return do_convert(MockModelInfo(model_path), *args)

    elif model_name != "":
        model_info = sd_models.checkpoints_list[model_name]
        return do_convert(MockModelInfo(model_info.filename), *args)

    else:
        return "Error: must choose a model"


def do_convert(model_info: MockModelInfo,
               checkpoint_formats,
               precision, conv_type, custom_name,
               bake_in_vae,
               unet_conv, text_encoder_conv, vae_conv, others_conv,
               fix_clip, force_position_id, delete_known_junk_data):
    if len(checkpoint_formats) == 0:
        return "Error: at least choose one model save format"

    extra_opt = {
        "unet": unet_conv,
        "clip": text_encoder_conv,
        "vae": vae_conv,
        "other": others_conv
    }
    shared.state.begin()
    shared.state.job = 'model-convert'
    shared.state.textinfo = f"Loading {model_info.filename}..."
    print(f"[Converter] Loading {model_info.filename}...")

    ok = {}
    state_dict = load_model(model_info.filepath)
    fix_model(state_dict, fix_clip=fix_clip, force_position_id=force_position_id)

    conv_func = _g_precision_func[precision]

    def _hf(wk: str, t: Tensor):
        if not isinstance(t, Tensor):
            return
        weight_type = check_weight_type(wk)
        conv_t = extra_opt[weight_type]
        if conv_t == "convert":
            ok[wk] = conv_func(t)
        elif conv_t == "copy":
            ok[wk] = t
        elif conv_t == "delete":
            return

    print("[Converter] Converting model...")

    if conv_type == "ema-only":
        for k in tqdm.tqdm(state_dict):
            ema_k = "___"
            try:
                ema_k = "model_ema." + k[6:].replace(".", "")
            except:
                pass
            if ema_k in state_dict:
                _hf(k, state_dict[ema_k])
                # print("ema: " + ema_k + " > " + k)
            elif not k.startswith("model_ema.") or k in ["model_ema.num_updates", "model_ema.decay"]:
                _hf(k, state_dict[k])
            #     print(k)
            # else:
            #     print("skipped: " + k)
    elif conv_type == "no-ema":
        for k, v in tqdm.tqdm(state_dict.items()):
            if "model_ema." not in k:
                _hf(k, v)
    else:
        for k, v in tqdm.tqdm(state_dict.items()):
            _hf(k, v)

    if delete_known_junk_data:
        known_junk_data_prefix = [
            "embedding_manager.embedder.",
            "lora_te_text_model",
            "control_model."
        ]
        need_delete = []
        for key in ok.keys():
            for jk in known_junk_data_prefix:
                if key.startswith(jk):
                    need_delete.append(key)

        for k in need_delete:
            del ok[k]

    bake_in_vae_filename = sd_vae.vae_dict.get(bake_in_vae, None)
    if bake_in_vae_filename is not None:
        print(f"[Converter] Baking in VAE from {bake_in_vae_filename}")
        vae_dict = sd_vae.load_vae_dict(bake_in_vae_filename, map_location='cpu')

        for k, v in vae_dict.items():
            _hf(k, vae_dict[k])

        del vae_dict

    output = ""
    ckpt_dir = os.path.dirname(model_info.filepath)
    save_name = f"{model_info.model_name}-{precision}"
    if conv_type != "disabled":
        save_name += f"-{conv_type}"

    if fix_clip:
        save_name += f"-clip-fix"

    if custom_name != "":
        save_name = custom_name

    for fmt in checkpoint_formats:
        ext = ".safetensors" if fmt == "safetensors" else ".ckpt"
        _save_name = save_name + ext

        save_path = os.path.join(ckpt_dir, _save_name)
        print(f"[Converter] Saving to {save_path}...")

        if fmt == "safetensors":
            safetensors.torch.save_file(ok, save_path)
        else:
            torch.save({"state_dict": ok}, save_path)
        output += f"Checkpoint saved to {save_path}\n"

    shared.state.end()
    return output[:-1]
