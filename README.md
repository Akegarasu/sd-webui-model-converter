# sd-webui-model-converter

Model convert extension , Used for [AUTOMATIC1111's stable diffusion webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
![image](https://user-images.githubusercontent.com/36563862/211143190-1bbf4fa9-5d95-4d00-b4dc-0f617caf3312.png)

## Features 

- convert to precisions: fp32, fp16, bf16
- pruning model: no-ema, ema-only
- checkpoint ext convert: ckpt, safetensors
- convert/copy/delete any parts of model: unet, text encoder(clip), vae
- Fix CLIP
- Force CLIP position_id to int64 before convert

### Fix CLIP

Sometimes, the CLIP position_id becomes incorrect due to model merging.
For example, Anything-v3.

This option will reset CLIP position to `torch.Tensor([list(range(77))]).to(torch.int64)`


### Force CLIP position_id to int64 before convert

If you use this extension to convert a model to fp16, which has an incorrect CLIP,
the precision of the CLIP position_id may decrease during the compression process,
which may coincidentally fix the offset.

![image](https://github.com/Akegarasu/sd-webui-model-converter/assets/36563862/d057d530-4b00-4937-ab30-8b8bd50fbd93)

If you do not want to fix this CLIP offset coincidentally (because fixing it would change the model, even though the fix is correct, not everyone likes the most correct, right :P ), use this option. It will force the CLIP position_id to int64,
and keep the incorrect CLIP.