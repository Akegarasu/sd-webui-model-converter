# sd-webui-model-converter

Model convert extension , Used for [AUTOMATIC1111's stable diffusion webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
![image](https://github.com/Akegarasu/sd-webui-model-converter/assets/36563862/3f160408-6816-4fb5-9b27-1217126f5a6e)

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

If you use this extension to convert a model to fp16, and the model has an incorrect CLIP, the precision of the CLIP position_id may decrease during the compression process, which might coincidentally correct the offset.

![image](https://github.com/Akegarasu/sd-webui-model-converter/assets/36563862/d057d530-4b00-4937-ab30-8b8bd50fbd93)

If you do not wish to correct this CLIP offset coincidentally (because fixing it would alter the model, 
even though the correction is accurate, not everyone prefers the most correct, right? :P), 
you can use this option. It will force the CLIP position_id to be int64 and retain the incorrect CLIP