from calendar import c
import os
import math
import gradio as gr
import numpy as np
import torch
import safetensors.torch as sf
import cv2
from datetime import datetime


from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from briarmbg import BriaRMBG
from enum import Enum
from torch.hub import download_url_to_file


class BGSource2(Enum):
    UPLOAD = "Use Background Image"
    UPLOAD_FLIP = "Use Flipped Background Image"
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"
    GREY = "Ambient"

now=os.getcwd()#获得当前工作目录

sd_tokenizer_path='./models/tokenizer/'
tokenizer = None

text_encoder_path='./models/text_encoder/'
text_encoder = None

unet_path='./models/unet/'
unet = None

vae_path='./models/vae/'
vae = None

rmbg_path='./models/rmbg/'
rmbg = None
unet_original_forward,device,t2i_pipe,i2i_pipe=None,None,None,None

base_model_path='./models/iclight_sd15_fbc.safetensors'
sd_offset=None


@torch.inference_mode()
def process_relight2(input_fg, input_bg, input_bg_only, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source):
    # print(input_fg.shape)
    input_fg, matting = run_rmbg(input_fg)
    # save matting
    # cv2.imwrite(os.path.join('/media/magic-4090/47236903-9d2a-4bc7-9828-df4fa4b40bd0/user/blender_workspace/IC-Light', 'matting.png'), (matting * 255.0).clip(0, 255).astype(np.uint8))
    # print(input_fg.shape)
    # input_fg = input_fg
    # cv2.imwrite(os.path.join('/media/magic-4090/47236903-9d2a-4bc7-9828-df4fa4b40bd0/user/blender_workspace/IC-Light', 'input_fg.png'), input_fg)
    
    results, extra_images = process(input_fg, input_bg, input_bg_only, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source)
    # save extra fg and bg
    
    results = [(x * 255.0).clip(0, 255).astype(np.uint8) for x in results]
    return results + extra_images


def change_state2(if_used):
    global base_model_path,sd_offset,sd_tokenizer_path,tokenizer,text_encoder_path,text_encoder,unet_path,unet,vae_path,vae,rmbg_path,rmbg
    global unet_original_forward,device,t2i_pipe,i2i_pipe
    if if_used:
        tokenizer = CLIPTokenizer.from_pretrained(sd_tokenizer_path)
        text_encoder = CLIPTextModel.from_pretrained(text_encoder_path)
        unet = UNet2DConditionModel.from_pretrained(unet_path)
        vae = AutoencoderKL.from_pretrained(vae_path)
        rmbg = BriaRMBG.from_pretrained(rmbg_path)
        with torch.no_grad():
            new_conv_in = torch.nn.Conv2d(12, unet.conv_in.out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding)
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
            new_conv_in.bias = unet.conv_in.bias
            unet.conv_in = new_conv_in
        unet_original_forward = unet.forward
        unet.forward = hooked_unet_forward
        sd_offset = sf.load_file(base_model_path)
        sd_origin = unet.state_dict()
        keys = sd_origin.keys()
        sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
        unet.load_state_dict(sd_merged, strict=True)
        del sd_offset, sd_origin, sd_merged, keys
        device = torch.device('cuda')
        text_encoder = text_encoder.to(device=device, dtype=torch.float16)
        vae = vae.to(device=device, dtype=torch.bfloat16)
        unet = unet.to(device=device, dtype=torch.float16)
        rmbg = rmbg.to(device=device, dtype=torch.float32)
        unet.set_attn_processor(AttnProcessor2_0())
        vae.set_attn_processor(AttnProcessor2_0())

        # Samplers

        ddim_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        euler_a_scheduler = EulerAncestralDiscreteScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            steps_offset=1
        )

        dpmpp_2m_sde_karras_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
            steps_offset=1
        )

        # Pipelines

        t2i_pipe = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=dpmpp_2m_sde_karras_scheduler,
            safety_checker=None,
            requires_safety_checker=False,
            feature_extractor=None,
            image_encoder=None
        )

        i2i_pipe = StableDiffusionImg2ImgPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=dpmpp_2m_sde_karras_scheduler,
            safety_checker=None,
            requires_safety_checker=False,
            feature_extractor=None,
            image_encoder=None
        )
        print("加载模型成功！")
    else:
        tokenizer = None
        text_encoder = None
        unet = None
        vae = None
        rmbg = None
        unet_original_forward = None
        device = None
        t2i_pipe = None
        i2i_pipe = None
        torch.cuda.empty_cache()
        print("卸载模型！")
    
def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
    c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
    c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
    new_sample = torch.cat([sample, c_concat], dim=1)
    kwargs['cross_attention_kwargs'] = {}
    return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)

@torch.inference_mode()
def encode_prompt_inner(txt: str):
    max_length = tokenizer.model_max_length
    chunk_length = tokenizer.model_max_length - 2
    id_start = tokenizer.bos_token_id
    id_end = tokenizer.eos_token_id
    id_pad = id_end

    def pad(x, p, i):
        return x[:i] if len(x) >= i else x + [p] * (i - len(x))

    tokens = tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
    chunks = [[id_start] + tokens[i: i + chunk_length] + [id_end] for i in range(0, len(tokens), chunk_length)]
    chunks = [pad(ck, id_pad, max_length) for ck in chunks]

    token_ids = torch.tensor(chunks).to(device=device, dtype=torch.int64)
    conds = text_encoder(token_ids).last_hidden_state

    return conds


@torch.inference_mode()
def encode_prompt_pair(positive_prompt, negative_prompt):
    c = encode_prompt_inner(positive_prompt)
    uc = encode_prompt_inner(negative_prompt)

    c_len = float(len(c))
    uc_len = float(len(uc))
    max_count = max(c_len, uc_len)
    c_repeat = int(math.ceil(max_count / c_len))
    uc_repeat = int(math.ceil(max_count / uc_len))
    max_chunk = max(len(c), len(uc))

    c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
    uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]

    c = torch.cat([p[None, ...] for p in c], dim=1)
    uc = torch.cat([p[None, ...] for p in uc], dim=1)

    return c, uc


@torch.inference_mode()
def pytorch2numpy(imgs, quant=True):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)

        if quant:
            y = y * 127.5 + 127.5
            y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        else:
            y = y * 0.5 + 0.5
            y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)

        results.append(y)
    return results


@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0  # so that 127 must be strictly 0.0
    h = h.movedim(-1, 1)
    return h


def resize_and_center_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    return np.array(cropped_image)


def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)


@torch.inference_mode()
def run_rmbg(img, sigma=0.0):
    alpha = img[:,:,3]
    # 将透明度为0的像素点的RGB值改为
    result = img[:,:,0:3]
    # result[alpha == 0] = (127,127,127)
    # 针对Image对象进行操作
    result[alpha == 0] = (127,127,127)
    return result.clip(0, 255).astype(np.uint8), alpha
# @torch.inference_mode()
# def run_rmbg(img, sigma=0.0):
#     H, W, C = img.shape
#     k = (256.0 / float(H * W)) ** 0.5
#     img = img[:,:,:3]
#     feed = resize_without_crop(img, int(64 * round(W * k)), int(64 * round(H * k)))
#     feed = numpy2pytorch([feed]).to(device=device, dtype=torch.float32)
#     alpha = rmbg(feed)[0][0]
#     alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear")
#     alpha = alpha.movedim(1, -1)[0]
#     alpha = alpha.detach().float().cpu().numpy().clip(0, 1)
#     result = 127 + (img.astype(np.float32) - 127) * alpha
#     return result.clip(0, 255).astype(np.uint8), alpha

@torch.inference_mode()
def process(input_fg, input_bg, input_bg_only, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source):
    global BGSource2
    bg_source = BGSource2(bg_source)
    global vae, t2i_pipe, i2i_pipe
    rng = torch.Generator(device=device).manual_seed(seed)
    fg = resize_and_center_crop(input_fg, image_width, image_height)
    # save fg
    fg_bgr = cv2.cvtColor(fg, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join('/media/magic-4090/47236903-9d2a-4bc7-9828-df4fa4b40bd0/user/blender_workspace/IC-Light', 'fg.png'), fg_bgr)
    bg = resize_and_center_crop(input_bg, image_width, image_height)
    bg_only = resize_and_center_crop(input_bg_only, image_width, image_height)
    concat_conds = numpy2pytorch([fg,bg_only]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor
    concat_conds = torch.cat([c[None, ...] for c in concat_conds], dim=1)

    conds, unconds = encode_prompt_pair(positive_prompt=prompt + ', ' + a_prompt, negative_prompt=n_prompt)
    # 将input_fg 转换为1*3*1600*2432的tensor
    input_fg_tensor = [fg]
    pixels = numpy2pytorch(input_fg_tensor).to(device=vae.device, dtype=vae.dtype)
    latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor
    latents = latents.to(device=unet.device, dtype=unet.dtype)

    latents = i2i_pipe(
        image=latents,
        strength=highres_denoise,
        prompt_embeds=conds,
        negative_prompt_embeds=unconds,
        width=image_width,
        height=image_height,
        num_inference_steps=int(round(steps / highres_denoise)),
        num_images_per_prompt=num_samples,
        generator=rng,
        output_type='latent',
        guidance_scale=cfg,
        cross_attention_kwargs={'concat_conds': concat_conds},
    ).images.to(vae.dtype) / vae.config.scaling_factor

    # concat_conds = numpy2pytorch([fg,bg_only]).to(device=vae.device, dtype=vae.dtype)
    # concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor
    # concat_conds = torch.cat([c[None, ...] for c in concat_conds], dim=1)
    # latents = i2i_pipe(
    #     image=latents,
    #     strength=highres_denoise,
    #     prompt_embeds=conds,
    #     negative_prompt_embeds=unconds,
    #     width=image_width,
    #     height=image_height,
    #     num_inference_steps=int(round(steps / highres_denoise)),
    #     num_images_per_prompt=num_samples,
    #     generator=rng,
    #     output_type='latent',
    #     guidance_scale=cfg,
    #     cross_attention_kwargs={'concat_conds': concat_conds},
    # ).images.to(vae.dtype) / vae.config.scaling_factor

    # pixels = vae.decode(latents).sample
    # latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor
    # latents = latents.to(device=unet.device, dtype=unet.dtype)

    # # 再针对目前的latents进行一次处理
    # latents = i2i_pipe(
    #     image=latents,
    #     strength=lowres_denoise,
    #     prompt_embeds=conds,
    #     negative_prompt_embeds=unconds,
    #     width=image_width,
    #     height=image_height,
    #     num_inference_steps=int(round(steps / lowres_denoise)),
    #     num_images_per_prompt=num_samples,
    #     generator=rng,
    #     output_type='latent',
    #     guidance_scale=cfg,
    #     cross_attention_kwargs={'concat_conds': concat_conds},
    # ).images.to(vae.dtype) / vae.config.scaling_factor

    pixels = vae.decode(latents).sample
    
    numpy_image=pytorch2numpy(pixels)
    img = Image.fromarray(numpy_image[0])
    output_dir ="./outputs"
    os.makedirs(output_dir, exist_ok=True)
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    img.save(os.path.join(output_dir, f"img_{timestamp}.png"))
    pixels = pytorch2numpy(pixels, quant=False)


    return pixels, [fg, bg]

        

