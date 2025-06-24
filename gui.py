import gradio as gr
import subprocess
import sys
import os
import shlex
import toml
import json
from tkinter import filedialog, Tk
import functools

# --- Helper Functions (Unchanged) ---
def run_script(command_list, cwd=".", stream_to_gradio=True):
    python_executable = sys.executable
    if command_list[0] in ["uv", "huggingface-cli"]:
        venv_scripts_path = os.path.join(os.path.dirname(python_executable))
        executable_path = os.path.join(venv_scripts_path, command_list[0])
        if sys.platform == "win32": executable_path += ".exe"
        final_command = [executable_path] + command_list[1:] if os.path.exists(executable_path) else command_list
    else:
        final_command = [python_executable] + command_list
    print(f"Executing in '{os.path.abspath(cwd)}': {' '.join(shlex.quote(c) for c in final_command)}")
    process = subprocess.Popen(
        final_command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding='utf-8', errors='replace', bufsize=1,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    )
    full_log = "--- Starting Process ---\n"
    if stream_to_gradio: yield full_log
    for line in iter(process.stdout.readline, ''):
        line = line.strip()
        full_log += line + "\n"
        if stream_to_gradio: yield full_log
    process.stdout.close()
    return_code = process.wait()
    result_message = f"--- Process finished with {'error code ' + str(return_code) if return_code else 'success'} ---"
    full_log += result_message + "\n"
    if stream_to_gradio: yield full_log
    else: return full_log

def select_folder(current_path=""):
    root = Tk(); root.withdraw(); root.attributes('-topmost', True)
    folder_path = filedialog.askdirectory(initialdir=current_path if current_path and os.path.isdir(current_path) else os.getcwd())
    root.destroy()
    return folder_path if folder_path else current_path

# --- Dataset TOML Builder Logic (Unchanged) ---
dataset_list_state = []
def add_dataset_entry(gen_res_w, gen_res_h, gen_caption_ext, gen_batch_size, gen_repeats, gen_bucket, gen_no_upscale, ds_type, ds_source, ds_path, ds_cache_path, ds_control_path, ds_target_frames, ds_frame_extraction, ds_max_frames, ds_source_fps):
    global dataset_list_state
    new_dataset = {"type": ds_type, "source": ds_source, "path": ds_path, "cache_path": ds_cache_path, "control_path": ds_control_path, "target_frames": ds_target_frames, "frame_extraction": ds_frame_extraction, "max_frames": ds_max_frames, "source_fps": ds_source_fps}
    dataset_list_state.append(new_dataset)
    return generate_toml_preview(gen_res_w, gen_res_h, gen_caption_ext, gen_batch_size, gen_repeats, gen_bucket, gen_no_upscale)
def generate_toml_preview(gen_res_w, gen_res_h, gen_caption_ext, gen_batch_size, gen_repeats, gen_bucket, gen_no_upscale):
    global dataset_list_state; config = {}; general = {}
    if gen_res_w and gen_res_h: general['resolution'] = [gen_res_w, gen_res_h]
    if gen_caption_ext: general['caption_extension'] = gen_caption_ext
    if gen_batch_size: general['batch_size'] = gen_batch_size
    if gen_repeats: general['num_repeats'] = gen_repeats
    general['enable_bucket'] = gen_bucket; general['bucket_no_upscale'] = gen_no_upscale
    if general: config['general'] = general
    datasets = []
    for ds in dataset_list_state:
        dataset_entry = {};
        if ds['type'] == 'Image': key = 'image_jsonl_file' if ds['source'] == 'JSONL' else 'image_directory'; dataset_entry[key] = ds['path']
        elif ds['type'] == 'Video':
            key = 'video_jsonl_file' if ds['source'] == 'JSONL' else 'video_directory'; dataset_entry[key] = ds['path']
            if ds['target_frames']:
                try: dataset_entry['target_frames'] = [int(x.strip()) for x in ds['target_frames'].split(',')]
                except: pass
            if ds['frame_extraction']: dataset_entry['frame_extraction'] = ds['frame_extraction']
            if ds['max_frames']: dataset_entry['max_frames'] = ds['max_frames']
            if ds['source_fps']: dataset_entry['source_fps'] = ds['source_fps']
        if ds['cache_path']: dataset_entry['cache_directory'] = ds['cache_path']
        if ds['control_path']: dataset_entry['control_directory'] = ds['control_path']
        datasets.append(dataset_entry)
    if datasets: config['datasets'] = datasets
    return toml.dumps(config)
def clear_all_datasets():
    global dataset_list_state; dataset_list_state = []
    return generate_toml_preview(960, 544, ".txt", 1, 1, True, False)
def save_toml_file(toml_content, save_path):
    if not save_path.strip(): return "Error: Save path cannot be empty.", ""
    if not save_path.endswith(".toml"): save_path += ".toml"
    try:
        if os.path.dirname(save_path): os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f: f.write(toml_content)
        return f"Successfully saved to {save_path}", save_path
    except Exception as e: return f"Error saving file: {e}", ""
def copy_path_to_tabs(filepath):
    return gr.update(value=filepath), gr.update(value=filepath)

# --- Installer & Other Logic ---
def download_model(model_choice, model_type):
    if not model_choice or model_choice == "n": yield "Skipping download."; return
    main_commands = {
        "hy": {"1": ["huggingface-cli", "download", "tencent/HunyuanVideo", "--local-dir", "./ckpts", "--exclude", "*mp_rank_00_model_states_fp8*"], "2": ["huggingface-cli", "download", "tencent/HunyuanVideo-I2V", "--local-dir", "./ckpts", "--include", "*hunyuan-video-i2v-720p*"]},
        "wan": {"1": ["huggingface-cli", "download", "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors", "--local-dir-use-symlinks", "False", "--local-dir", "./ckpts/wan"], "2": ["huggingface-cli", "download", "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/diffusion_models/wan2.1_t2v_14B_fp16.safetensors", "--local-dir-use-symlinks", "False", "--local-dir", "./ckpts/wan"], "3": ["huggingface-cli", "download", "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/diffusion_models/wan2.1_i2v_480p_14B_fp16.safetensors", "--local-dir-use-symlinks", "False", "--local-dir", "./ckpts/wan"], "4": ["huggingface-cli", "download", "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/diffusion_models/wan2.1_i2v_720p_14B_fp16.safetensors", "--local-dir-use-symlinks", "False", "--local-dir", "./ckpts/wan"], "5": ["huggingface-cli", "download", "alibaba-pai/Wan2.1-Fun-1.3B-Control", "diffusion_pytorch_model.safetensors", "--local-dir", "./ckpts/wan-1.3B-FC"], "6": ["huggingface-cli", "download", "alibaba-pai/Wan2.1-Fun-14B-Control", "diffusion_pytorch_model.safetensors", "--local-dir", "./ckpts/wan-14B-FC"]},
        "fp": {"1": ["huggingface-cli", "download", "Kijai/HunyuanVideo_comfy", "FramePackI2V_HY_bf16.safetensors", "--local-dir", "./ckpts/framepack"], "2": ["huggingface-cli", "download", "kabachuha/FramePack_F1_I2V_HY_20250503_comfy", "FramePack_F1_I2V_HY_20250503.safetensors", "--local-dir", "./ckpts/framepack"]}
    }
    dependencies = {
        "hy": [("./ckpts/text_encoder/llava_llama3_fp16.safetensors", ["huggingface-cli", "download", "Comfy-Org/HunyuanVideo_repackaged", "split_files/text_encoders/llava_llama3_fp16.safetensors", "--local-dir-use-symlinks", "False", "--local-dir", "./ckpts/text_encoder"]),("./ckpts/text_encoder_2/clip_l.safetensors", ["huggingface-cli", "download", "Comfy-Org/HunyuanVideo_repackaged", "split_files/text_encoders/clip_l.safetensors", "--local-dir-use-symlinks", "False", "--local-dir", "./ckpts/text_encoder_2"])],
        "wan_i2v_fc": [("./ckpts/text_encoder_2/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", ["huggingface-cli", "download", "Wan-AI/Wan2.1-I2V-14B-720P", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", "--local-dir", "./ckpts/text_encoder_2"])],
        "wan_common": [("./ckpts/text_encoder/models_t5_umt5-xxl-enc-bf16.pth", ["huggingface-cli", "download", "Wan-AI/Wan2.1-T2V-14B", "models_t5_umt5-xxl-enc-bf16.pth", "--local-dir", "./ckpts/text_encoder"]),("./ckpts/vae/Wan2.1_VAE.pth", ["huggingface-cli", "download", "Wan-AI/Wan2.1-T2V-14B", "Wan2.1_VAE.pth", "--local-dir", "./ckpts/vae"])],
        "fp": [("./ckpts/framepack/hunyuan_video_vae_fp32.safetensors", ["huggingface-cli", "download", "Kijai/HunyuanVideo_comfy", "hunyuan_video_vae_fp32.safetensors", "--local-dir", "./ckpts/framepack"]),("./ckpts/text_encoder/llava_llama3_fp16.safetensors", ["huggingface-cli", "download", "Comfy-Org/HunyuanVideo_repackaged", "split_files/text_encoders/llava_llama3_fp16.safetensors", "--local-dir-use-symlinks", "False", "--local-dir", "./ckpts/text_encoder"]),("./ckpts/text_encoder_2/clip_l.safetensors", ["huggingface-cli", "download", "Comfy-Org/HunyuanVideo_repackaged", "split_files/text_encoders/clip_l.safetensors", "--local-dir-use-symlinks", "False", "--local-dir", "./ckpts/text_encoder_2"]),("./ckpts/framepack/sigclip_vision_patch14_384.safetensors", ["huggingface-cli", "download", "Comfy-Org/sigclip_vision_384", "sigclip_vision_patch14_384.safetensors", "--local-dir", "./ckpts/framepack"])]
    }
    command_to_run = main_commands.get(model_type, {}).get(model_choice)
    if not command_to_run: yield f"Invalid model choice '{model_choice}' for type '{model_type}'."; return
    yield f"--- Downloading {model_type.upper()} Model: {model_choice} ---\n"
    for line in run_script(command_to_run): yield line
    deps_to_check = []
    if model_type == 'hy': deps_to_check = dependencies['hy']
    elif model_type == 'wan':
        if model_choice in ['3', '4', '5', '6']: deps_to_check.extend(dependencies['wan_i2v_fc'])
        deps_to_check.extend(dependencies['wan_common'])
    elif model_type == 'fp': deps_to_check = dependencies['fp']
    if deps_to_check:
        yield "\n--- Checking for required support files... ---\n"
        for check_path, download_cmd in deps_to_check:
            yield f"Checking for: {os.path.basename(check_path)}"
            if not os.path.exists(check_path):
                yield f"-> Not found. Downloading required file: {os.path.basename(check_path)}"
                for line in run_script(download_cmd): yield line
            else: yield f"-> Found. Skipping download."
    yield "\n--- All downloads and checks complete. ---"
def caching_logic(cache_mode, dataset_config, vae, skip_existing, vae_cache_cpu, clip, fp8_t5, t5, image_encoder, text_encoder1, text_encoder2):
    script_path_prefix = ""
    if cache_mode == "Wan": script_path_prefix = "wan_"
    elif cache_mode == "FramePack": script_path_prefix = "fpack_"
    latent_script = f"./musubi-tuner/{script_path_prefix}cache_latents.py"; latent_args = ["--dataset_config", dataset_config, "--vae", vae]
    if skip_existing: latent_args.append("--skip_existing")
    if cache_mode == "Wan":
        if vae_cache_cpu: latent_args.append("--vae_cache_cpu")
        if clip: latent_args.extend(["--clip", clip])
    elif cache_mode == "FramePack":
        if image_encoder: latent_args.extend(["--image_encoder", image_encoder])
    full_log = "--- Caching Latents ---\n";
    for line in run_script([latent_script] + latent_args): full_log += line + "\n"; yield full_log
    text_script = f"./musubi-tuner/{script_path_prefix}cache_text_encoder_outputs.py"; text_args = ["--dataset_config", dataset_config]
    if cache_mode == "Wan":
        if t5: text_args.extend(["--t5", t5])
        if fp8_t5: text_args.append("--fp8_t5")
    elif cache_mode in ["HunyuanVideo", "FramePack"]:
        if text_encoder1: text_args.extend(["--text_encoder1", text_encoder1])
        if text_encoder2: text_args.extend(["--text_encoder2", text_encoder2])
    full_log += "\n--- Caching Text Encoder Outputs ---\n";
    for line in run_script([text_script] + text_args): full_log += line + "\n"; yield full_log
def training_logic(*args):
    keys = [
        "train_mode", "dataset_config", "dit", "vae", "text_encoder1", "text_encoder2", "t5", "clip", "image_encoder",
        "max_train_epochs", "gradient_checkpointing", "gradient_accumulation_steps", "seed",
        "lr", "lr_scheduler", "lr_warmup_steps", "lr_decay_steps", "lr_scheduler_num_cycles", "lr_scheduler_power", "lr_scheduler_min_lr_ratio",
        "network_dim", "network_alpha", "network_dropout",
        "mixed_precision", "attn_mode",
        "guidance_scale", "timestep_sampling", "discrete_flow_shift", "sigmoid_scale", "weighting_scheme", "logit_mean", "logit_std", "mode_scale",
        "enable_lora_plus", "loraplus_lr_ratio", "enable_lycoris", "conv_dim", "conv_alpha", "lyco_algo", "lyco_dropout", "lyco_preset",
        "output_name", "save_every_n_epochs", "save_every_n_steps", "save_state",
        "optimizer_type", "max_grad_norm",
        "enable_sample", "sample_prompts", "sample_every_n_epochs",
        "training_comment"
    ]
    params = dict(zip(keys, args))
    laungh_script, network_module = "", ""
    if params["train_mode"] == "db": laungh_script = "hv_train.py"
    elif "Lora" in params["train_mode"]:
        if "HunyuanVideo" in params["train_mode"]: laungh_script, network_module = "hv_train_network.py", "networks.lora"
        elif "Wan" in params["train_mode"]: laungh_script, network_module = "wan_train_network.py", "networks.lora_wan"
        elif "FramePack" in params["train_mode"]: laungh_script, network_module = "fpack_train_network.py", "networks.lora_framepack"
    if not laungh_script: yield "Invalid training mode."; return
    script_path = f"./musubi-tuner/{laungh_script}"; launch_args = ["-m", "accelerate.commands.launch", "--num_cpu_threads_per_process=8"]
    if params["mixed_precision"] != 'no':
        launch_args.extend(["--mixed_precision", params["mixed_precision"]])
        if 'bf16' in params["mixed_precision"]: launch_args.append("--downcast_bf16")
    script_args = [
        "--dataset_config", params["dataset_config"], "--dit", params["dit"], "--vae", params["vae"], "--seed", str(params["seed"]),
        "--output_dir", "./output_dir", "--logging_dir", "./logs",
        "--max_train_epochs", str(params["max_train_epochs"]),
        "--gradient_accumulation_steps", str(params["gradient_accumulation_steps"]),
        "--learning_rate", str(params["lr"]), "--lr_scheduler", params["lr_scheduler"],
        "--optimizer_type", params["optimizer_type"], "--output_name", params["output_name"],
    ]
    if params["save_every_n_epochs"] > 0: script_args.extend(["--save_every_n_epochs", str(params["save_every_n_epochs"])])
    if params["gradient_checkpointing"]: script_args.append("--gradient_checkpointing")
    if params["attn_mode"] != "torch": script_args.append(f'--{params["attn_mode"]}')
    if "HunyuanVideo" in params["train_mode"] or "FramePack" in params["train_mode"] or params["train_mode"] == "db": script_args.extend(["--text_encoder1", params["text_encoder1"], "--text_encoder2", params["text_encoder2"]])
    if "Wan" in params["train_mode"]:
        if params["t5"]: script_args.extend(["--t5", params["t5"]])
        if params["clip"]: script_args.extend(["--clip", params["clip"]])
    if "FramePack" in params["train_mode"] and params["image_encoder"]: script_args.extend(["--image_encoder", params["image_encoder"]])
    if params["guidance_scale"] != 1.0: script_args.extend(["--guidance_scale", str(params["guidance_scale"])])
    if params["timestep_sampling"] != "sigma": script_args.extend(["--timestep_sampling", params["timestep_sampling"]])
    if params["timestep_sampling"] == "shift" and params["discrete_flow_shift"] != 1.0: script_args.extend(["--discrete_flow_shift", str(params["discrete_flow_shift"])])
    if params["timestep_sampling"] in ["sigmoid", "shift"] and params["sigmoid_scale"] != 1.0: script_args.extend(["--sigmoid_scale", str(params["sigmoid_scale"])])
    if params["weighting_scheme"] != "none":
        script_args.extend(["--weighting_scheme", params["weighting_scheme"]])
        if params["weighting_scheme"] == "logit_normal":
            if params["logit_mean"] != 0.0: script_args.extend(["--logit_mean", str(params["logit_mean"])])
            if params["logit_std"] != 1.0: script_args.extend(["--logit_std", str(params["logit_std"])])
        elif params["weighting_scheme"] == "mode" and params["mode_scale"] != 1.29: script_args.extend(["--mode_scale", str(params["mode_scale"])])
    if params["lr_warmup_steps"] > 0: script_args.extend(["--lr_warmup_steps", str(params["lr_warmup_steps"])])
    if params["lr_decay_steps"] > 0: script_args.extend(["--lr_decay_steps", str(params["lr_decay_steps"])])
    if params["lr_scheduler_num_cycles"] > 1: script_args.extend(["--lr_scheduler_num_cycles", str(params["lr_scheduler_num_cycles"])])
    if params["lr_scheduler_power"] != 1.0: script_args.extend(["--lr_scheduler_power", str(params["lr_scheduler_power"])])
    if params["lr_scheduler_min_lr_ratio"] > 0: script_args.extend(["--lr_scheduler_min_lr_ratio", str(params["lr_scheduler_min_lr_ratio"])])
    if params["max_grad_norm"] != 1.0: script_args.extend(["--max_grad_norm", str(params["max_grad_norm"])])
    if "Lora" in params["train_mode"]:
        script_args.extend(["--network_dim", str(params["network_dim"]), "--network_alpha", str(params["network_alpha"])])
        if params["network_dropout"] > 0: script_args.extend(["--network_dropout", str(params["network_dropout"])])
        network_args_list = []
        if params["enable_lycoris"]:
            network_module = "lycoris.kohya"; network_args_list.append(f"algo={params['lyco_algo']}")
            if params["conv_dim"] > 0: network_args_list.append(f"conv_dim={params['conv_dim']}")
            if params["conv_alpha"] > 0: network_args_list.append(f"conv_alpha={params['conv_alpha']}")
            if params["lyco_dropout"] > 0: network_args_list.append(f"dropout={params['lyco_dropout']}")
            if params["lyco_preset"]: network_args_list.append(f"preset={params['lyco_preset']}")
        elif params["enable_lora_plus"]:
            network_args_list.append(f"loraplus_lr_ratio={params['loraplus_lr_ratio']}")
        if network_args_list: script_args.extend(["--network_args"] + network_args_list)
        if network_module: script_args.extend(["--network_module", network_module])
    if params["save_every_n_steps"] > 0: script_args.extend(["--save_every_n_steps", str(params["save_every_n_steps"])])
    if params["save_state"]: script_args.append("--save_state")
    if params["training_comment"]: script_args.extend(["--training_comment", params["training_comment"]])
    if params["enable_sample"]:
        if params["sample_prompts"]: script_args.extend(["--sample_prompts", params["sample_prompts"]])
        if params["sample_every_n_epochs"] > 0: script_args.extend(["--sample_every_n_epochs", str(params["sample_every_n_epochs"])])
    full_command = launch_args + [script_path] + script_args; full_log = ""
    for line in run_script(full_command): full_log += line + "\n"; yield full_log
def generate_logic(generate_mode, dit, vae, text_encoder1, text_encoder2, t5, image_encoder, prompt, lora_weight, lora_multiplier, video_size_w, video_size_h, video_length, fps, infer_steps, seed, save_path, embedded_cfg_scale, guidance_scale, image_path_fp):
    script_name = ""; args = ["--dit", dit, "--vae", vae]
    if generate_mode == "HunyuanVideo": script_name = "hv_generate_video.py"; args.extend(["--text_encoder1", text_encoder1, "--text_encoder2", text_encoder2]); args.extend(["--embedded_cfg_scale", str(embedded_cfg_scale)])
    elif generate_mode == "Wan": script_name = "wan_generate_video.py"; args.extend(["--t5", t5]); args.extend(["--guidance_scale", str(guidance_scale)])
    elif generate_mode == "FramePack":
        script_name = "fpack_generate_video.py"; args.extend(["--text_encoder1", text_encoder1, "--text_encoder2", text_encoder2, "--image_encoder", image_encoder]);
        if image_path_fp: args.extend(["--image_path", image_path_fp]);
        args.extend(["--embedded_cfg_scale", str(embedded_cfg_scale)])
    else: yield "Invalid generation mode.", None; return
    script_path = f"./musubi-tuner/{script_name}"; args.extend(["--prompt", prompt, "--video_size", str(video_size_w), str(video_size_h), "--video_length", str(video_length), "--fps", str(fps), "--infer_steps", str(infer_steps), "--seed", str(seed)])
    if lora_weight: args.extend(["--lora_weight", lora_weight, "--lora_multiplier", str(lora_multiplier)])
    if os.path.isdir(save_path): filename = f"{generate_mode.lower()}_seed{seed}_{lora_multiplier}w.mp4"; output_filepath = os.path.join(save_path, filename)
    else: output_filepath = save_path
    args.extend(["--save_path", output_filepath]); full_log = ""; command = [script_path] + args
    for line in run_script(command): full_log += line + "\n"; yield full_log, gr.update(visible=False)
    if os.path.exists(output_filepath): yield full_log, gr.update(value=output_filepath, visible=True)
    else: yield full_log + f"\n\nERROR: Expected output file not found at {output_filepath}", None
def convert_lora_logic(input_path, output_path, target):
    script_path = "./musubi-tuner/convert_lora.py"; args = ["--input", input_path, "--output", output_path, "--target", target]; full_log = ""
    for line in run_script([script_path] + args): full_log += line + "\n"; yield full_log
# CORRECTED lora_ema_logic
def lora_ema_logic(lora_files, output_file_path, method, beta_c, beta_l1, beta_l2, sigma_rel):
    script_path = "./musubi-tuner/lora_post_hoc_ema.py"
    if not lora_files: yield "Error: No LoRA files were selected."; return
    args = lora_files + ["--output_file", output_file_path, "--method", method]
    if method == "constant": args.extend(["--beta", str(beta_c)])
    elif method == "linear": args.extend(["--beta", str(beta_l1), "--beta2", str(beta_l2)])
    elif method == "power": args.extend(["--sigma_rel", str(sigma_rel)])
    full_log = ""
    for line in run_script([script_path] + args): full_log += line + "\n"; yield full_log

# --- Save/Load GUI State Logic ---
CONFIG_SAVE_PATH = "./gui_config.json"
def save_gui_state(*all_inputs):
    keys_list = [comp.label for comp in all_ui_components if comp.label]
    state_dict = dict(zip(keys_list, all_inputs))
    with open(CONFIG_SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(state_dict, f, indent=4)
    return f"Configuration saved to {CONFIG_SAVE_PATH}"
def load_gui_state():
    if not os.path.exists(CONFIG_SAVE_PATH):
        return [gr.update() for _ in all_ui_components] + ["Config file not found."]
    with open(CONFIG_SAVE_PATH, 'r', encoding='utf-8') as f:
        state_dict = json.load(f)
    update_list = []
    for comp in all_ui_components:
        key = comp.label if comp.label else None
        if key and key in state_dict:
            update_list.append(gr.update(value=state_dict[key]))
        else:
            update_list.append(gr.update())
    return update_list + [f"Configuration loaded from {CONFIG_SAVE_PATH}"]

# --- Gradio UI Definition ---
with gr.Blocks(theme=gr.themes.Soft(), title="Musubi Tuner GUI") as demo:
    gr.Markdown("# Musubi Tuner GUI\n*A unified interface for model downloads, dataset configuration, caching, training, and generation.*")
    gr.Markdown("""[**GitHub Repository**](https://github.com/rorsaeed/mj-musubi-tuner-gui) | [**Documentation & Support**](https://eng.webphotogallery.store/musubi-tuner-gui/)""")
    with gr.Row():
        load_gui_btn = gr.Button("Load UI State"); save_gui_btn = gr.Button("Save UI State"); status_text = gr.Textbox(label="Status", interactive=False, scale=2)

    with gr.Tab("1. Model Downloader"):
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Accordion("HunyuanVideo Models", open=True): hy_model_choice = gr.Radio(label="Model", choices=[("T2V", "1"), ("I2V", "2"), ("Skip", "n")], value="n"); download_hy_btn = gr.Button("Download HunyuanVideo Model")
                with gr.Accordion("Wan Models", open=True): wan_model_choice = gr.Radio(label="Model", choices=[("T2V-1.3B", "1"), ("T2V-14B", "2"), ("I2V-480p", "3"), ("I2V-720p", "4"), ("1.3B-FC", "5"), ("14B-FC", "6"), ("Skip", "n")], value="n"); download_wan_btn = gr.Button("Download Wan Model")
                with gr.Accordion("FramePack Models", open=True): fp_model_choice = gr.Radio(label="Model", choices=[("FramePack Base", "1"), ("FramePack F1", "2"), ("Skip", "n")], value="n"); download_fp_btn = gr.Button("Download FramePack Model")
            with gr.Column(scale=2): console_output_install = gr.Textbox(label="Downloader Console", lines=25, interactive=False)

    with gr.Tab("2. Dataset Config"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### General Settings");
                with gr.Accordion("[general]", open=True):
                    dc_gen_res_w = gr.Number(label="Resolution Width", value=960); dc_gen_res_h = gr.Number(label="Resolution Height", value=544); dc_gen_caption_ext = gr.Textbox(label="Caption Extension", value=".txt"); dc_gen_batch_size = gr.Number(label="Batch Size", value=1, precision=0); dc_gen_repeats = gr.Number(label="Num Repeats", value=1, precision=0)
                    with gr.Row(): dc_gen_bucket = gr.Checkbox(label="Enable Bucketing", value=True); dc_gen_no_upscale = gr.Checkbox(label="Bucket - No Upscaling", value=False)
                gr.Markdown("### Add a Dataset Entry")
                with gr.Accordion("[[datasets]]", open=True):
                    dc_ds_type = gr.Radio(label="Dataset Type", choices=["Image", "Video"], value="Image", interactive=True); dc_ds_source = gr.Radio(label="Caption Source", choices=["Text Files", "JSONL"], value="Text Files")
                    with gr.Row(): dc_ds_path = gr.Textbox(label="Dataset Path", interactive=True); dc_ds_path_btn = gr.Button("📂")
                    with gr.Row(): dc_ds_cache_path = gr.Textbox(label="Cache Directory", interactive=True); dc_ds_cache_path_btn = gr.Button("📂")
                    with gr.Row(): dc_ds_control_path = gr.Textbox(label="Control Directory", interactive=True); dc_ds_control_path_btn = gr.Button("📂")
                    with gr.Group(visible=False) as video_params_group:
                        gr.Markdown("#### Video Parameters"); dc_ds_target_frames = gr.Textbox(label="Target Frames (e.g., 1,25,45)", value="1, 25, 45"); dc_ds_frame_extraction = gr.Dropdown(label="Frame Extraction", choices=["head", "chunk", "slide", "uniform", "full"], value="head"); dc_ds_max_frames = gr.Number(label="Max Frames", value=129); dc_ds_source_fps = gr.Number(label="Source FPS", value=None)
                    add_dataset_btn = gr.Button("Add This Dataset to Config", variant="secondary")
            with gr.Column(scale=1):
                gr.Markdown("### TOML File Preview & Saving"); toml_preview = gr.Code(label="TOML Preview", lines=20)
                with gr.Row(): clear_all_btn = gr.Button("Clear All Datasets"); regenerate_preview_btn = gr.Button("Refresh Preview")
                with gr.Accordion("Save & Use File", open=True):
                    toml_save_path = gr.Textbox(label="Save Path", value="./configs/generated_dataset.toml"); save_status_text = gr.Textbox(label="Status", interactive=False); save_toml_btn = gr.Button("Generate & Save TOML File", variant="primary")
                    saved_toml_filepath = gr.Textbox(visible=False); copy_path_btn = gr.Button("Copy Saved Path to Other Tabs")

    with gr.Tab("3. Caching"):
        with gr.Row():
            with gr.Column(scale=1):
                cache_mode = gr.Dropdown(label="Cache Mode", choices=["HunyuanVideo", "Wan", "FramePack"], value="Wan", interactive=True); dataset_config_cache = gr.Textbox(label="Dataset Config (.toml)", value="./configs/generated_dataset.toml"); vae_cache = gr.Textbox(label="VAE Path", value="./ckpts/vae/Wan2.1_VAE.pth"); skip_existing_cache = gr.Checkbox(label="Skip Existing Cache Files", value=True)
                with gr.Group(visible=True) as wan_cache_params:
                    gr.Markdown("#### Wan Specific"); vae_cache_cpu_wan = gr.Checkbox(label="Cache VAE on CPU", value=True); clip_wan_cache = gr.Textbox(label="CLIP Path (for I2V)", value="./ckpts/text_encoder_2/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"); t5_wan_cache = gr.Textbox(label="T5 Model Path", value="./ckpts/text_encoder/models_t5_umt5-xxl-enc-bf16.pth"); fp8_t5_cache = gr.Checkbox(label="Use FP8 for T5", value=True)
                with gr.Group(visible=False) as hy_cache_params:
                    gr.Markdown("#### HunyuanVideo/FramePack Specific"); text_encoder1_hy_cache = gr.Textbox(label="Text Encoder 1", value="./ckpts/text_encoder/llava_llama3_fp16.safetensors"); text_encoder2_hy_cache = gr.Textbox(label="Text Encoder 2", value="./ckpts/text_encoder_2/clip_l.safetensors")
                with gr.Group(visible=False) as fp_cache_params:
                    gr.Markdown("#### FramePack Specific"); image_encoder_fp_cache = gr.Textbox(label="Image Encoder", value="./ckpts/framepack/sigclip_vision_patch14_384.safetensors")
                run_caching_btn = gr.Button("Run Caching", variant="primary")
            with gr.Column(scale=2): console_output_cache = gr.Textbox(label="Console Output", lines=20, interactive=False)

    with gr.Tab("4. Training"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Basic Configuration"); train_mode_train = gr.Dropdown(label="Training Mode", choices=["db", "HunyuanVideo_Lora", "Wan_Lora", "FramePack_Lora"], value="Wan_Lora", interactive=True); dataset_config_train = gr.Textbox(label="Dataset Config (.toml)", value="./configs/generated_dataset.toml")
                with gr.Accordion("Model Paths", open=True):
                    dit_train = gr.Textbox(label="DiT Path", value="./ckpts/wan/split_files/diffusion_models/wan2.1_t2v_14B_fp16.safetensors"); vae_train = gr.Textbox(label="VAE Path", value="./ckpts/vae/Wan2.1_VAE.pth")
                    with gr.Group(visible=False) as hy_train_paths: text_encoder1_hy_train = gr.Textbox(label="Text Encoder 1", value="./ckpts/text_encoder/llava_llama3_fp16.safetensors"); text_encoder2_hy_train = gr.Textbox(label="Text Encoder 2", value="./ckpts/text_encoder_2/clip_l.safetensors")
                    with gr.Group(visible=True) as wan_train_paths: t5_wan_train = gr.Textbox(label="T5 Model Path", value="./ckpts/text_encoder/models_t5_umt5-xxl-enc-bf16.pth"); clip_wan_train = gr.Textbox(label="CLIP Path (for I2V)", value="")
                    with gr.Group(visible=False) as fp_train_paths: image_encoder_fp_train = gr.Textbox(label="Image Encoder", value="./ckpts/framepack/sigclip_vision_patch14_384.safetensors")
                with gr.Accordion("Main Training Parameters", open=True):
                    max_train_epochs_train = gr.Number(label="Max Train Epochs", value=15, precision=0); gradient_accumulation_steps_train = gr.Number(label="Gradient Accumulation Steps", value=1, precision=0); gradient_checkpointing_train = gr.Checkbox(label="Use Gradient Checkpointing", value=True); seed_train = gr.Number(label="Seed", value=1026, precision=0)
                with gr.Accordion("Learning Rate", open=True):
                    lr_train = gr.Textbox(label="Learning Rate", value="2e-4"); lr_scheduler_train = gr.Dropdown(label="LR Scheduler", choices=["constant", "linear", "cosine", "cosine_with_restarts", "polynomial", "constant_with_warmup", "cosine_with_min_lr", "warmup_stable_decay"], value="cosine_with_min_lr")
                with gr.Accordion("Network Settings (LoRA)", open=True) as lora_train_settings:
                    network_dim_train = gr.Slider(label="Network Dimension (Rank)", minimum=4, maximum=256, value=32, step=4); network_alpha_train = gr.Slider(label="Network Alpha", minimum=1, maximum=256, value=16, step=1); network_dropout_train = gr.Slider(label="Network Dropout", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                with gr.Accordion("Output & Saving", open=True):
                    output_name_train = gr.Textbox(label="Output Model Name", value="wan_lora_test"); save_every_n_epochs_train = gr.Number(label="Save every N epochs", value=4, precision=0)
                with gr.Accordion("Advanced Settings", open=False):
                    with gr.Tabs():
                        with gr.TabItem("Training & Timestep"):
                            guidance_scale_train = gr.Number(label="Guidance Scale", value=1.0); timestep_sampling_train = gr.Dropdown(label="Timestep Sampling", choices=["sigma", "uniform", "sigmoid", "shift"], value="shift"); weighting_scheme_train = gr.Dropdown(label="Timestep Weighting", choices=["none", "sigma_sqrt", "logit_normal", "mode", "cosmap", "uniform"], value="none", interactive=True)
                            with gr.Group(visible=False) as logit_normal_group: logit_mean_train = gr.Number(label="Logit Mean", value=0.0); logit_std_train = gr.Number(label="Logit Std Dev", value=1.0)
                            with gr.Group(visible=False) as mode_group: mode_scale_train = gr.Number(label="Mode Scale", value=1.29)
                            discrete_flow_shift_train = gr.Number(label="Discrete Flow Shift", value=3.0); sigmoid_scale_train = gr.Number(label="Sigmoid Scale", value=1.0)
                        with gr.TabItem("LoRA / Lycoris"):
                             enable_lora_plus = gr.Checkbox(label="Enable LoRA+", value=True, interactive=True); loraplus_lr_ratio = gr.Slider(label="LoRA+ LR Ratio", minimum=1, maximum=32, value=4, step=1, visible=True); gr.Markdown("---"); enable_lycoris = gr.Checkbox(label="Enable Lycoris", value=False, interactive=True)
                             with gr.Group(visible=False) as lycoris_group:
                                lyco_algo = gr.Dropdown(label="Lycoris Algorithm", choices=["lora", "loha", "ia3", "lokr", "dylora", "full", "diag-oft"], value="lokr"); lyco_preset = gr.Dropdown(label="Preset Modules", choices=["full", "full-lin", "attn-mlp", "attn-only", "unet-transformer-only", "unet-convblock-only"], value="attn-mlp"); conv_dim = gr.Slider(label="Conv Dimension", minimum=0, maximum=64, value=4, step=1); conv_alpha = gr.Slider(label="Conv Alpha", minimum=0.0, maximum=64.0, value=1.0, step=0.5); lyco_dropout = gr.Slider(label="Lycoris Dropout", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        with gr.TabItem("LR Scheduler"):
                            lr_warmup_steps_train = gr.Number(label="Warmup Steps", value=0, precision=0); lr_decay_steps_train = gr.Number(label="Decay Steps", value=0.2, step=0.01); lr_scheduler_num_cycles_train = gr.Number(label="Num Cycles", value=1, precision=0); lr_scheduler_power_train = gr.Number(label="Power", value=1.0); lr_scheduler_min_lr_ratio_train = gr.Number(label="Min LR Ratio", value=0.1)
                        with gr.TabItem("Precision & Optimizer"):
                            mixed_precision_train = gr.Dropdown(label="Mixed Precision", choices=["no", "fp16", "bf16"], value="fp16"); attn_mode_train = gr.Dropdown(label="Attention Mechanism", choices=["torch", "flash", "xformers", "sdpa"], value="flash"); optimizer_type_train = gr.Dropdown(label="Optimizer", choices=["AdamW8bit", "prodigy", "DAdaptAdam", "Lion", "adafactor", "adopt"], value="adopt"); max_grad_norm_train = gr.Number(label="Max Grad Norm", value=1.0)
                        with gr.TabItem("Saving & Metadata"):
                            save_every_n_steps_train = gr.Number(label="Save every N steps", value=0, precision=0); save_state_train = gr.Checkbox(label="Save Training State", value=False); training_comment_train = gr.Textbox(label="Metadata Comment", lines=2)
                        with gr.TabItem("Sampling"):
                            enable_sample_train = gr.Checkbox(label="Enable Sampling", value=False); sample_prompts_train = gr.Textbox(label="Path to Prompt File (.txt)", value="./toml/sample_prompts.txt"); sample_every_n_epochs_train = gr.Number(label="Sample every N epochs", value=1, precision=0)
                run_training_btn = gr.Button("Start Training", variant="primary")
            with gr.Column(scale=2): console_output_train = gr.Textbox(label="Console Output", lines=40, interactive=False)

    with gr.Tab("5. Generate Video"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Main Configuration"); generate_mode = gr.Dropdown(label="Generate Mode", choices=["HunyuanVideo", "Wan", "FramePack"], value="Wan", interactive=True)
                gr.Markdown("### Model Paths"); dit_gen = gr.Textbox(label="DiT Path", value="./ckpts/wan/split_files/diffusion_models/wan2.1_t2v_14B_fp16.safetensors"); vae_gen = gr.Textbox(label="VAE Path", value="./ckpts/vae/Wan2.1_VAE.pth")
                with gr.Group(visible=False) as hy_gen_params:
                    gr.Markdown("#### HunyuanVideo/FramePack Params"); text_encoder1_hy_gen = gr.Textbox(label="Text Encoder 1", value="./ckpts/text_encoder/llava_llama3_fp16.safetensors"); text_encoder2_hy_gen = gr.Textbox(label="Text Encoder 2", value="./ckpts/text_encoder_2/clip_l.safetensors"); embedded_cfg_scale_hy = gr.Slider(label="Embedded CFG Scale", minimum=1.0, maximum=20.0, value=7.0, step=0.5)
                with gr.Group(visible=True) as wan_gen_params:
                    gr.Markdown("#### Wan Parameters"); t5_wan_gen = gr.Textbox(label="T5 Model Path", value="./ckpts/text_encoder/models_t5_umt5-xxl-enc-bf16.pth"); guidance_scale_wan = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=15.0, value=5.0, step=0.5)
                with gr.Group(visible=False) as fp_gen_params:
                    gr.Markdown("#### FramePack Parameters"); image_encoder_fp_gen = gr.Textbox(label="Image Encoder", value="./ckpts/framepack/sigclip_vision_patch14_384.safetensors"); image_path_fp_gen = gr.File(label="Input Image (for I2V)", type="filepath")
                with gr.Group():
                    gr.Markdown("### LoRA (Optional)"); lora_weight_gen = gr.Textbox(label="LoRA Weight Path", value="./output_dir/wan_lora_test-000004.safetensors"); lora_multiplier_gen = gr.Slider(label="LoRA Multiplier", minimum=-1.0, maximum=2.0, value=1.0, step=0.05)
            with gr.Column(scale=2):
                gr.Markdown("### Generation Settings"); prompt_gen = gr.Textbox(label="Prompt", lines=5, value="1girl, solo, long hair, looking at viewer, open mouth, blue eyes, simple background")
                with gr.Row(): video_size_w_gen = gr.Number(label="Video Width", value=832); video_size_h_gen = gr.Number(label="Video Height", value=480)
                with gr.Row(): video_length_gen = gr.Slider(label="Video Length (frames)", minimum=16, maximum=256, value=81, step=1); fps_gen = gr.Slider(label="FPS", minimum=8, maximum=60, value=16, step=1)
                with gr.Row(): infer_steps_gen = gr.Slider(label="Inference Steps", minimum=10, maximum=100, value=20, step=1); seed_gen = gr.Number(label="Seed", value=1026, precision=0)
                save_path_gen = gr.Textbox(label="Output Path (File or Directory)", value="./output_dir/"); run_generate_btn = gr.Button("Generate Video", variant="primary")
                with gr.Accordion("Console Output", open=False): console_output_gen = gr.Textbox(label="Log", lines=10, interactive=False)
                video_output_gen = gr.Video(label="Generated Video", visible=False, interactive=False)

    with gr.Tab("6. Utilities"):
        with gr.Tabs():
            with gr.TabItem("Convert LoRA for ComfyUI"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("Converts a trained LoRA to a format compatible with other UIs like ComfyUI."); convert_lora_input = gr.Textbox(label="Input LoRA Path", value="./output_dir/wan_lora_test-000004.safetensors"); convert_lora_output = gr.Textbox(label="Output LoRA Path", value="./output_dir/wan_lora_test-000004_comfy.safetensors"); convert_lora_target = gr.Dropdown(label="Target Format", choices=["other", "default"], value="other"); convert_lora_btn = gr.Button("Convert LoRA", variant="primary")
                    with gr.Column(): console_output_convert = gr.Textbox(label="Console Output", lines=10, interactive=False)
            with gr.TabItem("LoRA Post Hoc EMA"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("Merge multiple LoRA checkpoints using EMA to potentially improve quality."); ema_lora_files = gr.File(label="Select multiple LoRA checkpoint files", file_count="multiple", type="filepath"); ema_output_path = gr.Textbox(label="Output Merged LoRA Path", value="./output_dir/lora_ema_merged.safetensors"); ema_method = gr.Dropdown(label="EMA Method", choices=["constant", "linear", "power"], value="power", interactive=True)
                        with gr.Group(visible=False) as ema_constant_params: ema_beta_c = gr.Slider(label="Beta", minimum=0.8, maximum=0.99, value=0.9, step=0.01)
                        with gr.Group(visible=False) as ema_linear_params: ema_beta_l1 = gr.Slider(label="Start Beta", minimum=0.8, maximum=0.99, value=0.9, step=0.01); ema_beta_l2 = gr.Slider(label="End Beta", minimum=0.8, maximum=0.99, value=0.95, step=0.01)
                        with gr.Group(visible=True) as ema_power_params: ema_sigma_rel = gr.Slider(label="Sigma Rel", minimum=0.05, maximum=0.5, value=0.2, step=0.01)
                        run_ema_btn = gr.Button("Run EMA Merge", variant="primary")
                    with gr.Column(): console_output_ema = gr.Textbox(label="Console Output", lines=10, interactive=False)

    # --- List of ALL UI components for Save/Load ---
    all_ui_components = [
        hy_model_choice, wan_model_choice, fp_model_choice,
        dc_gen_res_w, dc_gen_res_h, dc_gen_caption_ext, dc_gen_batch_size, dc_gen_repeats, dc_gen_bucket, dc_gen_no_upscale,
        dc_ds_type, dc_ds_source, dc_ds_path, dc_ds_cache_path, dc_ds_control_path, dc_ds_target_frames, dc_ds_frame_extraction, dc_ds_max_frames, dc_ds_source_fps,
        toml_save_path,
        cache_mode, dataset_config_cache, vae_cache, skip_existing_cache, vae_cache_cpu_wan, clip_wan_cache, t5_wan_cache, fp8_t5_cache, text_encoder1_hy_cache, text_encoder2_hy_cache, image_encoder_fp_cache,
        train_mode_train, dataset_config_train, dit_train, vae_train, text_encoder1_hy_train, text_encoder2_hy_train, t5_wan_train, clip_wan_train, image_encoder_fp_train,
        max_train_epochs_train, gradient_accumulation_steps_train, gradient_checkpointing_train, seed_train,
        lr_train, lr_scheduler_train,
        network_dim_train, network_alpha_train, network_dropout_train,
        output_name_train, save_every_n_epochs_train,
        guidance_scale_train, timestep_sampling_train, weighting_scheme_train, logit_mean_train, logit_std_train, mode_scale_train, discrete_flow_shift_train, sigmoid_scale_train,
        enable_lora_plus, loraplus_lr_ratio, enable_lycoris, lyco_algo, lyco_preset, conv_dim, conv_alpha, lyco_dropout,
        lr_warmup_steps_train, lr_decay_steps_train, lr_scheduler_num_cycles_train, lr_scheduler_power_train, lr_scheduler_min_lr_ratio_train,
        mixed_precision_train, attn_mode_train, optimizer_type_train, max_grad_norm_train,
        save_every_n_steps_train, save_state_train, training_comment_train,
        enable_sample_train, sample_prompts_train, sample_every_n_epochs_train,
        generate_mode, dit_gen, vae_gen, text_encoder1_hy_gen, text_encoder2_hy_gen, t5_wan_gen, embedded_cfg_scale_hy, guidance_scale_wan, image_encoder_fp_gen, image_path_fp_gen,
        lora_weight_gen, lora_multiplier_gen,
        prompt_gen, video_size_w_gen, video_size_h_gen, video_length_gen, fps_gen, infer_steps_gen, seed_gen, save_path_gen,
        convert_lora_input, convert_lora_output, convert_lora_target,
        ema_lora_files, ema_output_path, ema_method, ema_beta_c, ema_beta_l1, ema_beta_l2, ema_sigma_rel
    ]

    # --- Event Handlers ---
    save_gui_btn.click(fn=save_gui_state, inputs=all_ui_components, outputs=[status_text])
    load_gui_btn.click(fn=load_gui_state, inputs=None, outputs=all_ui_components + [status_text])
    download_hy_btn.click(fn=functools.partial(download_model, model_type="hy"), inputs=[hy_model_choice], outputs=[console_output_install])
    download_wan_btn.click(fn=functools.partial(download_model, model_type="wan"), inputs=[wan_model_choice], outputs=[console_output_install])
    download_fp_btn.click(fn=functools.partial(download_model, model_type="fp"), inputs=[fp_model_choice], outputs=[console_output_install])
    dc_ds_path_btn.click(lambda x: select_folder(x), inputs=[dc_ds_path], outputs=[dc_ds_path]); dc_ds_cache_path_btn.click(lambda x: select_folder(x), inputs=[dc_ds_cache_path], outputs=[dc_ds_cache_path]); dc_ds_control_path_btn.click(lambda x: select_folder(x), inputs=[dc_ds_control_path], outputs=[dc_ds_control_path])
    dc_ds_type.change(lambda x: gr.update(visible=x == 'Video'), inputs=dc_ds_type, outputs=video_params_group)
    dataset_builder_inputs = [dc_gen_res_w, dc_gen_res_h, dc_gen_caption_ext, dc_gen_batch_size, dc_gen_repeats, dc_gen_bucket, dc_gen_no_upscale]
    add_dataset_btn.click(fn=add_dataset_entry, inputs=dataset_builder_inputs + [dc_ds_type, dc_ds_source, dc_ds_path, dc_ds_cache_path, dc_ds_control_path, dc_ds_target_frames, dc_ds_frame_extraction, dc_ds_max_frames, dc_ds_source_fps], outputs=toml_preview)
    regenerate_preview_btn.click(fn=generate_toml_preview, inputs=dataset_builder_inputs, outputs=toml_preview)
    clear_all_btn.click(fn=clear_all_datasets, outputs=toml_preview)
    save_toml_btn.click(fn=save_toml_file, inputs=[toml_preview, toml_save_path], outputs=[save_status_text, saved_toml_filepath])
    copy_path_btn.click(fn=copy_path_to_tabs, inputs=[saved_toml_filepath], outputs=[dataset_config_cache, dataset_config_train])
    cache_mode.change(fn=lambda m: {wan_cache_params: gr.update(visible=m=="Wan"), hy_cache_params: gr.update(visible=m in ["HunyuanVideo","FramePack"]), fp_cache_params: gr.update(visible=m=="FramePack")}, inputs=cache_mode, outputs=[wan_cache_params, hy_cache_params, fp_cache_params])
    train_mode_train.change(fn=lambda m: {lora_train_settings: gr.update(visible="Lora" in m), hy_train_paths: gr.update(visible=any(s in m for s in ["HunyuanVideo","FramePack","db"])), wan_train_paths: gr.update(visible="Wan" in m), fp_train_paths: gr.update(visible="FramePack" in m)}, inputs=train_mode_train, outputs=[lora_train_settings, hy_train_paths, wan_train_paths, fp_train_paths])
    enable_lycoris.change(fn=lambda x: gr.update(visible=x), inputs=enable_lycoris, outputs=lycoris_group)
    enable_lora_plus.change(fn=lambda x: gr.update(visible=x), inputs=enable_lora_plus, outputs=loraplus_lr_ratio)
    weighting_scheme_train.change(fn=lambda x: {logit_normal_group: gr.update(visible=x=='logit_normal'), mode_group: gr.update(visible=x=='mode')}, inputs=weighting_scheme_train, outputs=[logit_normal_group, mode_group])
    generate_mode.change(fn=lambda m: {hy_gen_params: gr.update(visible=m in ["HunyuanVideo","FramePack"]), wan_gen_params: gr.update(visible=m=="Wan"), fp_gen_params: gr.update(visible=m=="FramePack")}, inputs=generate_mode, outputs=[hy_gen_params, wan_gen_params, fp_gen_params])
    ema_method.change(fn=lambda m: {ema_constant_params: gr.update(visible=m=="constant"), ema_linear_params: gr.update(visible=m=="linear"), ema_power_params: gr.update(visible=m=="power")}, inputs=ema_method, outputs=[ema_constant_params, ema_linear_params, ema_power_params])
    run_caching_btn.click(fn=caching_logic, inputs=[cache_mode, dataset_config_cache, vae_cache, skip_existing_cache, vae_cache_cpu_wan, clip_wan_cache, t5_wan_cache, fp8_t5_cache, text_encoder1_hy_cache, text_encoder2_hy_cache, image_encoder_fp_cache], outputs=[console_output_cache])
    
    # Corrected and explicit training inputs list
    training_inputs_list = [
        train_mode_train, dataset_config_train, dit_train, vae_train, text_encoder1_hy_train, text_encoder2_hy_train,
        t5_wan_train, clip_wan_train, image_encoder_fp_train, max_train_epochs_train, gradient_checkpointing_train,
        gradient_accumulation_steps_train, seed_train, lr_train, lr_scheduler_train, lr_warmup_steps_train,
        lr_decay_steps_train, lr_scheduler_num_cycles_train, lr_scheduler_power_train, lr_scheduler_min_lr_ratio_train,
        network_dim_train, network_alpha_train, network_dropout_train, mixed_precision_train, attn_mode_train,
        guidance_scale_train, timestep_sampling_train, discrete_flow_shift_train, sigmoid_scale_train,
        weighting_scheme_train, logit_mean_train, logit_std_train, mode_scale_train, enable_lora_plus,
        loraplus_lr_ratio, enable_lycoris, conv_dim, conv_alpha, lyco_algo, lyco_dropout, lyco_preset,
        output_name_train, save_every_n_epochs_train, save_every_n_steps_train, save_state_train,
        optimizer_type_train, max_grad_norm_train, enable_sample_train, sample_prompts_train,
        sample_every_n_epochs_train, training_comment_train
    ]
    run_training_btn.click(fn=training_logic, inputs=training_inputs_list, outputs=[console_output_train])
    
    run_generate_btn.click(fn=generate_logic, inputs=[generate_mode, dit_gen, vae_gen, text_encoder1_hy_gen, text_encoder2_hy_gen, t5_wan_gen, image_encoder_fp_gen, prompt_gen, lora_weight_gen, lora_multiplier_gen, video_size_w_gen, video_size_h_gen, video_length_gen, fps_gen, infer_steps_gen, seed_gen, save_path_gen, embedded_cfg_scale_hy, guidance_scale_wan, image_path_fp_gen], outputs=[console_output_gen, video_output_gen])
    convert_lora_btn.click(fn=convert_lora_logic, inputs=[convert_lora_input, convert_lora_output, convert_lora_target], outputs=[console_output_convert])
    run_ema_btn.click(fn=lora_ema_logic, inputs=[ema_lora_files, ema_output_path, ema_method, ema_beta_c, ema_beta_l1, ema_beta_l2, ema_sigma_rel], outputs=[console_output_ema])

if __name__ == "__main__":
    try: import toml
    except ImportError:
        print("toml library not found. Installing..."); subprocess.check_call([sys.executable, "-m", "pip", "install", "toml"])
        import toml
        
    demo.launch(inbrowser=True)