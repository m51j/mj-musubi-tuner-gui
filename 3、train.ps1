# Cache script by @bdsqlsz

# model_path
$dataset_config = "./toml/qinglong-datasets.toml"                                   # path to dataset config .toml file | 数据集配置文件路径
$dit = "./ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt" # DiT directory | DiT路径
$vae = "./ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt"                        # VAE directory | VAE路径
$text_encoder1 = "./ckpts/text_encoder/llava_llama3_fp16.safetensors"               # Text Encoder 1 directory | 文本编码器路径
$text_encoder2 = "./ckpts/text_encoder_2/clip_l.safetensors"                        # Text Encoder 2 directory | 文本编码器路径

$resume = ""                                                                        # resume from state | 从某个状态文件夹中恢复训练
$network_weights = ""                                                               # pretrained weights for LoRA network | 若需要从已有的 LoRA 模型上继续训练，请填写 LoRA 模型路径。

#COPY machine | 差异炼丹法
$base_weights = "" #指定合并到底模basemodel中的模型路径，多个用空格隔开。默认为空，不使用。
$base_weights_multiplier = "1.0" #指定合并模型的权重，多个用空格隔开，默认为1.0。

#train config | 训练配置
$max_train_steps = ""                                                                # max train steps | 最大训练步数
$max_train_epochs = 80                                                               # max train epochs | 最大训练轮数
$gradient_checkpointing = 1                                                          # 梯度检查，开启后可节约显存，但是速度变慢
$gradient_accumulation_steps = 4                                                     # 梯度累加数量，变相放大batchsize的倍数
$guidance_scale = 1.0
$seed = 1026 # reproducable seed | 设置跑测试用的种子，输入一个prompt和这个种子大概率得到训练图。可以用来试触发关键词

#timestep sampling
$timestep_sampling = "sigmoid" # 时间步采样方法，可选 sd3用"sigma"、普通DDPM用"uniform" 或 flux用"sigmoid" 或者 "shift". shift需要修改discarete_flow_shift的参数
$discrete_flow_shift = 1.0 # Euler 离散调度器的离散流位移，sd3默认为3.0
$sigmoid_scale = 1.0 # sigmoid 采样的缩放因子，默认为 1.0。较大的值会使采样更加均匀

$weighting_scheme = ""      # sigma_sqrt, logit_normal, mode, cosmap, uniform, none
$logit_mean = 0.0           # logit mean | logit 均值 默认0.0 只在logit_normal下生效
$logit_std = 1.0            # logit std | logit 标准差 默认1.0 只在logit_normal下生效
$mode_scale = 1.29          # mode scale | mode 缩放 默认1.29 只在mode下生效
$min_timestep = 0           #最小时序，默认值0
$max_timestep = 1000        #最大时间步 默认1000
$show_timesteps = ""        #是否显示timesteps

# Learning rate | 学习率
$lr = "1e-3"
# $unet_lr = "5e-4"
# $text_encoder_lr = "2e-5"
$lr_scheduler = "cosine_with_min_lr"
# "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup" | PyTorch自带6种动态学习率函数
# constant，常量不变, constant_with_warmup 线性增加后保持常量不变, linear 线性增加线性减少, polynomial 线性增加后平滑衰减, cosine 余弦波曲线, cosine_with_restarts 余弦波硬重启，瞬间最大值。
# 新增cosine_with_min_lr(适合训练lora)、warmup_stable_decay(适合训练db)、inverse_sqrt
$lr_warmup_steps = 0 # warmup steps | 学习率预热步数，lr_scheduler 为 constant 或 adafactor 时该值需要设为0。仅在 lr_scheduler 为 constant_with_warmup 时需要填写这个值
$lr_decay_steps = 0.25 # decay steps | 学习率衰减步数，仅在 lr_scheduler 为warmup_stable_decay时 需要填写，一般是10%总步数
$lr_scheduler_num_cycles = 1 # restarts nums | 余弦退火重启次数，仅在 lr_scheduler 为 cosine_with_restarts 时需要填写这个值
$lr_scheduler_power = 1     #Polynomial power for polynomial scheduler |余弦退火power
$lr_scheduler_timescale = 0 #times scale |时间缩放，仅在 lr_scheduler 为 inverse_sqrt 时需要填写这个值，默认同lr_warmup_steps
$lr_scheduler_min_lr_ratio = 0.1 #min lr ratio |最小学习率比率，仅在 lr_scheduler 为 cosine_with_min_lr、、warmup_stable_decay 时需要填写这个值，默认0

#network settings
$network_dim = 32 # network dim | 常用 4~128，不是越大越好
$network_alpha = 16 # network alpha | 常用与 network_dim 相同的值或者采用较小的值，如 network_dim的一半 防止下溢。默认值为 1，使用较小的 alpha 需要提升学习率。
$network_dropout = 0 # network dropout | 常用 0~0.3
$dim_from_weights = $True # use dim from weights | 从已有的 LoRA 模型上继续训练时，自动获取 dim
$scale_weight_norms = 0 # scale weight norms (1 is a good starting point)| scale weight norms (1 is a good starting point)

# $train_unet_only = 1 # train U-Net only | 仅训练 U-Net，开启这个会牺牲效果大幅减少显存使用。6G显存可以开启
# $train_text_encoder_only = 0 # train Text Encoder only | 仅训练 文本编码器

#precision and accelerate/save memory
$attn_mode = "sdpa"                                                                # "flash", "sageattn", "xformers", "sdpa"
$mixed_precision = "bf16"                                                           # fp16 |bf16 default: bf16
$dit_dtype = ""                                                                     # fp16 | fp32 |bf16 default: bf16

$vae_dtype = ""                                                                     # fp16 | fp32 |bf16 default: fp16
$vae_tiling = $True                                                                 # enable spatial tiling for VAE, default is False. If vae_spatial_tile_sample_min_size is set, this is automatically enabled
$vae_chunk_size = 32                                                                # chunk size for CausalConv3d in VAE
$vae_spatial_tile_sample_min_size = 256                                             # spatial tile sample min size for VAE, default 256

$text_encoder_dtype = ""                                                            # fp16 | fp32 |bf16 default: fp16

$fp8_base = $True                                                                   # fp8
$fp8_llm = $False                                                                   # fp8 for LLM
$max_data_loader_n_workers = 8                                                      # max data loader n workers | 最大数据加载线程数
$persistent_data_loader_workers = $True                                             # save every n epochs | 每多少轮保存一次

$blocks_to_swap = 0                                                                # 交换的块数
$img_in_txt_in_offloading = $False                                                   # img in txt in offloading

#optimizer
$optimizer_type = "AdamW8bit"                                                       
# adamw8bit | adamw32bit | adamw16bit | adafactor | Lion | Lion8bit | 
# PagedLion8bit | AdamW | AdamW8bit | PagedAdamW8bit | AdEMAMix8bit | PagedAdEMAMix8bit
# DAdaptAdam | DAdaptLion | DAdaptAdan | DAdaptSGD | Sophia | Prodigy
$max_grad_norm = 1.0 # max grad norm | 最大梯度范数，默认为1.0

# wandb log
$wandb_api_key = ""                   # wandbAPI KEY，用于登录

# save and load settings | 保存和输出设置
$output_name = "hyvideo-qinglong"  # output model name | 模型保存名称
$save_every_n_epochs = "10"           # save every n epochs | 每多少轮保存一次
$save_every_n_steps = ""              # save every n steps | 每多少步保存一次
$save_last_n_epochs = ""            # save last n epochs | 保存最后多少轮
$save_last_n_steps = ""               # save last n steps | 保存最后多少步

# save state | 保存训练状态
$save_state = $False                  # save training state | 保存训练状态
$save_state_on_train_end = $False     # save state on train end |只在训练结束最后保存训练状态
$save_last_n_epochs_state = ""        # save last n epochs state | 保存最后多少轮训练状态
$save_last_n_steps_state = ""         # save last n steps state | 保存最后多少步训练状态

#lycoris组件
$enable_lycoris = 0 # 开启lycoris
$conv_dim = 0 #卷积 dim，推荐＜32
$conv_alpha = 0 #卷积 alpha，推荐1或者0.3
$algo = "lokr" # algo参数，指定训练lycoris模型种类，
#包括lora(就是locon)、
#loha
#IA3
#lokr
#dylora
#full(DreamBooth先训练然后导出lora)
#diag-oft
#它通过训练适用于各层输出的正交变换来保留超球面能量。
#根据原始论文，它的收敛速度比 LoRA 更快，但仍需进行实验。
#dim 与区块大小相对应：我们在这里固定了区块大小而不是区块数量，以使其与 LoRA 更具可比性。

$dropout = 0 #lycoris专用dropout
$preset = "attn-mlp" #预设训练模块配置
#full: default preset, train all the layers in the UNet and CLIP|默认设置，训练所有Unet和Clip层
#full-lin: full but skip convolutional layers|跳过卷积层
#attn-mlp: train all the transformer block.|kohya配置，训练所有transformer模块
#attn-only：only attention layer will be trained, lot of papers only do training on attn layer.|只有注意力层会被训练，很多论文只对注意力层进行训练。
#unet-transformer-only： as same as kohya_ss/sd_scripts with disabled TE, or, attn-mlp preset with train_unet_only enabled.|和attn-mlp类似，但是关闭te训练
#unet-convblock-only： only ResBlock, UpSample, DownSample will be trained.|只训练卷积模块，包括res、上下采样模块
#./toml/example_lycoris.toml: 也可以直接使用外置配置文件，制定各个层和模块使用不同算法训练，需要输入位置文件路径，参考样例已添加。

$factor = 8 #只适用于lokr的因子，-1~8，8为全维度
$decompose_both = 0 #适用于lokr的参数，对 LoKr 分解产生的两个矩阵执行 LoRA 分解（默认情况下只分解较大的矩阵）
$block_size = 4 #适用于dylora,分割块数单位，最小1也最慢。一般4、8、12、16这几个选
$use_tucker = 0 #适用于除 (IA)^3 和full
$use_scalar = 0 #根据不同算法，自动调整初始权重
$train_norm = 0 #归一化层
$dora_wd = 1 #Dora方法分解，低rank使用。适用于LoRA, LoHa, 和LoKr
$full_matrix = 0  #全矩阵分解
$bypass_mode = 0 #通道模式，专为 bnb 8 位/4 位线性层设计。(QLyCORIS)适用于LoRA, LoHa, 和LoKr
$rescaled = 1 #适用于设置缩放，效果等同于OFT
$constrain = 0 #设置值为FLOAT，效果等同于COFT

#sample | 输出采样图片
$enable_sample = 0 #1开启出图，0禁用
$sample_at_first = 1 #是否在训练开始时就出图
$sample_every_n_epochs = 2 #每n个epoch出一次图
$sample_prompts = "./toml/qinglong.txt" #prompt文件路径

#metadata
$training_comment = "this LoRA model created by bdsqlsz'script" # training_comment | 训练介绍，可以写作者名或者使用触发关键词
$metadata_title = "" # metadata title | 元数据标题
$metadata_author = "" # metadata author | 元数据作者
$metadata_description = "" # metadata contact | 元数据联系方式
$metadata_license = "" # metadata license | 元数据许可证
$metadata_tags = "" # metadata tags | 元数据标签

#huggingface settings
$async_upload = $False # push to hub | 推送到huggingface
$huggingface_repo_id = "" # huggingface repo id | huggingface仓库id
$huggingface_repo_type = "dataset" # huggingface repo type | huggingface仓库类型
$huggingface_path_in_repo = "" # huggingface path in repo | huggingface仓库路径
$huggingface_token = "" # huggingface token | huggingface仓库token
$huggingface_repo_visibility = "" # huggingface repo visibility | huggingface仓库可见性
$save_state_to_huggingface = $False # save state to huggingface | 保存训练状态到huggingface
$resume_from_huggingface = $False # resume from huggingface | 从huggingface恢复训练

#DDP | 多卡设置
$multi_gpu = 0                         #multi gpu | 多显卡训练开关，0关1开， 该参数仅限在显卡数 >= 2 使用
$highvram = 0                            #高显存模式，开启后会尽量使用显存
# $deepspeed = 0                         #deepspeed | 使用deepspeed训练，0关1开， 该参数仅限在显卡数 >= 2 使用
# $zero_stage = 2                        #zero stage | zero stage 0,1,2,3,阶段2用于训练 该参数仅限在显卡数 >= 2 使用
# $offload_optimizer_device = ""      #offload optimizer device | 优化器放置设备，cpu或者nvme, 该参数仅限在显卡数 >= 2 使用
# $fp16_master_weights_and_gradients = 0 #fp16 master weights and gradients | fp16主权重和梯度，0关1开， 该参数仅限在显卡数 >= 2 使用

$ddp_timeout = 120 #ddp timeout | ddp超时时间，单位秒， 该参数仅限在显卡数 >= 2 使用
$ddp_gradient_as_bucket_view = 1 #ddp gradient as bucket view | ddp梯度作为桶视图，0关1开， 该参数仅限在显卡数 >= 2 使用
$ddp_static_graph = 1 #ddp static graph | ddp静态图，0关1开， 该参数仅限在显卡数 >= 2 使用

# ============= DO NOT MODIFY CONTENTS BELOW | 请勿修改下方内容 =====================
# Activate python venv
Set-Location $PSScriptRoot
if ($env:OS -ilike "*windows*") {
  if (Test-Path "./venv/Scripts/activate") {
    Write-Output "Windows venv"
    ./venv/Scripts/activate
  }
  elseif (Test-Path "./.venv/Scripts/activate") {
    Write-Output "Windows .venv"
    ./.venv/Scripts/activate
  }
}
elseif (Test-Path "./venv/bin/activate") {
  Write-Output "Linux venv"
  ./venv/bin/Activate.ps1
}
elseif (Test-Path "./.venv/bin/activate") {
  Write-Output "Linux .venv"
  ./.venv/bin/activate.ps1
}

$Env:HF_HOME = "huggingface"
#$Env:HF_ENDPOINT = "https://hf-mirror.com"
$Env:XFORMERS_FORCE_DISABLE_TRITON = "1"
$ext_args = [System.Collections.ArrayList]::new()
$launch_args = [System.Collections.ArrayList]::new()
$laungh_script = "hv_train_network"
$network_module = "networks.lora"

if ($attn_mode -ieq "sageattn") {
  [void]$ext_args.Add("--sage_attn")
}
elseif ($attn_mode -ieq "flash") {
  [void]$ext_args.Add("--flash_attn")
}
elseif ($attn_mode -ieq "xformers") {
  [void]$ext_args.Add("--xformers")
}
else {
  [void]$ext_args.Add("--sdpa")
}

if ($multi_gpu -eq 1) {
  $launch_args += "--multi_gpu"
  $launch_args += "--rdzv_backend=c10d"
  # if ($deepspeed -eq 1) {
  #   [void]$ext_args.Add("--deepspeed")
  #   if ($zero_stage -ne 0) {
  #     [void]$ext_args.Add("--zero_stage=$zero_stage")
  #   }
  #   if ($offload_optimizer_device) {
  #     [void]$ext_args.Add("--offload_optimizer_device=$offload_optimizer_device")
  #   }
  #   if ($fp16_master_weights_and_gradients -eq 1) {
  #     [void]$ext_args.Add("--fp16_master_weights_and_gradients")
  #   }
  # }
  if ($ddp_timeout -ne 0) {
    [void]$ext_args.Add("--ddp_timeout=$ddp_timeout")
  }
  if ($ddp_gradient_as_bucket_view -ne 0) {
    [void]$ext_args.Add("--ddp_gradient_as_bucket_view")
  }
  if ($ddp_static_graph -ne 0) {
    [void]$ext_args.Add("--ddp_static_graph")
  }
  if ($highvram -ne 0) {
    [void]$ext_args.Add("--highvram")
  }
}

if ($timestep_sampling -ine "sigma") {
  [void]$ext_args.Add("--timestep_sampling=$timestep_sampling")
  if ($timestep_sampling -ieq "sigmoid" -or $timestep_sampling -ieq "shift") {
    if ($discrete_flow_shift -ne 1.0 -and $timestep_sampling -ieq "shift") {
      [void]$ext_args.Add("--discrete_flow_shift=$discrete_flow_shift")
    }
    if ($sigmoid_scale -ne 1.0) {
      [void]$ext_args.Add("--sigmoid_scale=$sigmoid_scale")
    }
  }
}
if ($guidance_scale) {
  [void]$ext_args.Add("--guidance_scale=$guidance_scale")
}

if ($weighting_scheme) {
  [void]$ext_args.Add("--weighting_scheme=$weighting_scheme")
  if ($weighting_scheme -ieq "logit_normal") {
    if ($logit_mean -ne 0.0) {
      [void]$ext_args.Add("--logit_mean=$logit_mean")
    }
    if ($logit_std -ne 1.0) {
      [void]$ext_args.Add("--logit_std=$logit_std")
    }
  }
  elseif ($weighting_scheme -ieq "mode") {
    if ($mode_scale -ne 1.29) {
      [void]$ext_args.Add("--mode_scale=$mode_scale")
    }
  }
}

if ($min_timestep -ne 0) {
  [void]$ext_args.Add("--min_timestep=$min_timestep")
}

if ($max_timestep -ne 1000) {
  [void]$ext_args.Add("--max_timestep=$max_timestep")
}

if ($show_timesteps) {
  [void]$ext_args.Add("--show_timesteps=$show_timesteps")
}

if ($max_train_steps) {
  [void]$ext_args.Add("--max_train_steps=$max_train_steps")
}
if ($max_train_epochs) {
  [void]$ext_args.Add("--max_train_epochs=$max_train_epochs")
}
if ($gradient_checkpointing) {
  [void]$ext_args.Add("--gradient_checkpointing")
}
if ($gradient_accumulation_steps) {
  [void]$ext_args.Add("--gradient_accumulation_steps=$gradient_accumulation_steps")
}

if ($base_weights) {
  [void]$ext_args.Add("--base_weights")
  foreach ($base_weight in $base_weights.Split(" ")) {
    [void]$ext_args.Add($base_weight)
  }
  [void]$ext_args.Add("--base_weights_multiplier")
  foreach ($ratio in $base_weights_multiplier.Split(" ")) {
    [void]$ext_args.Add([float]$ratio)
  }
}

if ($network_weights) {
  [void]$ext_args.Add("--network_weights=$network_weights")
  if ($dim_from_weights) {
    [void]$ext_args.Add("--dim_from_weights")
  }
}

if ($enable_lycoris) {
  $network_module = "lycoris.kohya"
  $network_dropout = "0"
  [void]$ext_args.Add("--network_args")
  [void]$ext_args.Add("algo=$algo")
  if ($algo -ine "ia3" -and $algo -ine "diag-oft") {
    if ($algo -ine "full") {
      if ($conv_dim) {
        [void]$ext_args.Add("conv_dim=$conv_dim")
        if ($conv_alpha) {
          [void]$ext_args.Add("conv_alpha=$conv_alpha")
        }
      }
      if ($use_tucker) {
        [void]$ext_args.Add("use_tucker=True")
      }
      if ($algo -ine "dylora") {
        if ($dora_wd) {
          [void]$ext_args.Add("dora_wd=True")
        }
        if ($bypass_mode) {
          [void]$ext_args.Add("bypass_mode=True")
        }
        if ($use_scalar) {
          [void]$ext_args.Add("use_scalar=True")
        }
      }
    }
    [void]$ext_args.Add("preset=$preset")
  }
  if ($dropout -and $algo -ieq "locon") {
    [void]$ext_args.Add("dropout=$dropout")
  }
  if ($train_norm -and $algo -ine "ia3") {
    [void]$ext_args.Add("train_norm=True")
  }
  if ($algo -ieq "lokr") {
    [void]$ext_args.Add("factor=$factor")
    if ($decompose_both) {
      [void]$ext_args.Add("decompose_both=True")
    }
    if ($full_matrix) {
      [void]$ext_args.Add("full_matrix=True")
    }
  }
  elseif ($algo -ieq "dylora") {
    [void]$ext_args.Add("block_size=$block_size")
  }
  elseif ($algo -ieq "diag-oft") {
    if ($rescaled) {
      [void]$ext_args.Add("rescaled=True")
    }
    if ($constrain) {
      [void]$ext_args.Add("constrain=$constrain")
    }
  }
}

if ($network_dim) {
  [void]$ext_args.Add("--network_dim=$network_dim")
}

if ($network_alpha) {
  [void]$ext_args.Add("--network_alpha=$network_alpha")
}

if ($network_dropout -ne 0) {
  [void]$ext_args.Add("--network_dropout=$network_dropout")
}

if ($network_module) {
  [void]$ext_args.Add("--network_module=$network_module")
}

if ($scale_weight_norms -ne 0) {
  [void]$ext_args.Add("--scale_weight_norms=$scale_weight_norms")
}

if ($gradient_accumulation_steps) {
  [void]$ext_args.Add("--gradient_accumulation_steps=$gradient_accumulation_steps")
}

if ($optimizer_accumulation_steps) {
  [void]$ext_args.Add("--optimizer_accumulation_steps=$optimizer_accumulation_steps")
}

if ($lr_scheduler) {
  [void]$ext_args.Add("--lr_scheduler=$lr_scheduler")
}

if ($lr_scheduler_num_cycles) {
  [void]$ext_args.Add("--lr_scheduler_num_cycles=$lr_scheduler_num_cycles")
}

if ($lr_warmup_steps) {
  [void]$ext_args.Add("--lr_warmup_steps=$lr_warmup_steps")
}

if ($lr_decay_steps) {
  [void]$ext_args.Add("--lr_decay_steps=$lr_decay_steps")
}

if ($lr_scheduler_power -ne 1) {
  [void]$ext_args.Add("--lr_scheduler_power=$lr_scheduler_power")
}

if ($lr_scheduler_timescale) {
  [void]$ext_args.Add("--lr_scheduler_timescale=$lr_scheduler_timescale")
}

if ($lr_scheduler_min_lr_ratio) {
  [void]$ext_args.Add("--lr_scheduler_min_lr_ratio=$lr_scheduler_min_lr_ratio")
}

if ($mixed_precision) {
  [void]$launch_args.Add("--mixed_precision=$mixed_precision")
  if ($mixed_precision -ieq "bf16" -or $mixed_precision -ieq "bfloat16") {
    [void]$launch_args.Add("--downcast_bf16")
  }
  [void]$ext_args.Add("--mixed_precision=$mixed_precision")
}

if ($dit_dtype) {
  [void]$ext_args.Add("--dit_dtype=$dit_dtype")
}

if ($vae_dtype) {
  [void]$ext_args.Add("--vae_dtype=$vae_dtype")
}

if ($vae_tiling) {
  [void]$ext_args.Add("--vae_tiling")
}

if ($vae_chunk_size) {
  [void]$ext_args.Add("--vae_chunk_size=$vae_chunk_size")
}

if ($vae_spatial_tile_sample_min_size -ne 256) {
  [void]$ext_args.Add("--vae_spatial_tile_sample_min_size=$vae_spatial_tile_sample_min_size")
}

if ($text_encoder_dtype) {
  [void]$ext_args.Add("--text_encoder_dtype=$text_encoder_dtype")
}

if ($fp8_base) {
  [void]$ext_args.Add("--fp8_base")
}

if ($fp8_llm) {
  [void]$ext_args.Add("--fp8_llm")
}

if ($max_data_loader_n_workers -ne 8) {
  [void]$ext_args.Add("--max_data_loader_n_workers=$max_data_loader_n_workers")
}

if ($persistent_data_loader_workers) {
  [void]$ext_args.Add("--persistent_data_loader_workers")
}

if ($blocks_to_swap -ne 0) {
  [void]$ext_args.Add("--blocks_to_swap=$blocks_to_swap")
}

if ($img_in_txt_in_offloading) {
  [void]$ext_args.Add("--img_in_txt_in_offloading")
}

if ($optimizer_type -ieq "adafactor") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("scale_parameter=False")
  [void]$ext_args.Add("warmup_init=False")
  [void]$ext_args.Add("relative_step=False")
  if ($lr_scheduler -and $lr_scheduler -ine "constant") {
    $lr_warmup_steps = 100
  }
}

if ($optimizer_type -ilike "DAdapt*") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("weight_decay=0.01")
  if ($optimizer_type -ieq "DAdaptation" -or $optimizer_type -ilike "DAdaptAdam*") {
    [void]$ext_args.Add("decouple=True")
    if ($optimizer_type -ieq "DAdaptAdam") {
      [void]$ext_args.Add("use_bias_correction=True")
    }
  }
  $lr = "1"
  if ($unet_lr) {
    $unet_lr = $lr
  }
  if ($text_encoder_lr) {
    $text_encoder_lr = $lr
  }
}

if ($optimizer_type -ieq "Lion" -or $optimizer_type -ieq "Lion8bit" -or $optimizer_type -ieq "PagedLion8bit") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("weight_decay=0.01")
  [void]$ext_args.Add("betas=.95,.98")
}

if ($optimizer_type -ieq "PagedAdamW8bit" -or $optimizer_type -ieq "AdamW" -or $optimizer_type -ieq "AdamW8bit") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
}

if ($optimizer_type -ieq "PagedAdEMAMix8bit" -or $optimizer_type -ieq "AdEMAMix8bit") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("weight_decay=0.01")
}

if ($optimizer_type -ieq "Sophia") {
  [void]$ext_args.Add("--optimizer_type=pytorch_optimizer.SophiaH")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("weight_decay=0.01")
}

if ($optimizer_type -ieq "Prodigy") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("weight_decay=0.01")
  [void]$ext_args.Add("betas=.9,.99")
  [void]$ext_args.Add("decouple=True")
  [void]$ext_args.Add("use_bias_correction=True")
  [void]$ext_args.Add("d_coef=$d_coef")
  if ($lr_warmup_steps) {
    [void]$ext_args.Add("safeguard_warmup=True")
  }
  if ($d0) {
    [void]$ext_args.Add("d0=$d0")
  }
  $lr = "1"
  if ($unet_lr) {
    $unet_lr = $lr
  }
  if ($text_encoder_lr) {
    $text_encoder_lr = $lr
  }
}

if ($optimizer_type -ieq "Ranger") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
  if (-not($train_unet_only -or $train_text_encoder_only) -or $train_text_encoder) {
    [void]$ext_args.Add("--optimizer_args")
    [void]$ext_args.Add("decouple_lr=True")
  }
}

if ($optimizer_type -ieq "Adan") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
  if (-not($train_unet_only -or $train_text_encoder_only) -or $train_text_encoder) {
    [void]$ext_args.Add("--optimizer_args")
    [void]$ext_args.Add("decouple_lr=True")
  }
}

if ($optimizer_type -ieq "StableAdamW") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
  if (-not($train_unet_only -or $train_text_encoder_only) -or $train_text_encoder) {
    [void]$ext_args.Add("--optimizer_args")
    [void]$ext_args.Add("decouple_lr=True")
  }
}

if ($optimizer_type -ieq "Tiger") {
  [void]$ext_args.Add("--optimizer_type=pytorch_optimizer.Tiger")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("weight_decay=0.01")
}

if ($optimizer_type -ilike "*ScheduleFree") {
  $lr_scheduler = ""
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("weight_decay=0.08")
  #[void]$ext_args.Add("weight_lr_power=0.001")
}

if ($optimizer_type -ieq "adammini") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("weight_decay=0.01")
}

if ($optimizer_type -ieq "adamg") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("weight_decay=0.1")
  [void]$ext_args.Add("weight_decouple=True")
}

if ($optimizer_type -ieq "came") {
  [void]$ext_args.Add("--optimizer_type=pytorch_optimizer.CAME")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("weight_decay=0.01")
}

if ($optimizer_type -ieq "SOAP") {
  [void]$ext_args.Add("--optimizer_type=pytorch_optimizer.SOAP")
}

if ($optimizer_type -ieq "sara") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("weight_decay=0.01")
  [void]$ext_args.Add("threshold=2e-3")
}

if ($max_grad_norm -ne 1.0) {
  [void]$ext_args.Add("--max_grad_norm=$max_grad_norm")
}

if ($save_every_n_steps) {
  [void]$ext_args.Add("--save_every_n_steps=$save_every_n_steps")
}

if ($save_last_n_epochs) {
  [void]$ext_args.Add("--save_last_n_epochs=$save_last_n_epochs")
}

if ($save_last_n_steps) {
  [void]$ext_args.Add("--save_last_n_steps=$save_last_n_steps")
}

if ($save_state_on_train_end) {
  [void]$ext_args.Add("--save_state_on_train_end")
}

elseif ($save_state) {
  [void]$ext_args.Add("--save_state")
  if ($save_last_n_epochs_state) {
    [void]$ext_args.Add("--save_last_n_epochs_state=$save_last_n_epochs_state")
  }
  if ($save_last_n_steps_state) {
    [void]$ext_args.Add("--save_last_n_steps_state=$save_last_n_steps_state")
  }
}

if ($resume) {
  [void]$ext_args.Add("--resume=$resume")
}

if ($wandb_api_key) {
  [void]$ext_args.Add("--wandb_api_key=$wandb_api_key")
  [void]$ext_args.Add("--log_with=wandb")
  [void]$ext_args.Add("--log_tracker_name=" + $output_name)
}

if ($enable_sample) {
  if ($sample_at_first) {
    [void]$ext_args.Add("--sample_at_first")
  }
  [void]$ext_args.Add("--sample_every_n_epochs=$sample_every_n_epochs")
  [void]$ext_args.Add("--sample_prompts=$sample_prompts")
}

if ($training_comment) {
  [void]$ext_args.Add("--training_comment=$training_comment")
}

if ($metadata_title) {
  [void]$ext_args.Add("--metadata_title=$metadata_title")
}

if ($metadata_description) {
  [void]$ext_args.Add("--metadata_description=$metadata_description")
}

if ($metadata_author) {
  [void]$ext_args.Add("--metadata_author=$metadata_author")
}

if ($metadata_license) {
  [void]$ext_args.Add("--metadata_license=$metadata_license")
}

if ($metadata_tags) {
  [void]$ext_args.Add("--metadata_tags=$metadata_tags")
}

if ($async_upload) {
  [void]$ext_args.Add("--async_upload")
  if ($huggingface_token) {
    [void]$ext_args.Add("--huggingface_token=$huggingface_token")
  }
  if ($huggingface_repo_id) {
    [void]$ext_args.Add("--huggingface_repo_id=$huggingface_repo_id")
  }
  if ($huggingface_repo_type) {
    [void]$ext_args.Add("--huggingface_repo_type=$huggingface_repo_type")
  }
  if ($huggingface_path_in_repo) {
    [void]$ext_args.Add("--huggingface_path_in_repo=$huggingface_path_in_repo")
  }
  if ($huggingface_repo_visibility) {
    [void]$ext_args.Add("--huggingface_repo_visibility=$huggingface_repo_visibility")
  }
  if ($save_state_to_huggingface) {
    [void]$ext_args.Add("--save_state_to_huggingface=$save_state_to_huggingface")
  }
  if ($resume_from_huggingface) {
    [void]$ext_args.Add("--resume_from_huggingface=$resume_from_huggingface")
  }
}

# run Training
python -m accelerate.commands.launch --num_cpu_threads_per_process=8 $launch_args "./musubi-tuner/$laungh_script.py" `
  --dataset_config="$dataset_config" `
  --dit=$dit `
  --vae=$vae `
  --text_encoder1=$text_encoder1 `
  --text_encoder2=$text_encoder2 `
  --max_train_epochs=$max_train_epochs `
  --save_every_n_epochs=$save_every_n_epochs `
  --seed=$seed  `
  --learning_rate=$lr `
  --output_name=$output_name `
  --output_dir="./output_dir" `
  --logging_dir="./logs" `
  $ext_args

Write-Output "Training finished"
Read-Host | Out-Null ;
