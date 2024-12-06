import argparse, os, random, time
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
import io, imageio

from .base_gen_model import GenModel, GenModelInstance
from .text2video_metric import Text2VideoEvalMetric


text2video_models = {
    "text2vid-zero": (
        "Text2VideoZero", 
        "runwayml/stable-diffusion-v1-5"
    ),
    "zeroscope": (
        "ZeroScope",
        "cerspense/zeroscope_v2_576w"
    ),
    "modelscope-t2v": (
        "ModelScopeT2V", 
        "damo-vilab/text-to-video-ms-1.7b"
    ),
    
    "animatediff": (
        "AnimateDiff", [
            "guoyww/animatediff-motion-adapter-v1-5-2", 
            "SG161222/Realistic_Vision_V5.1_noVAE"
        ]
    ),
    
    "animateLCM": (
        "AnimateLCM", [
            "wangfuyun/AnimateLCM", 
            "emilianJR/epiCRealism", 
            "AnimateLCM_sd15_t2v_lora.safetensors", 
            "lcm-lora"
        ]
    ),
    "free-init": (
        "FreeInit", [
            "guoyww/animatediff-motion-adapter-v1-5-2", 
            "SG161222/Realistic_Vision_V5.1_noVAE"
        ]
    ),
    "VideoCraft2": (
        "VideoCraft2",[
            "model_zoo/VideoCrafter/configs/inference_t2v_512_v2.0.yaml",
            "model_zoo/VideoCrafter/checkpoints/base_512_v2/model.ckpt"
        ]
    ),
    "opensora": (
        "OpenSora",
        "model_zoo/Open-Sora/configs/opensora-v1-2/inference/sample.py"
    ),
    "cogvideox": (
        "CogVideoX",
        "THUDM/CogVideoX-2b"
    )
}
def set_model_key(model_name, key):
    text2video_models[model_name] = (text2video_models[model_name][0], key)

def list_text2video_models():
    return list(text2video_models.keys())


class Text2VideoModel(GenModel):
    def __init__(
        self,
        model_name: str,
        model: GenModelInstance = None,
        metrics: list[str] = None,
        metrics_device: str = "cuda",
        precision: torch.dtype = torch.float16,# None means using all the default metrics
        torch_device: str = "cuda",
        cache_path: str = None,
    ):
        super().__init__(model_name, cache_path)


        assert isinstance(torch_device, int) or torch_device in ["cpu","cuda"] or torch_device.startswith("cuda:")
        assert isinstance(metrics_device, int) or metrics_device in ["cpu","cuda"] or metrics_device.startswith("cuda:")
        if isinstance(metrics_device, str):
            metrics_device = torch.device(metrics_device)
        else:
            if metrics_device == -1:
                metrics_device = (
                    torch.device("cuda") if torch.cuda.is_available() else "cpu"
                )
            else:
                metrics_device = torch.device(f"cuda:{metrics_device}")
        if isinstance(torch_device, str):
            torch_device = torch.device(torch_device)
        else:
            if torch_device == -1:
                torch_device = (
                    torch.device("cuda") if torch.cuda.is_available() else "cpu"
                )
            else:
                torch_device = torch.device(f"cuda:{torch_device}")
                
        if metrics is not None:
            self.metrics = Text2VideoEvalMetric(selected_metrics=metrics, device=metrics_device)
        else: 
            self.metrics = "No Metrics"
        if model is None:
            print(f"Loading {model_name} ...")
            class_name, ckpt = text2video_models[model_name]
            self.model_presision = precision
            self.model = eval(class_name)(ckpt, precision, torch_device)
            print(f"Finish loading {model_name}")
        else:
            print("Using provided model...")
            self.model = model
            
    @torch.no_grad()
    def gen(self, gen_data):
        output = self._gen(gen_data)
        # release the memory
        torch.cuda.empty_cache()
        print(type(output))
        if self.metrics == "No Metrics":
            result = {"output": output}
        else:
            result = {"output": output, "metrics": self.metrics.eval_with_metrics(gen_data = gen_data, video=output)}
        return result
            
    def _data_to_str(self, data):
        # the data here for text2image model is the prompt, so it should be a str
        if isinstance(data, str):
            return data
        else:
            raise ValueError("Invalid data type")


class Text2VideoZero(GenModelInstance):
    def __init__(self, ckpt:str = "runwayml/stable-diffusion-v1-5", precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda"), fps = 16, video_length = 16, height = 512, width = 512):
        from diffusers import TextToVideoZeroPipeline
        self.pipeline = TextToVideoZeroPipeline.from_pretrained(ckpt, torch_dtype=precision).to(device)
        self.fps = fps
        self.video_length = video_length
        self.height = height
        self.width = width
        
        self.pipeline.safety_checker = lambda images, clip_input: (images, False)
    def gen(self, prompt):
        if self.video_length <= 8:
            result = self.pipeline(prompt, video_length = self.video_length, height = self.height, width = self.width).images
            with io.BytesIO() as video_bytes:
                writer = imageio.get_writer(video_bytes, format='mp4', fps=self.fps)
                
                for frame in result:
                    frame = (frame * 255).astype(np.uint8)
                    writer.append_data(frame)
                
                writer.close()
                video_bytes.seek(0)
                video_data = video_bytes.read()
        else:
            chunk_size = 8
            seed = 0
            result = []
            chunk_ids = np.arange(0, self.video_length, chunk_size - 1)
            generator = torch.Generator(device="cuda")
            for i in range(len(chunk_ids)):
                ch_start = chunk_ids[i]
                ch_end = self.video_length if i == len(chunk_ids) - 1 else chunk_ids[i + 1]
                # Attach the first frame for Cross Frame Attention
                frame_ids = [0] + list(range(ch_start, ch_end))
                # Fix the seed for the temporal consistency
                generator.manual_seed(seed)
                output = self.pipeline(prompt=prompt, video_length=len(frame_ids), generator=generator, frame_ids=frame_ids)
                result.append(output.images[1:])
            result = np.concatenate(result)
            with io.BytesIO() as video_bytes:
                writer = imageio.get_writer(video_bytes, format='mp4', fps=self.fps)
                
                for frame in result:
                    frame = (frame * 255).astype(np.uint8)
                    writer.append_data(frame)
                
                writer.close()
                video_bytes.seek(0)
                video_data = video_bytes.read()
            
        return video_data

class Text2VideoZeroSDXL(GenModelInstance):
    def __init__(self, ckpt:str = "stabilityai/stable-diffusion-xl-base-1.0", precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda"), fps = 16, video_length = 16, height = 256, width = 256):
        from diffusers import TextToVideoZeroSDXLPipeline
        self.pipeline = TextToVideoZeroSDXLPipeline.from_pretrained(ckpt, torch_dtype=precision, use_safetensors=True).to(device)
        self.pipeline.enable_model_cpu_offload()
        self.fps = fps  
        self.video_length = video_length
        self.height = height
        self.width = width
        
        self.pipeline.safety_checker = lambda images, clip_input: (images, False)
    def gen(self, prompt):
        if self.video_length <= 8:
            result = self.pipeline(prompt, video_length = self.video_length, height = self.height, width = self.width).images
            with io.BytesIO() as video_bytes:
                writer = imageio.get_writer(video_bytes, format='mp4', fps=self.fps)
                
                for frame in result:
                    frame = (frame * 255).astype(np.uint8)
                    writer.append_data(frame)
                
                writer.close()
                video_bytes.seek(0)
                video_data = video_bytes.read()
        else:
            chunk_size = 8
            seed = 0
            result = []
            chunk_ids = np.arange(0, self.video_length, chunk_size - 1)
            generator = torch.Generator(device="cuda")
            for i in range(len(chunk_ids)):
                ch_start = chunk_ids[i]
                ch_end = self.video_length if i == len(chunk_ids) - 1 else chunk_ids[i + 1]
                # Attach the first frame for Cross Frame Attention
                frame_ids = [0] + list(range(ch_start, ch_end))
                # Fix the seed for the temporal consistency
                generator.manual_seed(seed)
                output = self.pipeline(prompt=prompt, video_length=len(frame_ids), generator=generator, frame_ids=frame_ids)
                result.append(output.images[1:])
            result = np.concatenate(result)
            with io.BytesIO() as video_bytes:
                writer = imageio.get_writer(video_bytes, format='mp4', fps=self.fps)
                
                for frame in result:
                    frame = (frame * 255).astype(np.uint8)
                    writer.append_data(frame)
                
                writer.close()
                video_bytes.seek(0)
                video_data = video_bytes.read()
            
        return video_data

class ZeroScope(GenModelInstance):
    def __init__(self, ckpt: str = "cerspense/zeroscope_v2_576w", precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda"), fps = 16, video_length = 16, height = 320, width = 576):
            import torch
            from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
            from diffusers.utils import export_to_video
            self.pipeline = DiffusionPipeline.from_pretrained(ckpt, torch_dtype=precision)
            self.pipeline.enable_model_cpu_offload()
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)
            self.fps = fps
            self.video_length = video_length
            self.height = height
            self.width = width
            self.pipeline.safety_checker = lambda images, clip_input: (images, False)
    def gen(self, prompt):
        video_frames = self.pipeline(prompt, num_inference_steps=40, height=320, width=576, num_frames=self.video_length).frames[0]
        with io.BytesIO() as video_bytes:
            writer = imageio.get_writer(video_bytes, format='mp4', fps=self.fps)
            for frame in video_frames:
                frame = (frame * 255).astype(np.uint8)
                writer.append_data(frame)
            writer.close()
            video_bytes.seek(0)
            video_data = video_bytes.read()
        return video_data


class ZeroScopeXL(GenModelInstance):
    def __init__(self, ckpt: list = ["cerspense/zeroscope_v2_576w","cerspense/zeroscope_v2_XL"], precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda"), fps = 16, video_length = 16, height = 576, width = 1024):
        from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

        self.pipeline = DiffusionPipeline.from_pretrained(ckpt[0], torch_dtype=precision).to(device)
        self.pipeline.enable_model_cpu_offload()
        self.pipeline.unet.enable_forward_chunking(chunk_size=1, dim=1)
        self.pipeline.enable_vae_slicing() 

        self.upscale = DiffusionPipeline.from_pretrained(ckpt[1], torch_dtype=torch.float16).to(device)
        self.upscale.scheduler = DPMSolverMultistepScheduler.from_config(self.upscale.scheduler.config)
        self.upscale.enable_model_cpu_offload()
        self.upscale.unet.enable_forward_chunking(chunk_size=1, dim=1)
        self.upscale.enable_vae_slicing()
        
        self.fps = fps
        self.video_length = video_length
        self.height = height
        self.width = width
        
        self.pipeline.safety_checker = lambda images, clip_input: (images, False)

    def gen(self, prompt):
        video_frames = self.pipeline(prompt, num_frames=self.video_length).frames[0]
        video = [Image.fromarray((frame*255).astype(np.uint8)).resize((self.width, self.height)) for frame in video_frames]
        video_frames = self.upscale(prompt, video=video, strength=0.6).frames[0]
        with io.BytesIO() as video_bytes:
            writer = imageio.get_writer(video_bytes, format='mp4', fps=self.fps)
            
            for frame in video_frames:
                frame = (frame * 255).astype(np.uint8)
                writer.append_data(frame)
            
            writer.close()
            video_bytes.seek(0)
            video_data = video_bytes.read()
        
        return video_data


class ModelScopeT2V(GenModelInstance):
    def __init__(self, ckpt: str = "damo-vilab/text-to-video-ms-1.7b", precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda"),EnableVAESlicing=True, fps = 16, video_length = 16, height = 512, width = 512):
        from diffusers import DiffusionPipeline
        self.pipeline = DiffusionPipeline.from_pretrained(ckpt, torch_dtype=precision).to(device)
        self.pipeline.enable_model_cpu_offload()
        if EnableVAESlicing:
            self.pipeline.enable_vae_slicing()
            
        self.fps = fps
        self.video_length = video_length
        self.height = height
        self.width = width

        self.pipeline.safety_checker = lambda images, clip_input: (images, False)
    def gen(self, prompt):
        video_frames = self.pipeline(prompt, num_frames = self.video_length, height = self.height, width = self.width).frames[0]
        with io.BytesIO() as video_bytes:
            writer = imageio.get_writer(video_bytes, format='mp4', fps=self.fps)
            
            for frame in video_frames:
                frame = (frame * 255).astype(np.uint8)
                writer.append_data(frame)
            
            writer.close()
            video_bytes.seek(0)
            video_data = video_bytes.read()
        return video_data
    

class AnimateDiff(GenModelInstance):
    def __init__(self, ckpt: list = ["guoyww/animatediff-motion-adapter-v1-5-2", "SG161222/Realistic_Vision_V5.1_noVAE"], precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda"), fps = 16, video_length = 16, height = 512, width = 512):
        from diffusers import MotionAdapter, AnimateDiffPipeline, DDIMScheduler
        adapter = MotionAdapter.from_pretrained(ckpt[0])
        model_id = ckpt[1]
        self.pipeline = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=precision)
        self.pipeline.scheduler = DDIMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            beta_schedule="linear",
            clip_sample=False,
            timestep_spacing="linspace",
            steps_offset=1
        )
        self.pipeline.enable_vae_slicing()
        self.pipeline.enable_vae_tiling()
        self.pipeline.enable_model_cpu_offload()
        
        self.fps = fps
        self.video_length = video_length
        self.height = height
        self.width = width
        
        # self.pipeline.safety_checker = lambda images, clip_input: (images, False)

    @torch.no_grad()
    def gen(self, prompt):
        output = self.pipeline(prompt, num_frames=16, guidance_scale=7.5, num_inference_steps=25, generator=torch.Generator("cpu").manual_seed(42), height = None, width = None)
        result = output.frames[0]
        frame_array = [np.array(frame) for frame in result]
        with io.BytesIO() as video_bytes:
            # Use imageio to write frames to a video
            writer = imageio.get_writer(video_bytes, format='mp4', fps=self.fps)
            
            for frame in frame_array:
                # frame = (frame * 255).astype(np.uint8)
                writer.append_data(frame)
            
            writer.close()
            video_bytes.seek(0)
            video_data = video_bytes.read()
        
        return video_data
       

class AnimateLCM(GenModelInstance):
    def __init__(self, ckpt: list = ["wangfuyun/AnimateLCM", "emilianJR/epiCRealism", "AnimateLCM_sd15_t2v_lora.safetensors", "lcm-lora"], precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda"), fps = 16, video_length = 16, height = 512, width = 512):
        from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
        adapter = MotionAdapter.from_pretrained(ckpt[0])
        self.pipeline = AnimateDiffPipeline.from_pretrained(ckpt[1], motion_adapter=adapter, torch_dtype=precision).to(device)
        self.pipeline.scheduler = LCMScheduler.from_config(self.pipeline.scheduler.config, beta_schedule="linear")
        self.pipeline.load_lora_weights(ckpt[0], weight_name=ckpt[2], adapter_name=ckpt[3])
        self.pipeline.enable_vae_slicing()
        self.pipeline.enable_model_cpu_offload()
        
        self.fps = fps
        self.video_length = video_length
        self.height = height
        self.width = width
        
        self.pipeline.safety_checker = lambda images, clip_input: (images, False)

    def gen(self, prompt):
        output = self.pipeline(prompt, num_frames=16, guidance_scale=1.5, num_inference_steps=6, generator=torch.Generator("cpu").manual_seed(0), height = None, width = None)
        result = output.frames[0]
        frame_array = [np.array(frame) for frame in result]
        with io.BytesIO() as video_bytes:
            # Use imageio to write frames to a video
            writer = imageio.get_writer(video_bytes, format='mp4', fps=self.fps)
            
            for frame in frame_array:
                writer.append_data(frame)
            
            writer.close()
            video_bytes.seek(0)
            video_data = video_bytes.read()
        
        return video_data

class FreeInit(GenModelInstance):
    def __init__(self, ckpt: list = ["guoyww/animatediff-motion-adapter-v1-5-2", "SG161222/Realistic_Vision_V5.1_noVAE"], precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda"), fps = 16, video_length = 16, height = 512, width = 512):
        from diffusers import MotionAdapter, AnimateDiffPipeline, DDIMScheduler
        from diffusers.utils import export_to_gif
        adapter = MotionAdapter.from_pretrained(ckpt[0])
        model_id = ckpt[1]
        self.pipeline = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=precision)
        self.pipeline.scheduler = DDIMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            beta_schedule="linear",
            clip_sample=False,
            timestep_spacing="linspace",
            steps_offset=1
        )
        self.pipeline.enable_model_cpu_offload()
        self.pipeline.enable_vae_slicing()
        self.pipeline.enable_vae_tiling()
        self.pipeline.enable_free_init(method="butterworth", use_fast_sampling=True)
        
        self.fps = fps
        self.video_length = video_length
        self.height = height
        self.width = width
        
        self.pipeline.safety_checker = lambda images, clip_input: (images, False)

    def gen(self, prompt):
        output = self.pipeline(prompt, num_frames=16, guidance_scale=7.5, num_inference_steps=20, generator=torch.Generator("cpu").manual_seed(666), height = None, width = None)
        self.pipeline.disable_free_init()

        result = output.frames[0]
        frame_array = [np.array(frame) for frame in result]
        with io.BytesIO() as video_bytes:
            writer = imageio.get_writer(video_bytes, format='mp4', fps=self.fps)
            
            for frame in frame_array:
                writer.append_data(frame)
            
            writer.close()
            video_bytes.seek(0)
            video_data = video_bytes.read()
        
        return video_data
    
class VideoCraft2(GenModelInstance):
    def __init__(self, ckpt: list = ["model_zoo/VideoCrafter/configs/inference_t2v_512_v2.0.yaml","model_zoo/VideoCrafter/checkpoints/base_512_v2/model.ckpt"], precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
        self.input_ns = argparse.Namespace(**{})
        self.input_ns.seed = 123
        self.input_ns.mode = 'base'
        self.input_ns.ckpt_path = os.path.join(current_dir, ckpt[1])
        self.input_ns.config = os.path.join(current_dir, ckpt[0])
        self.input_ns.savefps = 16
        self.input_ns.ddim_steps = 50
        self.input_ns.n_samples = 1
        self.input_ns.ddim_eta = 1.0
        self.input_ns.bs = 1
        self.input_ns.height = 320
        self.input_ns.width = 512
        self.input_ns.frames = 16
        self.input_ns.fps = 16
        self.input_ns.unconditional_guidance_scale = 12.0
        seed_everything(self.input_ns.seed)            
        self.gpu_no = 0
    
    def gen(self, prompt, **kwargs):
        args = self.input_ns
        import lvdm
        from funcs import load_model_checkpoint, load_prompts, load_image_batch, get_filelist, save_videos
        from funcs import batch_ddim_sampling
        from utils.utils import instantiate_from_config
        config = OmegaConf.load(args.config)
        model_config = config.pop("model", OmegaConf.create())
        model = instantiate_from_config(model_config)
        model = model.cuda(self.gpu_no)
        assert os.path.exists(args.ckpt_path), f"Error: checkpoint [{args.ckpt_path}] Not Found!"
        model = load_model_checkpoint(model, args.ckpt_path)
        model.eval()
        ## sample shape
        assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
        ## latent noise shape
        h, w = args.height // 8, args.width // 8
        frames = model.temporal_length if args.frames < 0 else args.frames
        channels = model.channels

        ## step 2: load data
        # -----------------------------------------------------------------
        # assert os.path.exists(args.prompt_file), "Error: prompt file NOT Found!"
        prompts = [prompt]

        ## step 3: run over samples
        ## -----------------------------------------------------------------
        start = time.time()
        noise_shape = [1, channels, frames, h, w]
        fps = torch.tensor([args.fps]*1).to(model.device).long()

        if isinstance(prompts, str):
            prompts = [prompts]
        text_emb = model.get_learned_conditioning(prompts)

        cond = {"c_crossattn": [text_emb], "fps": fps}

        ## inference
        batch_samples = batch_ddim_sampling(model, cond, noise_shape, args.n_samples, \
                                                args.ddim_steps, args.ddim_eta, args.unconditional_guidance_scale, **kwargs)
        
        video = batch_samples[0][0].detach().cpu() # [c, t, h, w]
              
        
        video = torch.clamp(video.float(), -1., 1.)
        video.sub_(-1).div_(2)  
        video = video.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 3, 0).to(torch.uint8).numpy()
        
        with io.BytesIO() as video_bytes:
            # Use imageio to write frames to a video
            writer = imageio.get_writer(video_bytes, format='mp4', fps=16)
            
            for frame in video:
                writer.append_data(frame)
            
            writer.close()
            video_bytes.seek(0)
            video_data = video_bytes.read()
        return video_data
     
class OpenSora(GenModelInstance):
    
    def __init__(self, ckpt: str = "model_zoo/Open-Sora/configs/opensora-v1-2/inference/sample.py", precision: torch.dtype = torch.bfloat16, device: torch.device = torch.device("cuda")):
        from mmengine.runner import set_random_seed
        from opensora.datasets.aspect import get_num_frames
        from opensora.registry import MODELS, SCHEDULERS, build_module


        self.device = device
        self.args_dict = {
            "config": os.path.join(current_dir, ckpt),
            "num_frames": 18,
            "image_size": (512, 512),
            "fps": 16,
            
        }
        self._initialize_config()
        self.dtype = precision
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        set_random_seed(seed=self.cfg.get("seed", 1024))

        self.text_encoder = build_module(self.cfg.text_encoder, MODELS, device=device)
        self.vae = build_module(self.cfg.vae, MODELS).to(device, self.dtype).eval()

        self.image_size = self.cfg.get("image_size", None)
        self.num_frames = get_num_frames(self.cfg.num_frames)

        input_size = (self.num_frames, *self.image_size)
        self.latent_size = self.vae.get_latent_size(input_size)
        
        self.cfg.model.enable_flash_attn = False
        self.cfg.model.enable_layernorm_kernel = False
        self.model = (
            build_module(
                self.cfg.model,
                MODELS,
                input_size=self.latent_size,
                in_channels=self.vae.out_channels,
                caption_channels=self.text_encoder.output_dim,
                model_max_length=self.text_encoder.model_max_length,
                enable_sequence_parallelism=False,
            )
            .to(device, self.dtype)
            .eval()
        )
        self.text_encoder.y_embedder = self.model.y_embedder 
        self.scheduler = build_module(self.cfg.scheduler, SCHEDULERS)

    def _initialize_config(self):
        import argparse
        from opensora.utils.config_utils import merge_args, read_config
        parsed_args = self._parse_args()
        # parser = argparse.ArgumentParser()
        # parser.add_argument("--config", default=self.args_dict["config"], help="model config file path")
        # parsed_args = parser.parse_args()
        # args = argparse.Namespace(**self.args_dict)
        for key, value in self.args_dict.items():
            setattr(parsed_args, key, value)
        self.cfg = read_config(parsed_args.config)
        self.cfg = merge_args(self.cfg, parsed_args, training=False)

    def _parse_args(self):
        from opensora.utils.config_utils import str2bool
        class Args:
            config = "model_zoo/Open-Sora/configs/opensora-v1-2/inference/sample.py"
            seed = None
            ckpt_path = None
            batch_size = None
            outputs = None
            flash_attn = False
            layernorm_kernel = False
            resolution = None
            data_path = None
            dtype = None
            save_dir = None
            sample_name = None
            start_index = None
            end_index = None
            num_sample = None
            prompt_as_path = False
            verbose = None
            prompt_path = None
            prompt = None
            llm_refine = None
            prompt_generator = None
            
            
            num_frames = None
            fps = None
            save_fps = None
            image_size = None
            frame_interval = None
            aspect_ratio = None
            watermark = None
            num_sampling_steps = None
            cfg_scale = None
            loop = None
            condition_frame_length = None
            reference_path = None
            mask_strategy = None
            aes = None
            flow = None
            camera_motion = None
        args = Args()
        return args
            
    def gen(self, prompt):
        from opensora.models.text_encoder.t5 import text_preprocessing
        from opensora.utils.inference_utils import (
            append_score_to_prompts,
            apply_mask_strategy,
            extract_json_from_prompts,
            extract_prompts_loop,
            merge_prompt,
            prepare_multi_resolution_info,
            split_prompt,
        )
        self.fps = self.cfg.fps
        save_fps = self.cfg.get("save_fps", self.fps // self.cfg.get("frame_interval", 1))
        multi_resolution = self.cfg.get("multi_resolution", None)
        prompts = [prompt]

        batch_prompts, refs, ms  = extract_json_from_prompts(prompts, [''], [''])
        model_args = prepare_multi_resolution_info(
            multi_resolution, len(batch_prompts), self.image_size, self.num_frames, self.fps, self.device, self.dtype
        )
        batched_prompt_segment_list = []
        batched_loop_idx_list = []
        for prompt in batch_prompts:
            prompt_segment_list, loop_idx_list = split_prompt(prompt)
            batched_prompt_segment_list.append(prompt_segment_list)
            batched_loop_idx_list.append(loop_idx_list)

        for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
            batched_prompt_segment_list[idx] = append_score_to_prompts(
                prompt_segment_list,
                aes=self.cfg.get("aes", None),
                flow=self.cfg.get("flow", None),
                camera_motion=self.cfg.get("camera_motion", None),
            )

        for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
            batched_prompt_segment_list[idx] = [text_preprocessing(prompt) for prompt in prompt_segment_list]

        batch_prompts = []
        for prompt_segment_list, loop_idx_list in zip(batched_prompt_segment_list, batched_loop_idx_list):
            batch_prompts.append(merge_prompt(prompt_segment_list, loop_idx_list))

        video_clips = []
        batch_prompts_loop = extract_prompts_loop(batch_prompts, 0)

        with torch.no_grad():
            z = torch.randn(len(batch_prompts), self.vae.out_channels, *self.latent_size, device=self.device, dtype=self.dtype)
            masks = apply_mask_strategy(z, [], [""], 0, align=None)
            samples = self.scheduler.sample(
                self.model,
                self.text_encoder,
                z=z,
                prompts=batch_prompts_loop,
                device=self.device,
                additional_args=model_args,
                progress=False,
                mask=masks,
            )
            samples = self.vae.decode(samples.to(self.dtype), num_frames=self.num_frames)
            video_clips.append(samples)

        video = [video_clips[0][0]]
        video = torch.cat(video, dim=1)
        video.clamp_(min=-1, max=1)
        video.sub_(-1).div_(max(2, 1e-5))        
        video = video.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 3, 0).to("cpu", torch.uint8).numpy()
        with io.BytesIO() as video_bytes:
            # Use imageio to write frames to a video
            writer = imageio.get_writer(video_bytes, format='mp4', fps=16)
            
            for frame in video:
                writer.append_data(frame)
            
            writer.close()
            video_bytes.seek(0)
            video_data = video_bytes.read()
        
        return video_data
        



class CogVideoX(GenModelInstance):
    def __init__(self, ckpt: str = "THUDM/CogVideoX-2b", precision: torch.dtype = torch.float16, device: torch.device = torch.device("cuda")):
        from diffusers import CogVideoXPipeline
        self.pipeline = CogVideoXPipeline.from_pretrained(ckpt, torch_dtype=precision)
        self.pipeline.enable_model_cpu_offload()
        self.fps = 16
        self.video_length = 16
        self.height = 480
        self.width = 720
        
    def gen(self, prompt):
        video_frames = self.pipeline(prompt, num_frames=self.video_length, height=self.height, width=self.width, guidance_scale = 6).frames[0]
        frame_array = [np.array(frame) for frame in video_frames]

        with io.BytesIO() as video_bytes:
            writer = imageio.get_writer(video_bytes, format='mp4', fps=self.fps)
            
            for frame in frame_array:
                writer.append_data(frame)
            
            writer.close()
            video_bytes.seek(0)
            video_data = video_bytes.read()
        return video_data
