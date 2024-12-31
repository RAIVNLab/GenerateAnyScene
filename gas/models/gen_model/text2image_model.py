import torch

from .base_gen_model import GenModel, GenModelInstance
from .text2image_metric import Text2ImageEvalMetric

# DALLE, Stable Diffusion
text2image_models = {
	"stable-diffusion-3"  : (
		"StableDiffusion3",
		"stabilityai/stable-diffusion-3-medium-diffusers",
	),
	"stable-diffusion-2-1": (
		"StableDiffusion2",
		"stabilityai/stable-diffusion-2-1",
	),
	"stable-diffusion-xl" : (
		"StableDiffusionXL",
		"stabilityai/stable-diffusion-xl-base-1.0",
	),
	"pixart-alpha"        : (
		"PixArtAlpha",
		"PixArt-alpha/PixArt-XL-2-1024-MS",
	),
    "pixart_Sigma"        : (
        "PixArtSigma",
        "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    ),
    "deepfloyd-if-xl"     : (
        "DeepFloydIF",
        [
            "DeepFloyd/IF-I-XL-v1.0",
            "DeepFloyd/IF-II-L-v1.0",
            "stabilityai/stable-diffusion-x4-upscaler",
        ],
    ),
    "wuerstchen-V2"       : (
        "Wuerstchen",
        "warp-ai/wuerstchen",
    ),
    "playground-V2.5"     : (
        "Playground",
        "playgroundai/playground-v2.5-1024px-aesthetic",
    ),
    "flux1-schnell": (
        "FluxSchnell",
        "black-forest-labs/FLUX.1-schnell"
    ),
    "flux1-dev": (
        "FluxDev",
        "black-forest-labs/FLUX.1-dev"
    )
}

def set_model_key(model_name, key):
    text2image_models[model_name] = (text2image_models[model_name][0], key)


def list_text2image_models():
    return list(text2image_models.keys())


class Text2ImageModel(GenModel):
    def __init__(
            self,
            model_name: str,
            model: GenModelInstance = None,
            # metrics: list[str] = None,
            metrics: list = None,
            metrics_device: str = "cuda",
            precision: torch.dtype = torch.float16,  # None means using all the default metrics
            torch_device: str = "cuda",
            cache_path: str = None,
    ):
        super().__init__(GenModel, cache_path)
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
            self.metrics = Text2ImageEvalMetric(selected_metrics=metrics, device=metrics_device)
        else:
            self.metrics = "No Metrics"

        if model is None:
            print(f"Loading {model_name} ...")
            class_name, ckpt = text2image_models[model_name]
            self.model_presision = precision
            self.model = eval(class_name)(ckpt, precision, torch_device)
            print(f"Finish loading {model_name}")
        else:
            print("Using provided model...")
            self.model = model

    @torch.no_grad()
    def gen(self, gen_data):
        output = self._gen(gen_data)
        if self.metrics == "No Metrics":
            result = {"output": output}
        else:
            print(output)
            result = {"output": output, "metrics": self.metrics.eval_with_metrics(gen_data = gen_data, output = output)}
        return result

    def _data_to_str(self, data):
        # the data here for text2image model is the prompt, so it should be a str
        if isinstance(data, str):
            return data
        else:
            raise ValueError("Invalid data type")

class StableDiffusion2(GenModelInstance):
    def __init__(
            self,
            ckpt: str = "stabilityai/stable-diffusion-2-base",
            precision: torch.dtype = torch.float16,
            device: torch.device = torch.device("cuda"),
    ):
        from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

        self.pipeline = DiffusionPipeline.from_pretrained(
            ckpt, torch_dtype=precision, revision="fp16", safety_checker=None
        )
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipeline.scheduler.config
        )
        self.pipeline.enable_model_cpu_offload()

    def gen(self, prompt):
        return self.pipeline(prompt, num_inference_steps=25, height = 1024, width = 1024).images[0]


class StableDiffusionXL(GenModelInstance):
    def __init__(
            self,
            ckpt: str = "stabilityai/stable-diffusion-xl-base-1.0",
            precision: torch.dtype = torch.float16,
            device: torch.device = torch.device("cuda"),
    ):
        from diffusers import StableDiffusionXLPipeline


        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            ckpt, torch_dtype=precision
        )
        self.pipeline.enable_model_cpu_offload()

    def gen(self, prompt):
        return self.pipeline(prompt, height = 1024, width = 1024).images[0]


class StableDiffusion3(GenModelInstance):
    def __init__(
            self,
            ckpt: str = "stabilityai/stable-diffusion-3-medium-diffusers",
            precision: torch.dtype = torch.float16,
            device: torch.device = torch.device("cuda"),
    ):

        from diffusers import StableDiffusion3Pipeline
        
        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            ckpt, torch_dtype=precision
        )
        self.pipeline.enable_model_cpu_offload()

    def gen(self, prompt):
        return self.pipeline(prompt=prompt, height = 1024, width = 1024).images[0]

class DeepFloyd(GenModelInstance):
    def __init__(
        self,
        ckpt: dict = {
            "pipeline": [
                "DeepFloyd/IF-I-XL-v1.0",
                "DeepFloyd/IF-II-L-v1.0",
                "stabilityai/stable-diffusion-x4-upscaler",
            ]
        },
        precision: torch.dtype = torch.float16,
        device: torch.device = torch.device("cuda"),
    ):
        from diffusers import IFPipeline, IFSuperResolutionPipeline, DiffusionPipeline

        variant = "fp16" if precision == torch.float16 else "fp32"

        self.pipeline = IFPipeline.from_pretrained(
            ckpt[0], variant=variant, torch_dtype=precision, safety_checker=None
        )
        self.super_res_1_pipe = IFSuperResolutionPipeline.from_pretrained(
            ckpt[1], text_encoder=None, variant=variant, torch_dtype=precision
        )
        safety_modules = {
            "feature_extractor": self.pipeline.feature_extractor,
            "safety_checker": None,
            "watermarker": None,
        }
        self.super_res_2_pipe = DiffusionPipeline.from_pretrained(
            ckpt[2], **safety_modules, torch_dtype=precision
        ).to(device)
        self.pipeline.enable_model_cpu_offload()
        self.super_res_1_pipe.enable_model_cpu_offload()
        self.super_res_2_pipe.enable_model_cpu_offload()

    def gen(self, prompt):
        prompt_embeds, negative_embeds = self.pipeline.encode_prompt(prompt)
        image = self.pipeline(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            output_type="pt",
        ).images
        image = self.super_res_1_pipe(
            image=image,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            output_type="pt",
        ).images
        image = self.super_res_2_pipe(
            prompt=prompt,
            image=image,
        ).images[0]
        return image

class PixArtAlpha(GenModelInstance):
    def __init__(
            self,
            ckpt: str = "PixArt-alpha/PixArt-XL-2-1024-MS",
            precision: torch.dtype = torch.float16,
            device: torch.device = torch.device("cuda"),
    ):
        from diffusers import PixArtAlphaPipeline
        self.pipe = PixArtAlphaPipeline.from_pretrained(ckpt, torch_dtype=torch.float16, use_safetensors=True)
        self.pipe.enable_model_cpu_offload()

    def gen(self, prompt):
        image = self.pipe(prompt, height = 1024, width = 1024).images[0]
        return image


class PixArtSigma(GenModelInstance):
    def __init__(
            self,
            ckpt: str = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
            precision: torch.dtype = torch.float16,
            device: torch.device = torch.device("cuda"),
    ):
        from diffusers import PixArtSigmaPipeline
        self.pipeline = PixArtSigmaPipeline.from_pretrained(
            ckpt, torch_dtype=precision
        )
        self.pipeline.enable_model_cpu_offload()

    def gen(self, prompt):
        return self.pipeline(prompt, height = 1024, width = 1024).images[0]




class Wuerstchen(GenModelInstance):
    def __init__(
            self,
            ckpt: str = "warp-ai/wuerstchen",
            precision: torch.dtype = torch.float16,
            device: torch.device = torch.device("cuda"),
    ):
        from diffusers import WuerstchenCombinedPipeline
        self.pipeline = WuerstchenCombinedPipeline.from_pretrained(
            ckpt, torch_dtype=precision
        )
        self.pipeline.enable_model_cpu_offload()

    def gen(self, prompt):
        return self.pipeline(prompt, height = 1024, width = 1024).images[0]


class Playground(GenModelInstance):
    def __init__(
            self,
            ckpt: str = "playgroundai/playground-v2.5-1024px-aesthetic",
            precision: torch.dtype = torch.float16,
            device: torch.device = torch.device("cuda"),
    ):
        from diffusers import DiffusionPipeline
        self.pipeline = DiffusionPipeline.from_pretrained(
            ckpt, torch_dtype=precision, variant="fp16"
        )
        self.pipeline.enable_model_cpu_offload()

    def gen(self, prompt):
        return self.pipeline(prompt=prompt, num_inference_steps=50, guidance_scale=3, height = 1024, width = 1024).images[0]
    
class FluxSchnell(GenModelInstance):
    def __init__(
            self,
            ckpt: str = "black-forest-labs/FLUX.1-schnell",
            precision: torch.dtype = torch.bfloat16,
            device: torch.device = torch.device("cuda"),
    ):
        from diffusers import FluxPipeline

        self.pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", 
            torch_dtype=torch.bfloat16
            )
        self.pipeline.enable_model_cpu_offload()
    def gen(self, prompt):
        return self.pipeline(
            prompt, 
            height = 1024, 
            width = 1024, 
            guidance_scale=0.0, 
            num_inference_steps=4,
            max_sequence_length=256
            ).images[0]
        
class FluxDev(GenModelInstance):
    def __init__(
            self,
            ckpt: str = "black-forest-labs/FLUX.1-dev",
            precision: torch.dtype = torch.bfloat16,
            device: torch.device = torch.device("cuda"),
    ):
        from diffusers import FluxPipeline

        self.pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", 
            torch_dtype=torch.bfloat16
            )
        self.pipeline.enable_model_cpu_offload()
    def gen(self, prompt):
        return self.pipeline(
            prompt, 
            height = 1024, 
            width = 1024, 
            guidance_scale=3.5, 
            num_inference_steps=50,
            max_sequence_length=512
            ).images[0]
        

