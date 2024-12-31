import json

import diskcache
import torch

from .. import Model

class GenModelInstance:
    def gen(self, prompt):
        "(Abstract method) abstract method"


class GenModel(Model):
    def __init__(
        self,
        model_name: str,
        cache_path: str = None,
	):
        self.model = None
        self.model_name = model_name
        self.cache_path = cache_path

        if self.cache_path is not None:
            print(f"[IMPORTANT] model cache is enabled, cache path: {cache_path}")
        else:
            print("[IMPORTANT] model cache is disabled")

    def _output_to_str(self, output):
        """ abstract method """

    @torch.no_grad()
    def _gen(self, gen_data):
        prompt = gen_data["prompt"]
        if self.cache_path is None:
            return self.model.gen(prompt)
        else:  # TODO: implement cache here waiting for test
            with diskcache.Cache(self.cache_path, size_limit=10 * (2 ** 30)) as cache:
                key = json.dumps([prompt, self.model_name])  # fix bugs, add model name
                response = cache.get(key, None)
                if response is None:
                    response = self.model.gen(prompt)
                    cache.set(key, response)
                return response

    @torch.no_grad()
    def gen(self, gen_data):
        output = self._gen(gen_data)
        result = {"output": output}
        return result

class Metric:
    def __init__(self, device: str = "cuda"):
        # check if the device is cuda or cpu or a digital string
        self.device = device

    @torch.no_grad()
    def compute(self, gen_data, image):
        "(Abstract method) abstract compute matric method"
