import torch
from PIL import Image
from tempfile import NamedTemporaryFile
import os

metrics_dict = {
    "ClipScore": "ClipScore",
    "VQAScore": "VQAScore",
    "PickScore": "PickScore",
    "ImageRewardScore": "ImageRewardScore",
    "ProgrammaticDSGTIFAScore": "ProgrammaticDSGTIFAScore"
}

class Textto3DEvalMetric():
    def __init__(
        self,
        are_metrics_preloaded: bool = True,
        selected_metrics: list = None,
        device: str = "cuda",
    ):
        self.device = device
        if selected_metrics is None:
            selected_metrics = list(metrics_dict.keys())
        
        if are_metrics_preloaded is True:
            self.metrics = {}
            for metric_name in selected_metrics:
                self.metrics[metric_name] = eval(metrics_dict[metric_name])(device=self.device)
        else:
            self.metrics = {}
            for metric_name in selected_metrics:
                self.metrics[metric_name] = eval(metrics_dict[metric_name])(device=self.device)
            for metric_name in selected_metrics:
                self.metrics[metric_name] = metrics_dict[metric_name]
        print(f"use {self.metrics.keys()}, Are metrics preloaded?: {are_metrics_preloaded}")


    def _decode_video(self, video_data):
        import imageio
        import numpy as np
        import io
        with io.BytesIO(video_data) as video_bytes:
            video_bytes.seek(0)
            reader = imageio.get_reader(video_bytes, format='mp4')
            frames = []
            for frame in reader:
                frame = Image.fromarray(frame)
                frames.append(frame)
            reader.close()
        return frames

    def _grid_video(self, frames, row = 3, col = 3):
        num_cells = row * col  

        if len(frames) < num_cells:
            raise ValueError("Not enough frames to fill the grid.")

        segment_size = len(frames) // num_cells

        sampled_frames = [frames[i * segment_size + segment_size // 2] for i in range(num_cells)]

        width, height = sampled_frames[0].size
        grid_image = Image.new('RGB', (width * col, height * row))

        for idx, frame in enumerate(sampled_frames):
            x = (idx % col) * width
            y = (idx // col) * height
            grid_image.paste(frame, (x, y))

        return grid_image   
        
    def eval_with_metrics(self, gen_data: dict, video: bytes, mode="frame", sample_frames=16):

        frames = self._decode_video(video)

        # Generate grid if mode is grid
        if mode == "grid":
            grid_image = self._grid_video(frames)
            images_to_evaluate = [grid_image]
        elif mode == "frame":
            images_to_evaluate = frames
        else:
            raise ValueError("Invalid mode. Use 'frame' or 'grid'.")

        results = {}
        for metric_name in self.metrics:
            if isinstance(self.metrics[metric_name], str):
                current_metric = eval(self.metrics[metric_name])(device=self.device)
                if mode == "grid":
                    score = current_metric.compute(gen_data, images_to_evaluate[0])
                    results[metric_name] = {"score": score}
                else:
                    frame_scores = [current_metric.compute(gen_data, img) for img in images_to_evaluate]
                    avg_score = sum(frame_scores) / len(frame_scores) if len(frame_scores) > 0 else 0.0
                    results[metric_name] = {
                        "average": avg_score,
                        "frame_scores": frame_scores
                    }
            else:
                if mode == "grid":
                    score = self.metrics[metric_name].compute(gen_data, images_to_evaluate[0])
                    results[metric_name] = {"score": score}
                else:
                    frame_scores = [self.metrics[metric_name].compute(gen_data, img) for img in images_to_evaluate]
                    avg_score = sum(frame_scores) / len(frame_scores) if len(frame_scores) > 0 else 0.0
                    results[metric_name] = {
                        "average": avg_score,
                        "frame_scores": frame_scores
                    }
        return results
    
    def list_metrics(self):
        return list(metrics_dict.keys())

class Metric:
    def __init__(self, device: str = "cuda"):
        self.device = device
        
    @torch.no_grad()
    def compute(self, gen_data, video):
        "(Abstract method) abstract compute metric method"

class ClipScore(Metric):
    def __init__(self, model_name_or_path='openai:ViT-L-14-336', device="cuda"):
        super().__init__(device=device)
        import t2v_metrics
        self.clipscore = t2v_metrics.CLIPScore(model='openai:ViT-L-14-336')

    def compute(self, image: Image.Image, gen_data: dict):
        prompt = gen_data['prompt']
        with NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            image.save(temp_file, format="PNG")
            temp_file_path = temp_file.name

        # Calculate the score using the temporary file path
        score = self.clipscore(images=[temp_file_path], texts=[prompt]).item()

        return score


class VQAScore(Metric):
    def __init__(self, model="clip-flant5-xxl", device="cuda"):
        super().__init__(device=device)
        import t2v_metrics
        self.clip_flant5_score = t2v_metrics.VQAScore(model=model)

    def compute(self, image: Image.Image, gen_data: dict):
        prompt = gen_data['prompt']
        with NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            image.save(temp_file, format="PNG")
            temp_file_path = temp_file.name

        score = self.clip_flant5_score(images=[temp_file_path], texts=[prompt]).item()
        return score


class PickScore(Metric):
    def __init__(self, model='pickscore-v1', device="cuda"):
        super().__init__(device=device)

        import t2v_metrics
        self.pick_score = t2v_metrics.CLIPScore(model=model)

    def compute(self, image: Image.Image, gen_data: dict):
        prompt = gen_data['prompt']
        with NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            image.save(temp_file, format="PNG")
            temp_file_path = temp_file.name

        # Calculate the score using the temporary file path
        score = self.pick_score(images=[temp_file_path], texts=[prompt]).item()
        return score


class ImageRewardScore(Metric):
    def __init__(self, model='image-reward-v1', device="cuda"):
        super().__init__(device=device)
        import t2v_metrics
        self.image_reward_score = t2v_metrics.ITMScore(model=model)

    def compute(self, image: Image.Image, gen_data: dict):
        prompt = gen_data['prompt']
        with NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            image.save(temp_file, format="PNG")
            temp_file_path = temp_file.name

        score = self.image_reward_score(images=[temp_file_path], texts=[prompt]).item()
        return score


class ProgrammaticDSGTIFAScore(Metric):
    def __init__(self, model='Phi-3-vision-128k-instruct', device="cuda"):
        super().__init__(device=device)
        from ..qa_model.imageqa_model import ImageQAModel
        from ..qa_model.prompt import succinct_prompt
        self.vqa_model = ImageQAModel(model_name=model, torch_device = self.device, prompt_name="succinct", prompt_func=succinct_prompt)
        
    @ staticmethod
    def _get_dsg_questions(dsg):
        dsg_questions = {}

        for node in dsg.nodes(data=True):
            node_id, node_data = node
            node_type = node_data['type']
            node_value = node_data['value']

            if node_type == 'object_node':
                dsg_questions[f"{node_id}:{node_value}"] = {}
                # preposition
                dsg_questions[f"{node_id}:{node_value}"]['question'] = f"Is there a {node_value}?"
                dsg_questions[f"{node_id}:{node_value}"]['dependency'] = []
                for neighbor_id in dsg.neighbors(node_id):
                    neighbor_data = dsg.nodes[neighbor_id]
                    neighbor_type = neighbor_data['type']
                    neighbor_value = neighbor_data['value']
                    if neighbor_type == 'attribute_node':
                        # combine color
                        dsg_questions[f"{neighbor_id}:{neighbor_value}"]={}
                        dsg_questions[f"{neighbor_id}:{neighbor_value}"]['question']= f"Is the {node_value} {neighbor_value}?"
                        dsg_questions[f"{neighbor_id}:{neighbor_value}"]['dependency']= [f"{node_id}:{node_value}"]

        for edge in dsg.edges(data=True):
            source_node, target_node, edge_data = edge
            edge_type = edge_data['type']

            if edge_type == 'relation_edge':
                edge_value = edge_data['value']
                source_node_value = dsg.nodes[source_node]['value']
                target_node_value = dsg.nodes[target_node]['value']
                dsg_questions[f"{edge_value}|{source_node}:{source_node_value}|{target_node}:{target_node_value}"] = {}
                dsg_questions[f"{edge_value}|{source_node}:{source_node_value}|{target_node}:{target_node_value}"]['question'] = f"Is the {source_node_value} {edge_value} the {target_node_value}?"
                dsg_questions[f"{edge_value}|{source_node}:{source_node_value}|{target_node}:{target_node_value}"]['dependency'] = [f"{source_node}:{source_node_value}", f"{target_node}:{target_node_value}"]
        return dsg_questions

    @ staticmethod
    def _compute_score_with_dependencies(dsg_questions):
        cnt, tnt = 0, 0.0
        for element in dsg_questions:
            cnt += 1
            if dsg_questions[element]['result'] is True:
                true_with_dependencies = True
                for dependent_object in dsg_questions[element]['dependency']:
                    if dsg_questions[dependent_object]['result'] is False:
                        true_with_dependencies = False
                        break
                if true_with_dependencies is True:
                    tnt += 1
        return tnt / cnt if cnt > 0 else 0.0
    
    @ staticmethod
    def _compute_score_without_dependencies(dsg_questions):
        cnt, tnt = 0, 0.0
        for element in dsg_questions:
            cnt += 1
            if dsg_questions[element]['result'] is True:
                tnt += 1
        return tnt / cnt if cnt > 0 else 0.0
        
    def compute(self, image: Image.Image, gen_data: dict):
        from ...captions_generation.prompt_generator import convert_json_to_sg
        # the scene graph in gen_data is json format, we need to convert it to nx.DiGraph format here
        scene_graph = convert_json_to_sg(gen_data['scene_graph'])
        dsg_questions = self._get_dsg_questions(scene_graph)
        
        for element in dsg_questions:
            prompt = "Based on the image, answer: " + dsg_questions[element]['question'] + ". Only output yes or no"
            print(prompt)
            model_answer  = self.vqa_model.qa(image, prompt)
            print(model_answer)
            dsg_questions[element]['result'] = "yes" in model_answer.lower()

        score_without_dependencies = self._compute_score_without_dependencies(dsg_questions)
        score_with_dependencies = self._compute_score_with_dependencies(dsg_questions)

        # return the weighted score
        return 0.5 * score_with_dependencies + 0.5 * score_without_dependencies

