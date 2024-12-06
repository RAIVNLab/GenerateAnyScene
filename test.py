import os

# import networkx as nx
# from gas.models.gen_model import text2image_model
# from gas.text2vision.prompt_generator import *
# from gas.text2vision.metadata import Text2ImageMetaData
# os.chdir("/home/murphy/data-ssd/weikai/genverse/development/TaskMeAnything")
# os.environ["TRANSFORMERS_CACHE"] = "/home/murphy/data-hdd/videoqa"
# os.environ["TORCH_CACHE"] = "/home/murphy/data-hdd/videoqa"
# os.environ["HF_HOME"] = "/home/murphy/data-hdd/videoqa"
# os.environ["HF_HUB_CACHE"] = "/home/murphy/data-hdd/videoqa"
# os.environ["HF_INFERENCE_ENDPOINT"] = "/home/murphy/data-hdd/videoqa"

# # metadata = Text2ImageMetaData(path_to_metadata = "/home/murphy/data-ssd/weikai/genverse/development/TaskMeAnything/metadata",
# #                               path_to_sg_template= "/home/murphy/data-ssd/weikai/genverse/data_generation_and_processing/sg_templates")
# metadata = Text2ImageMetaData(path_to_metadata = "/home/murphy/data-ssd/weikai/genverse/development/TaskMeAnything/metadata")
# generator = Text2ImagePromptGenerator(metadata = metadata)


# # building the seed graph and generating prompt
# seed_graph = nx.DiGraph()

# seed_graph.add_node(0, type="object_node", value="dog")
# seed_graph.add_node(1, type="object_node", value="rabbit")
# seed_graph.add_node(2, type="attribute_node", value="black")
# seed_graph.add_edge(0, 1, type="relation_edge", value = "chasing after")
# seed_graph.add_edge(0, 2, type="attribute_edge")

# task_plans = generator.sample_task_plans(seed_graph=seed_graph, number_of_global_attributes=1, complexity=6, sample_numbers=10)
# prompt = generator.generate(task_plans[0])
from gas.models.gen_model import text2video_model
gen_data = {
        "prompt": "A painter (skilled worker); a landscaping; a pointed-leaf maple.",
        "global_attributes": {},
        "scene_graph": {
            "nodes": [
                [
                    "object_1",
                    {
                        "type": "object_node",
                        "value": "painter (skilled worker)"
                    }
                ],
                [
                    "object_2",
                    {
                        "type": "object_node",
                        "value": "landscaping"
                    }
                ],
                [
                    "object_3",
                    {
                        "type": "object_node",
                        "value": "pointed-leaf maple"
                    }
                ]
            ],
            "edges": []
        }
}
models_list = [
    "animateLCM",
    # "stable-diffusion-3",
    # "stable-diffusion-2-1",
    # "stable-diffusion-xl"
]
print(gen_data["prompt"])
for model_name in models_list:
    model = text2video_model.Text2VideoModel(
        model_name = model_name,
        metrics = ["ProgrammaticDSGTIFAScore","ClipScore","VQAScore"],
        metrics_device = 1,
        torch_device = 0
    )
    result = model.gen(gen_data)

print(result)


# # ablated eval function
# from  gas.models.gen_model import text2image_metric
# im = text2image_metric.Text2ImageEvalMetric()
# result = im.eval_with_metrics(prompt, result['output'])
# print(result)


