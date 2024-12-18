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


# ablated eval function
from  gas.models.gen_model import text2image_metric
im = text2image_metric.Text2ImageEvalMetric()
result = im.eval_with_metrics(gen_data["prompt"], result['output'])
print(result)


