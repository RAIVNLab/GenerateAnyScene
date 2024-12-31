import argparse
import json
import os
from gas.models.gen_model import text2image_model, text2video_model, textto3d_model  
from gas.models.gen_model import text2image_metric, text2video_metric, textto3d_metric

def main(input_file, gen_type, models_list, metrics_list, output_dir="./generated_output"):
    os.makedirs(output_dir, exist_ok=True)
    # Load JSON file
    with open(input_file, 'r') as f:
        captions = json.load(f)

    # Extract the first caption
    gen_data = captions[next(iter(captions))]
    
    print(f"Using the following prompt for generation: {gen_data['prompt']}")

    # Perform generation based on the type
    for model_name in models_list:
        if gen_type == "image":
            model = text2image_model.Text2ImageModel(
                model_name=model_name,
                metrics=metrics_list,
                metrics_device="cuda",  # Specify device for metrics computation, "cuda" or int value (0, 1, etc.)
                torch_device="cuda"      # Specify device for model computation, "cuda" or int value (0, 1, etc.)
            )
        elif gen_type == "video":
            model = text2video_model.Text2VideoModel(
                model_name=model_name,
                metrics=metrics_list,
                metrics_device="cuda",  
                torch_device="cuda"      
            )
        elif gen_type == "3d":
            # Uncomment if 3D support is needed
            model = textto3d_model.Textto3DModel(
                model_name=model_name,
                metrics=metrics_list,
                metrics_device="cuda",  
                torch_device="cuda"
            )
        else:
            raise ValueError(f"Unsupported generation type: {gen_type}")

        print(f"Generating with model: {model_name}")
        result = model.gen(gen_data)
        
        # Save the generated output
        if gen_type == "image":
            output_file = f"{output_dir}/image_{gen_data['prompt'].replace(' ', '_').replace('/','_')}_{model_name}.png"
            result["output"].save(output_file)
        elif gen_type == "video" or gen_type == "3d":
            output_file = f"{output_dir}/video_{gen_data['prompt'].replace(' ', '_').replace('/','_')}_{model_name}.mp4"
            with open(output_file, "wb") as f:
                f.write(result["output"])
        print(f"Saved output to {output_file}")
        print(f"Metrics: {result['metrics']}")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo for generating images/videos/3D scenes using captions from a JSON file.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the JSON file containing captions.")
    parser.add_argument("--gen_type", type=str, choices=["image", "video", "3d"], required=True, help="Type of generation: image, video, or 3d.")
    parser.add_argument("--models", nargs='+', help="List of models to use for generation.")
    parser.add_argument("--metrics", nargs='+', help="List of metrics to compute for the generated output.")
    parser.add_argument("--output_dir", type=str, default="./generated_output", help="Directory to save the generated output.")
    args = parser.parse_args()
    if args.models is None:
        if args.gen_type == "3d":
            args.models = textto3d_model.list_textto3d_models()
        elif args.gen_type == "video":
            args.models = text2video_model.list_text2video_models()
        else:
            args.models = text2image_model.list_text2image_models()
    if args.metrics is None:
        args.metrics = text2image_metric.list_image_metrics()

    main(args.input_file, args.gen_type, args.models, args.metrics)
