import json
import os
import random
import argparse
import multiprocessing as mp
from tqdm import tqdm
from gas.captions_generation.prompt_generator import Text2ImagePromptGenerator, Text2VideoPromptGenerator, Text2ThreeDPromptGenerator
from gas.captions_generation.metadata import Text2ImageMetaData, Text2VideoMetaData, Text2ThreeDMetaData

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate text-to-vision captions using multiprocessing.")
    parser.add_argument("--metadata_path", type=str, default="./metadata", help="Path to the metadata file.")
    parser.add_argument("--output_dir", type=str, default="./", help="Directory to save the generated captions.")
    parser.add_argument("--total_prompts", type=int, default=5, help="Total number of captions to generate.")
    parser.add_argument("--num_workers", type=int, default=1, help="Numbers of workers (enables parallelism).")
    parser.add_argument("--min_complexity", type=int, default=3, help="Minimum complexity of captions.")
    parser.add_argument("--max_complexity", type=int, default=8, help="Maximum complexity of captions.")
    parser.add_argument("--min_attributes", type=int, default=0, help="Minimum number of scene attributes.")
    parser.add_argument("--max_attributes", type=int, default=5, help="Maximum number of scene attributes.")
    parser.add_argument("--modality_type", type=str, default="text2image", help="Type of modality to generate captions for.")
    return parser.parse_args()

def create_metadata(modality_type, metadata_path):
    if modality_type == "text2image": 
        return Text2ImageMetaData(path_to_metadata=metadata_path)
    elif modality_type == "text2video":
        return Text2VideoMetaData(path_to_metadata=metadata_path)
    elif modality_type == "text2threed":
        return Text2ThreeDMetaData(path_to_metadata=metadata_path)
    else:
        raise ValueError(f"Unsupported modality type: {modality_type}")

def generate_prompt(generator, complexity, num_scene_attributes):
    task_plans = generator.sample_task_plans(number_of_scene_attributes=num_scene_attributes, complexity=complexity, sample_numbers=1)
    sg = task_plans[0]
    return generator.generate(sg)

def generate_batch(batch_idx, complexities, scene_attributes, prompts_per_attribute, metadata_path, seed, output_dir, modality_type):
    prompts_list = []
    metadata = create_metadata(modality_type, metadata_path)
    if modality_type == "text2image":
        generator = Text2ImagePromptGenerator(metadata=metadata, seed=seed)
    elif modality_type == "text2video":
        generator = Text2VideoPromptGenerator(metadata=metadata, seed=seed)
    elif modality_type == "text2threed":
        generator = Text2ThreeDPromptGenerator(metadata=metadata, seed=seed)

    for complexity in tqdm(complexities, desc=f"Batch {batch_idx} - Complexity"):
        for num_attributes in scene_attributes:
            for _ in range(prompts_per_attribute):
                prompts_list.append(generate_prompt(generator, complexity, num_attributes))

    prompts_dict = {idx: prompt for idx, prompt in enumerate(prompts_list)}
    try:
        file_name = f"{output_dir}/prompts_batch_{batch_idx}.json"
        with open(file_name, "w") as f:
            json.dump(prompts_dict, f, indent=4)
        print(f"Saved {len(prompts_list)} prompts to {file_name}")
    except Exception as e:
        print(f"Error saving prompts for batch {batch_idx}: {e}")

def main():
    args = parse_arguments()
    if args.num_workers > args.total_prompts:
        raise ValueError("Number of files cannot exceed total prompts.")
    os.makedirs(args.output_dir, exist_ok=True)
    complexities = range(args.min_complexity, args.max_complexity + 1)
    scene_attributes = range(args.min_attributes, args.max_attributes + 1)
    prompts_per_file = args.total_prompts // args.num_workers
    prompts_per_complexity = prompts_per_file // len(complexities)
    prompts_per_attribute = prompts_per_complexity // len(scene_attributes)
    seeds = [random.randint(0, 100) for _ in range(args.num_workers)]

    with mp.Pool(processes=args.num_workers) as pool:
        pool.starmap(generate_batch, [
            (batch_idx, complexities, scene_attributes, prompts_per_attribute, args.metadata_path, seeds[batch_idx], args.output_dir, args.modality_type)
            for batch_idx in range(args.num_workers)
        ])

if __name__ == "__main__":
    main()