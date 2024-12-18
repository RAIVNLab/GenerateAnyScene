# a multi-processing script to generate a large number of prompts

import json
import random
import argparse
import multiprocessing as mp
from tqdm import tqdm
from gas.text2vision.prompt_generator import Text2ImagePromptGenerator, Text2VideoPromptGenerator, Text2ThreeDScenePromptGenerator, Text2ThreeDObjectPromptGenerator
from gas.text2vision.metadata import Text2ImageMetaData, Text2VideoMetaData, Text2ThreeDMetaData


def parse_arguments():
    """
    Parse command-line arguments to configure the script.
    """
    parser = argparse.ArgumentParser(description="Generate text-to-vision prompts using multiprocessing.")
    parser.add_argument("--metadata_path", type=str, default="./metadata",
                        help="Path to the metadata file.")
    parser.add_argument("--output_dir", type=str, default="./",
                        help="Directory to save the generated prompts.")
    parser.add_argument("--total_prompts", type=int, default=5,
                        help="Total number of prompts to generate.")
    parser.add_argument("--num_files", type=int, default=1,
                        help="Number of output files (enables parallelism).")
    parser.add_argument("--min_complexity", type=int, default=3,
                        help="Minimum complexity of prompts.")
    parser.add_argument("--max_complexity", type=int, default=8,
                        help="Maximum complexity of prompts.")
    parser.add_argument("--min_attributes", type=int, default=0,
                        help="Minimum number of scene attributes.")
    parser.add_argument("--max_attributes", type=int, default=5,
                        help="Maximum number of scene attributes.")
    parser.add_argument("--modality_type", type=str, default="text2image",
                        help="Type of modality to generate prompts for.")
    return parser.parse_args()


def generate_prompt(generator, complexity, num_global_attributes):
    """
    Generate a single prompt using the provided generator.

    Args:
        generator (Text2ImagePromptGenerator): The prompt generator.
        complexity (int): Complexity level of the prompt.
        num_global_attributes (int): Number of global scene attributes.

    Returns:
        str: Generated prompt.
    """
    task_plans = generator.sample_task_plans(number_of_global_attributes=num_global_attributes, complexity=complexity, sample_numbers=1)
    sg = task_plans[0]
    return generator.generate(sg)


def generate_batch(batch_idx, complexities, scene_attributes, prompts_per_attribute, metadata, seed, output_dir):
    """
    Generate a batch of prompts and save them to a file.

    Args:
        batch_idx (int): Batch index for parallel processing.
        complexities (range): Range of complexities for prompts.
        scene_attributes (range): Range of numbers of scene attributes.
        prompts_per_attribute (int): Number of prompts to generate per complexity-attribute pair.
        metadata (Text2ImageMetaData): Metadata for the prompt generator.
        seed (int): Seed for randomness.
        output_dir (str): Directory to save the output files.
    """
    prompts_list = []
    generator = Text2ImagePromptGenerator(metadata=metadata, seed=seed)

    # Generate prompts for each complexity and attribute count
    for complexity in tqdm(complexities, desc=f"Batch {batch_idx} - Complexity"):
        for num_attributes in scene_attributes:
            for _ in range(prompts_per_attribute):
                prompts_list.append(generate_prompt(generator, complexity, num_attributes))

    # Save the generated prompts to a file
    prompts_dict = {idx: prompt for idx, prompt in enumerate(prompts_list)}
    file_name = f"{output_dir}/prompts_batch_{batch_idx}.json"
    with open(file_name, "w") as f:
        json.dump(prompts_dict, f, indent=4)
    print(f"Saved {len(prompts_list)} prompts to {file_name}")


def main():
    """
    Main function to configure and execute prompt generation.
    """
    args = parse_arguments()

    # Setup metadata and other parameters
    metadata = Text2ImageMetaData(path_to_metadata=args.metadata_path)
    complexities = range(args.min_complexity, args.max_complexity + 1)
    scene_attributes = range(args.min_attributes, args.max_attributes + 1)
    prompts_per_file = args.total_prompts // args.num_files
    prompts_per_complexity = prompts_per_file // len(complexities)
    prompts_per_attribute = prompts_per_complexity // len(scene_attributes)

    # Seed for reproducibility
    if args.num_files > 1:
        seeds = [random.randint(0, 100) for _ in range(args.num_files)]
    else:
        seeds = [42]  # Default seed for single process generation

    print(f"Generating {args.total_prompts} prompts across {args.num_files} file(s)")
    print(f"Complexities: {list(complexities)}")
    print(f"Scene Attributes: {list(scene_attributes)}")

    # Use multiprocessing to generate batches in parallel
    with mp.Pool(processes=args.num_files) as pool:
        pool.starmap(generate_batch, [
            (batch_idx, complexities, scene_attributes, prompts_per_attribute, metadata, seeds[batch_idx], args.output_dir)
            for batch_idx in range(args.num_files)
        ])


if __name__ == "__main__":
    main()