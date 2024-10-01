import os
import json
import shutil
from jsonargparse import CLI

def merge_and_move_bin_files(source_dir: str, target_dir: str):
    """Move the bin files within different subdirectories in the source directory to a single target directory
    and create a single metadata file (index.json)

    Args:
        source_dir: Path to the directory containing bin files (Streaming dataset)
        target_dir: Target path where all the bin files will be moved.
    """
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # This dictionary will aggregate all chunk data from various index.json files
    aggregate_data = {
        "chunks": [],
        "config": {}
    }

    # Loop through each subdirectory in the source directory
    for subdir in os.listdir(source_dir):
        subdir_path = os.path.join(source_dir, subdir)
        print(subdir_path)
        
        if os.path.isdir(subdir_path):
            # Path to the index.json file in the subdirectory
            index_path = os.path.join(subdir_path, "index.json")
            
            if os.path.exists(index_path):
                # Read the index.json file
                with open(index_path, 'r') as file:
                    index_data = json.load(file)

                # Add the chunks to the aggregate data
                # Loop through each chunk, update filename, and move file
                for chunk in index_data.get('chunks', []):
                    original_filename = chunk.get('filename')
                    new_filename = f"chunk-{subdir}-{original_filename.split('-')[1]}-{original_filename.split('-')[2]}"
                    source_file_path = os.path.join(subdir_path, original_filename)
                    target_file_path = os.path.join(target_dir, new_filename)

                    # Move the file, avoiding overwrites since filenames are now unique
                    if not os.path.exists(target_file_path):
                        shutil.move(source_file_path, target_file_path)
                    else:
                        # Handling unexpected cases where new filename might still conflict
                        base_name, ext = os.path.splitext(new_filename)
                        counter = 1
                        new_target_file_path = os.path.join(target_dir, f"{base_name}_{counter}{ext}")
                        while os.path.exists(new_target_file_path):
                            counter += 1
                            new_target_file_path = os.path.join(target_dir, f"{base_name}_{counter}{ext}")
                        shutil.move(source_file_path, new_target_file_path)
                        new_filename = f"{base_name}_{counter}{ext}"

                    # Update the filename in the aggregated index.json
                    chunk['filename'] = new_filename
                    aggregate_data['chunks'].append(chunk)

                # Optionally update the config, assuming it's the same for all or taking the last one
                aggregate_data['config'] = index_data.get('config', {})

    # Write the aggregated index.json to the target directory
    with open(os.path.join(target_dir, 'index.json'), 'w') as file:
        json.dump(aggregate_data, file, indent=4)

# Example usage:
# source_directory = "/home/ray/efs/cluster/test-runs/train"
# target_directory = "/home/ray/efs/cluster/data/optimized_sample/train"
# merge_and_move_bin_files(source_directory, target_directory)

if __name__ == "__main__":
    CLI(merge_and_move_bin_files, as_positional=False)