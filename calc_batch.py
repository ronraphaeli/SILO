import os
import sys
import json
import numpy as np
import torch
from datetime import datetime

def parse_log_file(file_path):
    with open(file_path, 'r') as file:
        log_data = json.load(file)
    return log_data['metrics']

def calculate_stats(metrics_list):
    aggregated_metrics = {}
    for metrics in metrics_list:
        for metric_name, value in metrics.items():
            if metric_name in aggregated_metrics:
                aggregated_metrics[metric_name].append(value)
            else:
                aggregated_metrics[metric_name] = [value]

    stats = {}
    for metric_name, values in aggregated_metrics.items():
        mean = np.mean(values)
        variance = np.var(values)
        stats[metric_name] = {'mean': mean, 'variance': variance}
    return stats

def write_log_file(output_file, original_log_file, stats, num_of_files,percp_string):
    with open(output_file, 'w') as file_out:
        file_out.write(f"date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        file_out.write(f"\n")
        file_out.write(f"{num_of_files = }")
        file_out.write(f"\n")
        with open(original_log_file, 'r') as file_in:
            file_out.write(file_in.read())  # Copy original log file content
            file_out.write("\n\n--- Calculated Stats ---\n")
            for metric_name, stat in stats.items():
                file_out.write(f"{metric_name} Mean: {stat['mean']:.4f}\n")
                file_out.write(f"{metric_name} Variance: {stat['variance']:.4f}\n")
                file_out.write(f"---------\n")
            
            if percp_string:
                file_out.write(f"---------\n")
                file_out.write(percp_string)



def calc_perception(folder_path):
    #imports
    import torch
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    from datasets import load_dataset
    import torch.nn.functional as F
    import torchvision.datasets as dset
    import torchvision.transforms as transforms
    from torchvision.transforms.functional import to_pil_image , to_tensor , normalize , resize
    from PIL import Image
    from datasets import Dataset
    import os
    from silo_utils import check_folder_for_1000_pngs

    device = "cuda"
    torch.manual_seed(123)
    fid = FrechetInceptionDistance(normalize=True).to(device)
    kid = KernelInceptionDistance(normalize=True).to(device)
    batch_size = 128
    resolution = 512
    calculate_on_1000 = True
    p = folder_path
    path_to_img_folder = p
    path_to_img_folder = os.path.join(path_to_img_folder,"images")
    if check_folder_for_1000_pngs(path_to_img_folder)[0] == False:
        #than that mean we dont have 1000 images
        return False
    #create the real dataset
    
    from datasets import load_dataset , load_from_disk
    #put here paths to the real datasets!!
    if "coco" in p:
        dataset = load_from_disk("path to coco")
    else:
        dataset = load_from_disk("path to ffhq")
        if calculate_on_1000:
            dataset = dataset.train_test_split(train_size=1000,shuffle=False)
            dataset = dataset["train"] #this is actually the test set. see the training code

    dataset_columns = ("image", "text")
    image_column = dataset_columns[0]

    train_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution) ,
            transforms.ToTensor(),
        ]
    )

    def preprocess_train(examples):
        examples["pixel_values"] = [train_transforms(image) for image in examples["image"]]

        return examples



    true_dataset = dataset.with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        return {"pixel_values": pixel_values}


    true_dataloader = torch.utils.data.DataLoader(
            true_dataset,
            shuffle=False,
            collate_fn=collate_fn,
            batch_size=batch_size,
        )

    for batch in true_dataloader:
        fid.update(batch['pixel_values'].to(device), real=True)
        kid.update(batch['pixel_values'].to(device), real=True)

    fake_dataset_dict = {"image":[]}
    png_files = [f for f in os.listdir(path_to_img_folder) if f.endswith('.png')]

    for file in png_files:
        with Image.open(os.path.join(path_to_img_folder, file)) as img:
            # Create a copy of the image in memory
            image_ = img.copy()
            fake_dataset_dict["image"].append(image_)

    fake_dataset = Dataset.from_dict(fake_dataset_dict)
    fake_dataset = fake_dataset.with_transform(preprocess_train)
    fake_dataloader = torch.utils.data.DataLoader(
            fake_dataset,
            shuffle=False,
            collate_fn=collate_fn,
            batch_size=batch_size,
        )

    for batch in fake_dataloader:
        fid.update(batch['pixel_values'].to(device), real=False)
        kid.update(batch['pixel_values'].to(device), real=False)

    fid_score = fid.compute()
    kid_score = kid.compute()
    string_to_return = f"{path_to_img_folder = }" + "\n"
    string_to_return += f"on {len(true_dataset)} true images, {len(fake_dataset)} fake images" + "\n"
    string_to_return += f"the FID is = {fid_score:.4f} the KID * 1e3 is = {kid_score[0].cpu().item() * 1e3:.4f}"
    return string_to_return


import os
import json
import glob
import numpy as np
from typing import List, Dict, Any, Tuple, Set


def check_experiment_logs(folder_path: str, arg_keys_to_check: List[str] = None) -> Dict[str, Any]:
    """
    Check experiment log files for completeness, consistency, and outliers.
    
    Args:
        folder_path: Path to the folder containing experiment log files
        arg_keys_to_check: List of argument keys to check for consistency (if None, check all)
        
    Returns:
        Dictionary with check results
    """
    # Initialize results dictionary
    results = {
        "missing_files": [],
        "inconsistent_args": {},
        "outliers": {},
        "all_files_present": False,
        "args_consistent": True,
    }
    
    # Check if folder exists
    if not os.path.isdir(folder_path):
        return {"error": f"Folder {folder_path} does not exist"}
    
    # Get all JSON files in the folder
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
    # Extract file numbers and check for range 0-999
    file_numbers = set()
    file_data = {}
    
    for file_path in json_files:
        file_name = os.path.basename(file_path)
        try:
            file_num = int(os.path.splitext(file_name)[0])
            if 0 <= file_num <= 999:
                file_numbers.add(file_num)
                
                # Read the JSON file
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    file_data[file_num] = data
        except ValueError:
            # Skip files that don't have numeric names
            continue
    
    # Check 1: Are all 1000 files present?
    expected_files = set(range(1000))
    missing_files = expected_files - file_numbers
    results["missing_files"] = sorted(list(missing_files))
    results["all_files_present"] = len(missing_files) == 0
    
    if not file_data:
        return {**results, "error": "No valid log files found"}
    
    # Check 2: Consistency of arguments
    if file_data:
        # Get reference arguments from the first file
        first_file_num = min(file_data.keys())
        reference_args = file_data[first_file_num]["args"]
        
        # If specific arg keys are not provided, check all keys
        if arg_keys_to_check is None:
            arg_keys_to_check = list(reference_args.keys())
        
        # Check consistency across all files
        for file_num, data in file_data.items():
            if "args" not in data:
                results["inconsistent_args"][file_num] = "Missing args section"
                results["args_consistent"] = False
                continue
                
            file_args = data["args"]
            for key in arg_keys_to_check:
                if key not in file_args:
                    if file_num not in results["inconsistent_args"]:
                        results["inconsistent_args"][file_num] = {}
                    results["inconsistent_args"][file_num][key] = "Missing key"
                    results["args_consistent"] = False
                    continue
                    
                if key in reference_args and file_args[key] != reference_args[key]:
                    if file_num not in results["inconsistent_args"]:
                        results["inconsistent_args"][file_num] = {}
                    results["inconsistent_args"][file_num][key] = {
                        "expected": reference_args[key],
                        "found": file_args[key]
                    }
                    results["args_consistent"] = False
    
    # Check 3: Outlier detection in metrics
    metrics_data = {}
    
    # Collect metrics from all files
    for file_num, data in file_data.items():
        if "metrics" in data:
            for metric_name, metric_value in data["metrics"].items():
                if metric_name not in metrics_data:
                    metrics_data[metric_name] = []
                metrics_data[metric_name].append((file_num, metric_value))
    
    # Detect outliers using Z-score method (considering values > 3 std dev as outliers)
    for metric_name, metric_values in metrics_data.items():
        file_nums = [item[0] for item in metric_values]
        values = np.array([item[1] for item in metric_values])
        
        if len(values) < 2:
            continue
            
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:  # Skip if all values are identical
            continue
            
        z_scores = np.abs((values - mean) / std)
        outlier_indices = np.where(z_scores > 3)[0]
        
        if len(outlier_indices) > 0:
            results["outliers"][metric_name] = {
                "mean": float(mean),
                "std": float(std),
                "outliers": []
            }
            
            for idx in outlier_indices:
                file_num = file_nums[idx]
                value = float(values[idx])
                z_score = float(z_scores[idx])
                results["outliers"][metric_name]["outliers"].append({
                    "file_num": file_num,
                    "value": value,
                    "z_score": z_score,
                    "type": "high" if values[idx] > mean else "low"
                })
    
    return results



def ensure_good_batch(folder_path,print_outliers=True):
    folder_path = os.path.join(folder_path,"logs")
    # Example usage:
    # folder_path = folder_path +"/logs"
    # List of argument keys we specifically want to check for consistency
    arg_keys = ["pretrained_model_name_or_path", "dataset_name", "seed", "noise_sigma", "sr_factor", "dataset_name", "prompt", "seed", "cfg",
                "scale", "steps", "ckpt_exp_name", "ckpt_step", "clamp","model", "dps_operator", "dps_noiser" ]

    results = check_experiment_logs(folder_path, arg_keys)

    # Print formatted results
    print("Experiment Log Check Results:")
    print("-" * 50)

    print(f"All files present (0-999): {'Yes' if results['all_files_present'] else 'No'}")
    if not results['all_files_present']:
        print(f"Missing {len(results['missing_files'])} files:", end=" ")
        # Print first 10 missing files with ellipsis if more
        if len(results['missing_files']) > 10:
            print(f"{results['missing_files'][:10]} ... and {len(results['missing_files'])-10} more")
        else:
            print(results['missing_files'])
    
    print(f"\nArguments consistent: {'Yes' if results['args_consistent'] else 'No'}")
    if not results['args_consistent']:
        print(f"Found {len(results['inconsistent_args'])} files with inconsistent arguments")
        # Show a few examples
        for i, (file_num, inconsistencies) in enumerate(results['inconsistent_args'].items()):
            if i >= 3:  # Show only first 3 examples
                print(f"... and {len(results['inconsistent_args'])-3} more files with inconsistencies")
                break
            print(f"  File {file_num}: {inconsistencies}")
        print([k for k in results['inconsistent_args']])
        raise RuntimeError("args not consistent !!!")
    if print_outliers:
        print("\nOutlier detection:")
        if not results['outliers']:
            print("  No outliers detected")
        else:
            for metric_name, outlier_info in results['outliers'].items():
                print(f"  Metric: {metric_name}")
                print(f"    Mean: {outlier_info['mean']:.4f}, Std: {outlier_info['std']:.4f}")
                print(f"    Found {len(outlier_info['outliers'])} outliers:")
                for i, outlier in enumerate(outlier_info['outliers']):
                    if i >= 5:  # Show only first 5 outliers
                        print(f"    ... and {len(outlier_info['outliers'])-5} more outliers")
                        break
                    print(f"      File {outlier['file_num']}: {outlier['value']:.4f} "
                            f"(z-score: {outlier['z_score']:.2f}, {outlier['type']} outlier)")

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    logs_path = os.path.join(folder_path, "logs")

    if not os.path.isdir(logs_path):
        print(f"Logs directory not found at {logs_path}")
        sys.exit(1)

    log_files = [os.path.join(logs_path, f) for f in os.listdir(logs_path) if f.endswith(".json")]
    num_of_files = len(log_files)
    if not log_files:
        print(f"No log files found in {logs_path}")
        sys.exit(1)
    
    ensure_good_batch(folder_path)

    metrics_list = [parse_log_file(log_file) for log_file in log_files]
    stats = calculate_stats(metrics_list)

    output_log_file = os.path.join(folder_path, "summary.json")
    # Use the first log file to copy its contents
    if torch.cuda.is_available():
        percp_string = calc_perception(folder_path)
    else:
        percp_string = False
    write_log_file(output_log_file, log_files[0], stats, num_of_files,percp_string)

    print(f"Summary log file created at {output_log_file}")
    print(f"{num_of_files = }")
if __name__ == "__main__":
    main()
