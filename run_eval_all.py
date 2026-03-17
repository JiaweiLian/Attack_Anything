import os
import subprocess
import re
import csv
import sys
from datetime import datetime

# ==============================================================
# Configuration
# ==============================================================
# Test models to evaluate against
TEST_MODELS = ["ssd","faster_rcnn","swin","yolov5x","cascade_rcnn","retinanet","mask_rcnn","free_anchor","fsaf","reppoints","tood","atss","foveabox"]
# TEST_MODELS = ["foveabox"]

# Attack methods you want to evaluate
# ATTACK_MODES = ["clean", "tba", "bba", "ccba"]
ATTACK_MODES = ["tba"]

# Surrogate models used to train the adversarial patch
# TRAIN_MODELS = ["faster_rcnn", "yolov5x"] 
TRAIN_MODELS = ["yolov5x_tood_tv_laplacian"] 

# Define corresponding patch paths for each attack mode (must have the same length as ATTACK_MODES)
# Note: Use `{train_model}` placeholder where the training model name should be inserted.
# Use `{attack_mode}` placeholder where the attack mode name should be inserted.
# Use `{test_model}` if you need the test model name (for white-box tests).
ATTACK_PATCH_PATHS = [
    "patches/patch_NN_response/{attack_mode}_{train_model}.png"
]

# Log file path
OUTPUT_CSV = "logs/eval_results.csv"

# Function to get the path to the adversarial patch for a specific model and attack.
# Modify this path rule according to your actual folder structure.
def get_patch_path(test_model, train_model, attack_mode, path_template):
    if attack_mode == "clean" or path_template is None or path_template.lower() == "none":
        return "None"
    
    # Format the path string replacing placeholders with actual values
    patch_path = path_template.format(test_model=test_model, train_model=train_model, attack_mode=attack_mode)
    return patch_path

def main():
    print(f"--- Automated Evaluation Script ---")
    print(f"Test Models: {TEST_MODELS}")
    print(f"Attacks: {ATTACK_MODES}")
    print(f"Train Models: {TRAIN_MODELS}")
    
    if len(ATTACK_MODES) != len(ATTACK_PATCH_PATHS):
        print("Error: ATTACK_MODES and ATTACK_PATCH_PATHS must have the same length!")
        return
        
    # Prepare CSV file
    file_exists = os.path.isfile(OUTPUT_CSV)
    with open(OUTPUT_CSV, mode="a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["Timestamp", "Test_Model", "Train_Model", "Attack_Mode", "Patch_Path", "AP@0.5:0.95", "AP@0.5"])
        
        # Calculate total evaluations to show progress
        total_evals = len(TEST_MODELS) * len(TRAIN_MODELS) * len(ATTACK_MODES)
        current_eval = 0

        for test_model in TEST_MODELS:
            for train_model in TRAIN_MODELS:
                for i, attack_mode in enumerate(ATTACK_MODES):
                    current_eval += 1
                    path_template = ATTACK_PATCH_PATHS[i]
                    patch_path = get_patch_path(test_model, train_model, attack_mode, path_template)
                    
                    progress_perc = (current_eval / total_evals) * 100
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] [{current_eval}/{total_evals} - {progress_perc:.1f}%] Evaluating | Test Model: {test_model} | Train Model: {train_model} | Attack: {attack_mode}")
                
                    # Construct command
                    # Notice we assume evaluate.py is in the parent directory of scripts/ a.k.a Attack_Anything
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    parent_dir = os.path.dirname(current_dir)
                    
                    cmd = ["python", "evaluate.py", "--model", test_model, "--attack_mode", attack_mode]
                    if attack_mode != "clean" and patch_path != "None":
                        cmd.extend(["--patch_path", patch_path])
                        
                    print("Running command:", " ".join(cmd))
                    
                    # Run evaluation script and capture output in real-time
                    try:
                        # use Popen to stream stdout line by line
                        process = subprocess.Popen(
                            cmd,
                            cwd=parent_dir,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            universal_newlines=True,
                            bufsize=1
                        )
                        
                        output_lines = []
                        # Print stdout in real-time to show inner progress (tqdm)
                        for line in process.stdout:
                            sys.stdout.write(line)
                            sys.stdout.flush()
                            output_lines.append(line)
                            
                        process.wait()
                        output = "".join(output_lines)
                        
                        # Parse mAP from output
                        ap_50_95 = None
                        ap_50 = None
                        
                        # Look for lines like:
                        # AP@0.5:0.95 = 0.1234
                        # AP@0.5      = 0.5678  <-- Target Metric
                        match_50_95 = re.search(r"AP@0.5:0.95\s*=\s*([\d\.]+)", output)
                        match_50 = re.search(r"AP@0.5\s*=\s*([\d\.]+)", output)
                        
                        if match_50_95:
                            ap_50_95 = match_50_95.group(1)
                        if match_50:
                            ap_50 = match_50.group(1)
                            
                        if ap_50 is None:
                            print(f"  [!] Failed to extract AP values. Check if errors occurred during evaluation.")
                            # 只打印最后几行错误信息，避免刷屏
                            print(f"  [!] Output snippet:\n{output[-500:]}\n")
                            ap_50_95, ap_50 = "Error", "Error"
                        else:
                            print(f"  --> AP@0.5:0.95: {ap_50_95}")
                            print(f"  --> AP@0.5     : {ap_50}")
                            
                        # Write record to CSV
                        writer.writerow([
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            test_model,
                            train_model,
                            attack_mode,
                            patch_path,
                            ap_50_95,
                            ap_50
                        ])
                        # Flush right away to secure data
                        csvfile.flush() 
                        
                    except Exception as e:
                        print(f"  [X] Failed to run evaluation for test_model: {test_model}, train_model: {train_model}, attack: {attack_mode}: {e}")

    print(f"\n--- Evaluation Completed! Results saved to {OUTPUT_CSV} ---")

if __name__ == "__main__":
    main()
