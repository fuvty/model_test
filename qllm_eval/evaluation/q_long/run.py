import os
import re
import subprocess
import tempfile
from threading import Thread
from time import sleep

# Base settings
model_path = "/share/datasets/public_models/lmsys_vicuna-7b-v1.5-16k"
task = "lines"
block_size = 64
lut_base_path = "/share/futianyu/repo/NLP-playground/local/universal/test-model/profile_test"
mem_threshold_mb = 30 * 1024 # the least memory to run the model
gpu_pool = ["0", "1", "2", "3", "4", "5", "6", "7"]
max_num_jobs = 8
summary_file = "test_summary.txt"
gpu_in_use = {gpu: False for gpu in gpu_pool}

def get_best_gpu(mem_threshold_mb=40*1024):
    """
    Gets the ID of the best GPU based on the free memory available.
    :param mem_threshold_mb: Memory threshold in MiB. Only GPUs with free memory above this value are considered.
    :return: The GPU ID as a string.
    """
    # Execute nvidia-smi to get GPU memory usage
    nvidia_smi_output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits']).decode()

    # Initialize best GPU selection variables
    best_gpu_id = None
    max_free_memory = mem_threshold_mb  # Start with the threshold to ensure we select a GPU with enough free memory

    # Parse the output
    for line in nvidia_smi_output.strip().split('\n'):
        gpu_id, free_memory = line.split(', ')
        free_memory = int(free_memory)  # Convert free memory from string to integer

        # Check if this GPU is in the pool and has more free memory than the current best, and is not currently in use
        print("GPU usage: ", gpu_id, free_memory, gpu_in_use[gpu_id])
        if gpu_id in gpu_pool and free_memory > max_free_memory and not gpu_in_use[gpu_id]:
            best_gpu_id = gpu_id
            max_free_memory = free_memory

    # If a suitable GPU is found, mark it as in use and return its ID
    if best_gpu_id is not None:
        gpu_in_use[best_gpu_id] = True
        return best_gpu_id
    else:
        # If no suitable GPU is found, wait and try again
        print("Waiting for a suitable GPU to become available in the pool...")
        sleep(60)
        # release the gpu with the least memory now
        free_memory_list = []
        nvidia_smi_output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits']).decode()
        for line in nvidia_smi_output.strip().split('\n'):
            gpu_id, free_memory = line.split(', ')
            free_memory = int(free_memory)
            free_memory_list.append((gpu_id, free_memory))
        free_memory_list.sort(key=lambda x: x[1])
        gpu_id_to_free, max_free_memory = free_memory_list[-1]
        if max_free_memory > mem_threshold_mb:
            print(f"GPU {gpu_id_to_free} has the least free memory ({max_free_memory} MiB), releasing it")
            release_gpu(gpu_id_to_free)
        return get_best_gpu(mem_threshold_mb)


def release_gpu(gpu_id):
    gpu_in_use[gpu_id] = False

def run_command(cmd, tmp_file):
    with open(tmp_file, 'w') as f:
        subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
    with open(tmp_file, 'r') as f:
        last_line = f.readlines()[-1].strip()
    return cmd, last_line

def run_tests():
    threads = []
    tmp_files = []
    results = []
    best_gpus = []

    for lut_file in os.listdir(lut_base_path):
        if match := re.match(r'lut_(\d+)_plan_(\d+)\.pt', lut_file):
            token_count = match.group(1)
            test_dir = {
                "2048": "2k_cases",
                "4096": "4k_cases",
                "8192": "8k_cases",
                "16384": "15k_cases"
            }.get(token_count, None)

            num_lines = {
                "2048": 70,
                "4096": 165,
                "8192": 320,
                "16384": 650
            }.get(token_count, None)

            if test_dir is None or num_lines is None:
                print(f"Unknown token count: {token_count} for {lut_file}")
                continue

            best_gpu = get_best_gpu()
            best_gpus.append(best_gpu)
            gpu_in_use[best_gpu] = True

            cmd = f'CUDA_VISIBLE_DEVICES={best_gpu} python main_longeval_lut.py --model-name-or-path "{model_path}" --task {task} --test_dir {test_dir} --num_lines {num_lines} --block_size {block_size} --lut_path {os.path.join(lut_base_path, lut_file)}'
            print(f"Running command: {cmd}")

            tmp_file = tempfile.mktemp()
            tmp_files.append(tmp_file)
            
            thread = Thread(target=lambda q, arg1, arg2: q.append(run_command(arg1, arg2)), args=(results, cmd, tmp_file))
            threads.append(thread)
            thread.start()

            if len(threads) == max_num_jobs:
                print("Waiting for threads to finish...")
                for thread in threads:
                    thread.join()
                threads = []
                for tmp_file in tmp_files:
                    os.remove(tmp_file)
                tmp_files = []
                for best_gpu in best_gpus:
                    release_gpu(best_gpu)
                best_gpus = []
                print(results)

    # Wait for any remaining threads
    for thread in threads:
        thread.join()

    # Write results to the summary file
    with open(summary_file, 'w') as f:
        f.write("Test Summary:\n")
        for cmd, last_line in results:
            f.write(f"Command: {cmd}\nResult: {last_line}\n-------\n")

    # Cleanup
    for tmp_file in tmp_files:
        os.remove(tmp_file)

if __name__ == "__main__":
    run_tests()
