"""
Run decoder depth ablation experiments across 3 GPUs.

Experiments: depth 12 (default), 16, 20, 24
GPUs: 0, 1, 2 — the 4th experiment waits for the first free GPU.

Usage:
    python run_experiments.py --ckpt 48k/last.ckpt
    python run_experiments.py --ckpt 48k/last.ckpt --dry_run
"""

import argparse
import subprocess
import os
import sys
import time

DEPTHS = [12, 16, 20, 24]
NUM_GPUS = 3


def extend_checkpoint(ckpt_path, target_depth):
    output_path = f"extended_{target_depth}.pt"
    if os.path.exists(output_path):
        print(f"[extend] {output_path} already exists, skipping")
        return output_path
    print(f"[extend] Creating extended checkpoint for depth {target_depth}...")
    subprocess.run(
        [sys.executable, "extend_decoder.py",
         "--ckpt", ckpt_path,
         "--target_depth", str(target_depth),
         "--output", output_path],
        check=True,
    )
    return output_path


def build_train_cmd(depth, gpu_id, ckpt_arg):
    log_dir = f"ablation_depth_{depth}"
    wandb_name = f"depth_{depth}"
    cmd = [
        sys.executable, "train.py",
        f"log_dir={log_dir}",
        f"wandb_name={wandb_name}",
        "wandb_project=neucodec_48k_ablation",
        "train.trainer.devices=1",
        f"model.codec_decoder.depth={depth}",
        "+train.accumulate_grad_batches=16",
        'dataset.train.filelist=real_train.txt',
        'dataset.val.filelist=real_test.txt',
        "dataset.train.batch_size=8",
    ]
    if ckpt_arg is not None:
        cmd.append(f"ckpt={ckpt_arg}")
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Base PL checkpoint path")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running")
    args = parser.parse_args()

    # Prepare extended checkpoints for non-default depths
    experiments = []
    for depth in DEPTHS:
        if depth == 12:
            ckpt_arg = None
            experiments.append((depth, ckpt_arg))
        else:
            if not args.dry_run:
                ckpt_arg = extend_checkpoint(args.ckpt, depth)
            else:
                ckpt_arg = f"extended_{depth}.pt"
            experiments.append((depth, ckpt_arg))

    # Launch experiments: 3 GPUs, 4 experiments
    # First 3 run in parallel, 4th waits for any to finish
    processes = {}  # gpu_id -> (process, depth)

    for i, (depth, ckpt_arg) in enumerate(experiments):
        if i >= NUM_GPUS:
            # Wait for any GPU to free up
            print(f"\n[scheduler] All GPUs busy, waiting for one to finish...")
            while True:
                for gpu_id, (proc, d) in list(processes.items()):
                    ret = proc.poll()
                    if ret is not None:
                        print(f"[scheduler] GPU {gpu_id} finished (depth={d}, exit={ret})")
                        del processes[gpu_id]
                        break
                if len(processes) < NUM_GPUS:
                    break
                time.sleep(10)

        # Find a free GPU
        gpu_id = None
        for g in range(NUM_GPUS):
            if g not in processes:
                gpu_id = g
                break

        cmd = build_train_cmd(depth, gpu_id, ckpt_arg)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        print(f"\n[launch] GPU {gpu_id} | depth={depth}")
        print(f"  CUDA_VISIBLE_DEVICES={gpu_id}")
        print(f"  {' '.join(cmd)}")

        if not args.dry_run:
            proc = subprocess.Popen(cmd, env=env)
            processes[gpu_id] = (proc, depth)
        else:
            print("  (dry run, skipping)")

    # Wait for remaining processes
    if not args.dry_run and processes:
        print(f"\n[scheduler] Waiting for {len(processes)} remaining experiments...")
        for gpu_id, (proc, depth) in processes.items():
            ret = proc.wait()
            print(f"[done] GPU {gpu_id} | depth={depth} | exit={ret}")

    print("\nAll experiments finished!")


if __name__ == "__main__":
    main()
