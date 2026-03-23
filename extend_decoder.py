"""
Extend decoder transformer layers by replicating the last layer.

Usage:
    python extend_decoder.py --ckpt 48k/last.ckpt --target_depth 24 --output extended_24.pt

This loads a PL checkpoint, strips the 'model.' prefix to get NeuCodec-level
state dict, replicates the last decoder transformer layer to reach target depth,
and saves a .pt file loadable via NeuCodec._from_pretrained(local_ckpt_path=...).

To train with the extended checkpoint:
    python train.py model.codec_decoder.depth=24 ckpt=extended_24.pt
"""

import argparse
import torch


def main():
    parser = argparse.ArgumentParser(description='Extend decoder transformer layers')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to PL checkpoint (.ckpt)')
    parser.add_argument('--target_depth', type=int, required=True, help='Target number of transformer layers')
    parser.add_argument('--output', type=str, required=True, help='Output path (.pt)')
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cpu')
    pl_state_dict = ckpt['state_dict']

    # Strip 'model.' prefix to get NeuCodec-level keys
    # PL wraps as model.generator.backbone.transformers.* -> generator.backbone.transformers.*
    model_state_dict = {}
    for key, value in pl_state_dict.items():
        if key.startswith('model.'):
            model_state_dict[key[len('model.'):]] = value

    # Find current decoder depth
    prefix = 'generator.backbone.transformers.'
    layer_indices = set()
    for key in model_state_dict:
        if key.startswith(prefix):
            idx = int(key[len(prefix):].split('.')[0])
            layer_indices.add(idx)

    current_depth = max(layer_indices) + 1
    print(f"Current decoder depth: {current_depth}")
    print(f"Target decoder depth: {args.target_depth}")

    if args.target_depth <= current_depth:
        print(f"Target depth {args.target_depth} <= current depth {current_depth}, nothing to do.")
        return

    # Get the last layer's weights
    last_idx = current_depth - 1
    last_layer_keys = {
        key: model_state_dict[key]
        for key in model_state_dict
        if key.startswith(f'{prefix}{last_idx}.')
    }
    print(f"Last layer has {len(last_layer_keys)} parameters")

    # Replicate the last layer to fill up to target_depth
    for new_idx in range(current_depth, args.target_depth):
        for key, value in last_layer_keys.items():
            new_key = key.replace(f'{prefix}{last_idx}.', f'{prefix}{new_idx}.')
            model_state_dict[new_key] = value.clone()
        print(f"  Replicated layer {last_idx} -> {new_idx}")

    print(f"Saving to: {args.output}")
    torch.save(model_state_dict, args.output)
    print(f"Done! {len(model_state_dict)} keys (was {len(model_state_dict) - (args.target_depth - current_depth) * len(last_layer_keys)})")
    print(f"\nTo train:")
    print(f"  python train.py model.codec_decoder.depth={args.target_depth} ckpt={args.output}")


if __name__ == '__main__':
    main()
