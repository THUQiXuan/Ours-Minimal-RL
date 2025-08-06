#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import warnings
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union

def parse_args():
    parser = argparse.ArgumentParser(description='Convert model to Hugging Face format')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the merged model (.pt file or directory containing model files)')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output path for the HF model')
    parser.add_argument('--base_model_name', type=str, required=True,
                        help='Base model name, used to get configuration and tokenizer (e.g.: Qwen/Qwen2.5-Math-7B)')
    parser.add_argument('--model_key', type=str, default='model',
                        help='Key to access model weights if model is a dictionary')
    parser.add_argument('--add_prefix', type=str, default=None,
                        help='Prefix to add to all parameter names')
    parser.add_argument('--remove_prefix', type=str, default=None,
                        help='Prefix to remove from all parameter names')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to load the model on (cpu/cuda)')
    parser.add_argument('--half', action='store_true',
                        help='Convert model to float16 precision')
    parser.add_argument('--safe_serialization', action='store_true',
                        help='Save model using safetensors format')
    parser.add_argument('--copy_tokenizer', action='store_true',
                        help='Copy tokenizer files directly from base model instead of re-saving')
    return parser.parse_args()

def load_model_weights(model_path: str, model_key: Optional[str] = None, device: str = 'cpu') -> Dict[str, torch.Tensor]:
    
    if os.path.isdir(model_path):
        weight_files = [f for f in os.listdir(model_path) if f.endswith('.pt') or f.endswith('.bin') or f.endswith('.pth')]
        if not weight_files:
            raise FileNotFoundError(f"No weight files found in {model_path}")
        
        model_path = os.path.join(model_path, weight_files[0])
    try:
        checkpoint = torch.load(model_path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Error: {e}")

    if isinstance(checkpoint, dict):
        if model_key is not None and model_key in checkpoint:
            state_dict = checkpoint[model_key]
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'module' in checkpoint:
            state_dict = checkpoint['module']
        elif len(checkpoint) == 1:
            key = list(checkpoint.keys())[0]
            state_dict = checkpoint[key]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise ValueError(f"Loaded object is not a dictionary, but {type(state_dict)}")

    tensor_count = sum(1 for v in state_dict.values() if isinstance(v, torch.Tensor))
    if tensor_count == 0:
        raise ValueError("No tensors found in state_dict")
    
    return state_dict

def process_state_dict(state_dict: Dict[str, Any], add_prefix: Optional[str] = None, remove_prefix: Optional[str] = None) -> Dict[str, torch.Tensor]:
    
    processed_dict = {}

    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        
        new_key = key

        if remove_prefix and new_key.startswith(remove_prefix):
            new_key = new_key[len(remove_prefix):]

        if add_prefix:
            new_key = f"{add_prefix}{new_key}"
        
        processed_dict[new_key] = value

    return processed_dict

def convert_to_hf_format(processed_state_dict: Dict[str, torch.Tensor], 
                        base_model_name: str, 
                        output_path: str, 
                        device: str = 'cpu',
                        half_precision: bool = False,
                        safe_serialization: bool = True,
                        copy_tokenizer: bool = False):
    try:
        from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        raise ImportError("Need to install transformers library: pip install transformers")
    
    if safe_serialization:
        import safetensors

    os.makedirs(output_path, exist_ok=True)

    try:
        config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
    except Exception as e:
        print(f"Warning: Failed to load configuration: {e}")

    if copy_tokenizer:
        try:
            from huggingface_hub import snapshot_download

            cache_dir = snapshot_download(
                repo_id=base_model_name,
                allow_patterns=["tokenizer*", "vocab*", "merges*", "special*", "added*"], 
                local_dir=output_path
            )

        except Exception as e:
            try:
                tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
                tokenizer.save_pretrained(output_path)
            except Exception as e:
                print(f"Warning: Failed to save tokenizer: {e}")
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
            tokenizer.save_pretrained(output_path)
        except Exception as e:
            print(f"Warning: Failed to load tokenizer: {e}")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = AutoModelForCausalLM.from_config(
                config,
                trust_remote_code=True,
            )
    except Exception as e:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    config=config,
                    torch_dtype=torch.float32,
                    device_map=None,
                    trust_remote_code=True
                )
        except Exception as e:
            raise RuntimeError(f"Unable to load model: {e}")

    if half_precision:
        processed_state_dict = {k: v.half() if v.dtype == torch.float32 else v 
                              for k, v in processed_state_dict.items()}

    model_state_dict = model.state_dict()

    model_keys = list(model_state_dict.keys())[:5]
    processed_keys = list(processed_state_dict.keys())[:5]

    for key in model_keys:
        print(f"  {key}: {model_state_dict[key].shape}")

    for key in processed_keys:
        if key in processed_state_dict:
            print(f"  {key}: {processed_state_dict[key].shape}")

    matched = 0
    mismatched_shapes = 0
    not_found = 0

    matched_state_dict = {}

    for key in model_state_dict.keys():
        if key in processed_state_dict and model_state_dict[key].shape == processed_state_dict[key].shape:
            matched_state_dict[key] = processed_state_dict[key]
            matched += 1

    if matched < len(model_state_dict) * 0.5:  
        common_prefixes = ['model.', 'transformer.', 'module.', '']
        
        for model_key in model_state_dict.keys():
            if model_key in matched_state_dict:
                continue
                
            for prefix in common_prefixes:
                for source_prefix in common_prefixes:
                    possible_key = prefix + model_key.lstrip(source_prefix)
                    if possible_key in processed_state_dict and model_state_dict[model_key].shape == processed_state_dict[possible_key].shape:
                        matched_state_dict[model_key] = processed_state_dict[possible_key]
                        matched += 1
                        break
                else:
                    continue
                break

    shape_to_keys = {}
    for key, tensor in processed_state_dict.items():
        shape = tuple(tensor.shape)
        if shape not in shape_to_keys:
            shape_to_keys[shape] = []
        shape_to_keys[shape].append(key)
    
    for model_key, model_tensor in model_state_dict.items():
        if model_key in matched_state_dict:
            continue
            
        shape = tuple(model_tensor.shape)
        if shape in shape_to_keys and len(shape_to_keys[shape]) > 0:

            best_key = shape_to_keys[shape][0]
            matched_state_dict[model_key] = processed_state_dict[best_key]
            shape_to_keys[shape].remove(best_key)
            matched += 1

    for key in model_state_dict.keys():
        if key not in matched_state_dict:
            not_found += 1

    try:
        missing_keys, unexpected_keys = model.load_state_dict(matched_state_dict, strict=False)
        
        if missing_keys:
            print(f"\nMissing keys ({len(missing_keys)}):")
            for key in missing_keys[:10]:
                print(f"  {key}")
            if len(missing_keys) > 10:
                print(f"  ... and {len(missing_keys) - 10} more keys")
        
        if unexpected_keys:
            print(f"\nUnexpected keys ({len(unexpected_keys)}):")
            for key in unexpected_keys[:10]:
                print(f"  {key}")
            if len(unexpected_keys) > 10:
                print(f"  ... and {len(unexpected_keys) - 10} more keys")
    except Exception as e:
        print(f"Error loading weights: {e}")

    if matched / len(model_state_dict) < 0.8:
        print("\n⚠️ Warning: Less than 80% parameters matched, model may not work properly!")
    
    if half_precision:
        model = model.half()

    if safe_serialization:
        try:
            model.save_pretrained(output_path, safe_serialization=True)
        except:
            model.save_pretrained(output_path)
    else:
        model.save_pretrained(output_path)

    if not os.path.exists(os.path.join(output_path, "config.json")):
        try:
            config.save_pretrained(output_path)
        except Exception as e:
            print(f"Warning: Failed to save configuration file: {e}")
            
    return model

def main():
    args = parse_args()
    
    try:
        state_dict = load_model_weights(
            args.model_path,
            args.model_key,
            args.device
        )

        processed_dict = process_state_dict(
            state_dict,
            args.add_prefix,
            args.remove_prefix
        )

        model = convert_to_hf_format(
            processed_dict,
            args.base_model_name,
            args.output_path,
            args.device,
            args.half,
            args.safe_serialization,
            args.copy_tokenizer
        )
        
    except Exception as e:
        print(f"\nConversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
