import tensorflow as tf
import yaml
import os
import numpy as np

def get_weights_size(model):
    total_params = sum([tf.size(w).numpy() for w in model.weights])
    size_bytes = int(total_params * 4)  # 4 bytes for float32
    size_kb = size_bytes / 1024
    size_mb = size_kb / 1024
    return {
        'kilobytes': round(size_kb, 2),
        'megabytes': round(size_mb, 2)
    }

def convert(obj):
    if isinstance(obj, tuple):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert(i) for i in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, tf.Tensor):
        return convert(obj.numpy())
    else:
        return obj
    
def save_model_info(name, model, path):
    # Gather info
    model_info = {
        'model_name': model.name,
        'input_shape': list(model.input_shape) if isinstance(model.input_shape, tuple) else model.input_shape,
        'output_shape': list(model.output_shape) if isinstance(model.output_shape, tuple) else model.output_shape,
        'num_layers': len(model.layers),
        'trainable_params': int(tf.reduce_sum([tf.size(w) for w in model.trainable_weights])),
        'non_trainable_params': int(tf.reduce_sum([tf.size(w) for w in model.non_trainable_weights])),
        'total_params': model.count_params(),
        'size': get_weights_size(model),
        'layers': []
    }

    # Layer-wise details
    for layer in model.layers:
        layer_info = {
            'name': layer.name,
            'class_name': layer.__class__.__name__,
            'output_shape': layer.output_shape if hasattr(layer, 'output_shape') else None,
            'trainable': layer.trainable,
            'num_params': layer.count_params() if hasattr(layer, 'count_params') else 0
        }
        model_info['layers'].append(layer_info)

    if os.path.exists(path):
        with open(path, 'r') as f:
            all_model_data = yaml.safe_load(f) or {}  # if file is empty, use {}
    else:
        all_model_data = {}
    
    cleaned_model_info = convert(model_info)
    all_model_data.update({name: cleaned_model_info}) 

    with open(path, 'w') as f:
        yaml.dump(all_model_data, f, sort_keys=False)

    print(f"Model info saved to {path}")