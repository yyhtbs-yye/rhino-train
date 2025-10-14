import yaml
import os
import copy
from typing import Dict, Any

def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file with imports and variable substitutions.
    
    Args:
        file_path: Path to the YAML file.
        
    Returns:
        The fully resolved configuration.
    """
    # Get the absolute path
    abs_path = os.path.abspath(file_path)
    base_dir = os.path.dirname(abs_path)
    
    # Load the YAML file
    with open(abs_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Process imports
    if '_import' in config:
        import_path = config.pop('_import')
        
        # Load the imported configuration
        imported_config = load_yaml_config(import_path)
        
        # Extract variables
        vars_dict = {}
        if '_vars' in config:
            vars_dict = config.pop('_vars')
            
        # Merge configurations (handles direct overrides)
        merged_config = deep_merge(imported_config, config)
        
        # Apply variables (handles anchor-based overrides)
        if vars_dict:
            merged_config = apply_variables(merged_config, vars_dict)
            
        return merged_config
    else:
        # No imports, just return the configuration
        return config


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        base: Base dictionary.
        override: Dictionary with values to override base.
        
    Returns:
        Merged dictionary.
    """
    result = copy.deepcopy(base)
    
    for key, value in override.items():
        # If both are dictionaries, merge recursively
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            # Otherwise, override the value
            result[key] = copy.deepcopy(value)
            
    return result


def apply_variables(config: Dict[str, Any], vars_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply variables to replace anchors in the configuration.
    
    Args:
        config: Configuration dictionary.
        vars_dict: Dictionary of variables.
        
    Returns:
        Configuration with variables applied.
    """
    # Make a deep copy of the config to avoid modifying the original
    result = copy.deepcopy(config)
    
    # Define recursive function to traverse and replace values
    def traverse_and_replace(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    traverse_and_replace(value)
                elif isinstance(value, str) and value.startswith('$'):
                    var_name = value[1:]  # Remove the '$' prefix
                    if var_name in vars_dict:
                        obj[key] = copy.deepcopy(vars_dict[var_name])
                    
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, (dict, list)):
                    traverse_and_replace(item)
                elif isinstance(item, str) and item.startswith('$'):
                    var_name = item[1:]  # Remove the '$' prefix
                    if var_name in vars_dict:
                        obj[i] = copy.deepcopy(vars_dict[var_name])
    
    # Apply the replacement
    traverse_and_replace(result)
    return result

if __name__=="__name__":
    # Load the configuration
    config = load_yaml_config('configs/restoration/train_sr3_ddim_16_128.yaml')

    # Now you can access the merged configuration
    print(yaml.dump(config, default_flow_style=False))