#!/usr/bin/env python
import sys
import json
import re
import ast



def update_nested_dict_full_key(nested_dict, target_key, value):
    ''' if target_key is key1.key2.....keyN, update nested_dict['key1']['key2']...['keyN'] to value
    '''
    keys = target_key.split('.')
    for key in keys[:-1]:
        nested_dict = nested_dict[key]
    nested_dict[keys[-1]] = value


def update_nested_dict_unique_key(nested_dict, target_key, value):
    """
    Recursively updates a nested dictionary by finding the specified key and assigning the new value to it.
    If the key is not found, raise a ValueError.
    If the key is found multiple times, a ValueError is raised.

    Args:
    - nested_dict (dict): the nested dictionary to update
    - target_key (str): the key to find and update
    - value (any): the new value to assign to the key

    """
    def update_dict(d, key, new_value):
        found = 0
        for k, v in d.items():
            if k == key:
                d[k] = new_value
                found += 1
            elif isinstance(v, dict):
                sub_found = update_dict(v, key, new_value)
                if sub_found > 0:
                    found += sub_found
        return found

    found_count = update_dict(nested_dict, target_key, value)
    
    if found_count > 1:
        raise ValueError('Key "{}" found multiple times in dictionary'.format(target_key))
    elif found_count == 0:
        raise ValueError('Key "{}" not found in dictionary'.format(target_key))

    return found_count


def update_nested_dict(nested_dict, target_key, value):
    if '.' in target_key:
        return update_nested_dict_full_key(nested_dict, target_key, value)
    else:
        return update_nested_dict_unique_key(nested_dict, target_key, value)



def copy_nested_dict(orig_dict, update_dict):
    for key, value in update_dict.items():
        if isinstance(value, dict):
            orig_dict[key] = copy_nested_dict(orig_dict.get(key, {}), value)
        else:
            orig_dict[key] = value
    return orig_dict


def get_nested_dict_unique_key(nested_dict, target_key):
    """
    get value from nested dict, if not found, result is None
    """
    for k, v in nested_dict.items():
        if k == target_key:
            return v
        elif isinstance(v, dict):
            result = get_nested_dict_unique_key(v, target_key)
            if result is not None:
                return result

def get_nested_dict_full_key(nested_dict, target_key):
    ''' if target_key is key1.key2.....keyN, get nested_dict['key1']['key2']...['keyN']
    '''
    keys = target_key.split('.')
    for key in keys[:-1]:
        nested_dict = nested_dict[key]
    return nested_dict[keys[-1]]

def get_nested_dict(nested_dict, target_key):
    if '.' in target_key:
        return get_nested_dict_full_key(nested_dict, target_key)
    else:
        return get_nested_dict_unique_key(nested_dict, target_key)

class BaseOptions:
    def __init__(self):
        self.opts = None
    
    

    def parse_nest_args(self, *args):
        # parse args according to dictionary
        i = 0
        while i < len(args):
            key = args[i]
            default_val = get_nested_dict(self.opts, key)
            # if default_val is string, also save arg-value as string
            # otherwise, try to convert to other type
            if isinstance(default_val,str):
                val = args[i+1]
            elif isinstance(default_val, list):
                val = args[i+1].split(',')
                # Try to convert each element to float if it's numeric
                converted_val = []
                for item in val:
                    item = item.strip()
                    try:
                        # Try to convert to float
                        converted_val.append(float(item))
                    except ValueError:
                        # If conversion fails, keep as string
                        converted_val.append(item)
                val = converted_val
            else:
                try:
                    val = ast.literal_eval(args[i+1])
                except ValueError as ve:
                    print(f'error parsing {args[i]} and {args[i+1]}: {ve}')
                    raise
            
            found = update_nested_dict(self.opts, key, val)
            if found==0:
                raise ValueError('Key %s not found in dictionary' % key)
            i +=2
    
    def convert_to_dict(self, param_val_str):
        ''' convert string of param1,value1,param2,value2 to dictionary, handling tuples/lists with commas '''
        if param_val_str == '':
            return {}

        # Split on commas not inside parentheses
        parts = re.split(r',(?![^(]*\))', param_val_str)
        param_val_list = [p.strip() for p in parts]
        param_val_dict = {}
        for i in range(0, len(param_val_list), 2):
            key = param_val_list[i]
            value = ast.literal_eval(param_val_list[i+1])
            param_val_dict[key] = value
        return param_val_dict