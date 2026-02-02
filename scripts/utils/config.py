import os, inspect, yaml, argparse

DEFAULT_CONFIG_FILE = "config/config_yolo.yaml"

def replace_with_constants(config_dict, constants):
    for key, value in config_dict.items():
        if isinstance(value, str):
            for string_template, path_part in constants.items():
                value = config_dict[key]
                if string_template in value:
                    config_dict[key] = value.replace(string_template, path_part)
        elif isinstance(value, dict):
            config_dict[key] = replace_with_constants(value, constants)
    return config_dict

def get_config(info_str):
    caller = os.path.basename(inspect.stack()[1].filename)
    parser = argparse.ArgumentParser(info_str)
    parser.add_argument("--cfg", type=str, help="Path to the config file", default=DEFAULT_CONFIG_FILE)
    with open(parser.parse_args().cfg) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
        constants = cfg['constants']
        cfg = cfg.get(caller, {}) | cfg['global']
        return replace_with_constants(cfg, constants)