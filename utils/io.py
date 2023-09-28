import os
import yaml


def load_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        return yaml.full_load(f)



