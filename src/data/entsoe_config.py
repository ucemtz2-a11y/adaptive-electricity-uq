# Read the small amount of shared setup needed by the ENTSO-E download scripts.

"""Shared configuration and credential helpers for ENTSO-E downloads."""

import os

import yaml
from dotenv import load_dotenv


# Read dates and other download settings from the project YAML file.
def load_config(path="config/config.yaml"):
    with open(path, "r") as file:
        return yaml.safe_load(file)


# Prefer the environment variable, but allow the YAML value for older local setups.
def get_api_key(config):
    load_dotenv()

    api_key = os.getenv("ENTSOE_API_KEY")

    if api_key is None and "entsoe" in config:
        api_key = config["entsoe"].get("api_key")

    if api_key is None:
        raise ValueError("ENTSOE API key not found.")

    return api_key
