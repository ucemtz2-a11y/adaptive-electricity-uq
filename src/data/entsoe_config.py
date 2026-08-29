# Module purpose: Load shared ENTSO-E configuration and API credentials.

"""Shared configuration and credential helpers for ENTSO-E downloads."""

import os

import yaml
from dotenv import load_dotenv


# Load config.
def load_config(path="config/config.yaml"):
    with open(path, "r") as file:
        return yaml.safe_load(file)


# Get API key.
def get_api_key(config):
    load_dotenv()

    api_key = os.getenv("ENTSOE_API_KEY")

    if api_key is None and "entsoe" in config:
        api_key = config["entsoe"].get("api_key")

    if api_key is None:
        raise ValueError("ENTSOE API key not found.")

    return api_key
