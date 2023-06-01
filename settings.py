import os
from dotenv import dotenv_values
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent


config = dotenv_values(get_project_root() / ".env")
os.environ["OPENAI_API_KEY"] = config['OPENAI_API_KEY']
#openai.api_key = os.getenv("OPENAI_API_KEY")