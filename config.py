import os
import pathlib

from dotenv import load_dotenv

ROOT_DIR = pathlib.Path(__file__).parents[1].resolve()

dotenv_path = os.path.join(ROOT_DIR, ".env")
load_dotenv(dotenv_path)

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
