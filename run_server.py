import os

from django.core.management import execute_from_command_line
from dotenv import load_dotenv


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


if __name__ == "__main__":
    load_dotenv()
    address = f"{required_env('APP_HOST')}:{required_env('APP_PORT')}"
    execute_from_command_line(["manage.py", "runserver", address, "--noreload"])
