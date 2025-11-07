import os
from pathlib import Path
from dotenv import load_dotenv


def load_configuration(env_file_path: str = r"C:\Users\910464\OneDrive - Cognizant\Documents\GitHub\DevOrganization\StorySense_2025_AIG_Latest\.env") -> bool:
    return load_dotenv(dotenv_path=env_file_path)
 