from main import app
import uvicorn
from src.configuration_handler.config_loader import load_configuration

load_configuration()
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7979)
