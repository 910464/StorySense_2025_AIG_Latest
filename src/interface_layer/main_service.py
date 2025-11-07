from main import app
import uvicorn
from src.configuration_handler.config_loader import load_configuration

load_configuration()
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7979)
# if __name__ == "__main__":
#     try:
#         # Changed port to 8000 (common FastAPI default)
#         # Or you could use: 8080, 3000, 5000, or 8888
#         PORT = 8000
#         uvicorn.run(app, host="127.0.0.1", port=PORT)
#     except Exception as e:
#         logger.error(f"Failed to start server: {str(e)}")