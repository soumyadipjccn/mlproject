import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_path = os.path.join(os.getcwd(), "logs")

try:
    os.makedirs(log_path, exist_ok=True)
except Exception as e:
    print("Error creating log directory:", e)

LOG_FILEPATH = os.path.join(log_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILEPATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s -%(message)s",
    level=logging.INFO,
)

if __name__ == "__main__":
    logging.info("Logging Has Started")
