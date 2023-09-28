import os
from loguru import logger as pylogger
from datetime import date


class PythonLogger:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PythonLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not PythonLogger._initialized:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            today = date.today()
            dir_name = os.path.join(script_dir, "log", str(today))
            file_name = f"{today}.log"
            log_path = os.path.join(dir_name, file_name)

            os.makedirs(dir_name, exist_ok=True)
            pylogger.add(log_path, rotation="1 days")

            PythonLogger._initialized = True

    def debug(self, msg):
        pylogger.debug(msg)

    def info(self, msg):
        pylogger.info(msg)

    def success(self, msg):
        pylogger.success(msg)

    def warning(self, msg):
        pylogger.warning(msg)

    def error(self, msg):
        pylogger.error(msg)

    def critical(self, msg):
        pylogger.critical(msg)
