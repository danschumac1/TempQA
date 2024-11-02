from datetime import datetime
import logging.config
import json
import os

def gen_logger(type_log: str = "INFO", message: str='', log_file: str = "logs/generate.log", init: bool = False):
    # Ensure log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Initialize new run header if requested
    if init:
        with open(log_file, 'a') as f:
            f.write("\n" + "="*40 + f"\nNew Run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" + "="*40 + "\n")

    # Format the log entry with timestamp and log type
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} [{type_log}] {message}\n"
    
    # Append the log entry to the file
    try:
        with open(log_file, "a") as f:
            f.write(log_entry)
    except IOError as e:
        print(f"Logging Error: {e}")

def setup_logging(logger_name) -> logging.Logger:
    """
    Set up logging configuration using the specified JSON file and return the logger for the given phase.
    
    Args:
        logger_name (str): The name of the logger to retrieve (e.g., 'trainer_logger', 'generation_logger', 'evaluation_logger').
    
    Returns:
        logging.Logger: The configured logger instance.
    """
    config_file = "resources/logging.json"
    
    with open(config_file, "rt") as f:
        config = json.load(f)
    
    # Apply logging configuration
    logging.config.dictConfig(config)
    
    # Return the specific logger for the given phase
    logger = logging.getLogger(logger_name)
    
    return logger