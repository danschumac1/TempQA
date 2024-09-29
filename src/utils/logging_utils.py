import logging.config
import json

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