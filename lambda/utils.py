import logging

logger = logging.getLogger(__name__)


def load_system_prompt(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read().strip()
    except Exception as e:
        logger.error(f"Error reading system prompt file: {str(e)}")
        return "Default system prompt in case of error"  
