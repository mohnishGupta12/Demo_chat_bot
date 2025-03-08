from  langchain_community.document_loaders import DirectoryLoader
from langchain.schema import Document
import mistune
import re
import logging
from functools import wraps

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("AppLogger")

def handle_exceptions(func):
    """
    Decorator to handle exceptions and log errors.

    Args:
        func (function): Function to wrap with error handling.

    Returns:
        function: Wrapped function that logs errors instead of raising exceptions.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as err:
            logger.exception(f"Error in {func.__name__}: {str(err)}") 
            return None  # Return None to prevent crashes
    return wrapper

@handle_exceptions
def clean_markdown(text: str) -> str:
    """
    Cleans Markdown text by removing unnecessary formatting elements.

    Args:
        text (str): The raw markdown text.

    Returns:
        str: Cleaned text without markdown syntax.
    """
    if not isinstance(text, str):
        logger.error("Invalid input: clean_markdown expects a string.")
        return ""

    logger.info("Cleaning markdown text...")
    
    text = re.sub(r"\n+", "\n", text)  # Remove multiple newlines
    text = re.sub(r"#+\s*", "", text)  # Remove Markdown headers
    text = re.sub(r"\*{1,2}|\_{1,2}", "", text)  # Remove bold/italic markers (*, **, _, __)
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)  # Remove links
    text = re.sub(r"`+", "", text)  # Remove inline code markers

    cleaned_text = text.strip()
    logger.info("Markdown text cleaned successfully.")

    return cleaned_text
