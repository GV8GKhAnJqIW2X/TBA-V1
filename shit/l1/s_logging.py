from loguru import logger

logger.add(
    "logs/{time:DD-MM-YYYY}.log", 
    format="{time} {level} {message}", 
    level="INFO",
    rotation="00:00", 
    serialize=False,
)