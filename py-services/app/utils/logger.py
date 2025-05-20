import sys
import os
from loguru import logger
from pathlib import Path

from app.config.settings import settings

def setup_logger():
    """
    配置并初始化日志记录器
    设置日志格式、级别和输出位置
    """
    # 移除默认处理器
    logger.remove()
    
    # 日志格式配置
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    
    # 配置控制台日志
    logger.add(
        sys.stderr,
        format=log_format,
        level=settings.LOG_LEVEL,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # 如果配置了日志文件，添加文件日志处理器
    if settings.LOG_FILE:
        log_path = Path(settings.LOG_FILE)
        if not log_path.is_absolute():
            log_path = settings.BASE_DIR / settings.LOG_FILE
            
        # 确保日志目录存在
        os.makedirs(log_path.parent, exist_ok=True)
        
        # 添加文件日志处理器
        logger.add(
            str(log_path),
            format=log_format,
            level=settings.LOG_LEVEL,
            rotation="10 MB",  # 日志文件达到10MB时轮转
            retention="1 week",  # 保留一周的日志
            compression="zip",   # 压缩旧日志
            backtrace=True,
            diagnose=True
        )
    
    logger.info(f"Logger initialized with level: {settings.LOG_LEVEL}")
    return logger

# 初始化并导出日志记录器
logger = setup_logger()