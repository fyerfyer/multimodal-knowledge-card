from typing import Optional
from pydantic_settings import BaseSettings
from pathlib import Path
import os

if os.environ.get('USE_HF_MIRROR', 'True').lower() in ('true', '1', 't'):
    os.environ['HF_ENDPOINT'] = os.environ.get('HF_MIRROR', 'https://hf-mirror.com')
    print(f"Setting HuggingFace mirror in settings.py: {os.environ['HF_ENDPOINT']}")

class Settings(BaseSettings):
    """
    系统配置类，用于管理应用程序的所有配置参数
    使用环境变量或.env文件加载配置
    """
    # 基础配置
    PROJECT_NAME: str = "多模态智能知识卡片系统"
    API_PREFIX: str = "/api"
    DEBUG: bool = True
    
    # 服务配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # 路径配置
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    
    # OCR配置
    OCR_ENGINE: str = "paddleocr"  # paddleocr 或 easyocr
    OCR_LANG: str = "ch"  # 中文识别
    
    # 图像理解模型配置
    VISION_MODEL: str = "blip"  # blip 或 clip
    VISION_MODEL_PATH: Optional[str] = None
    VISION_VQA_MODEL_PATH: Optional[str] = None  # 添加VQA模型路径配置
    
    # 多模态语言模型配置
    LLM_MODEL: str = "qwen-vl"  # qwen-vl 或 deepseek-vl
    LLM_MODEL_PATH: Optional[str] = None
    LLM_API_KEY: Optional[str] = None
    
    # DashScope API配置
    DASHSCOPE_API_KEY: Optional[str] = None
    
    # DeepSeek API配置
    DEEPSEEK_API_KEY: Optional[str] = None  
    
    # 模型下载配置
    HF_MIRROR: str = "https://hf-mirror.com"
    USE_HF_MIRROR: bool = True
    HF_ENDPOINT: Optional[str] = None  
    
    # 缓存配置
    ENABLE_CACHE: bool = True
    CACHE_EXPIRATION: int = 3600  # 缓存过期时间（秒）
    
    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = "app.log"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

    def create_upload_dir(self) -> None:
        """创建上传目录（如果不存在）"""
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)
    
    @property
    def api_url_prefix(self) -> str:
        """获取API URL前缀"""
        return self.API_PREFIX

# 创建全局设置实例
settings = Settings()

# 确保上传目录存在
settings.create_upload_dir()