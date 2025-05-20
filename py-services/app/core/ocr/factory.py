from typing import Dict, Type, Optional
import importlib

from app.core.ocr.interface import OCRInterface
from app.core.ocr.paddle_ocr import PaddleOCREngine
from app.config.settings import settings
from app.utils.logger import logger

class OCRFactory:
    """OCR引擎工厂类，用于创建不同类型的OCR引擎实例"""
    
    # 注册可用的OCR引擎
    _engines: Dict[str, Type[OCRInterface]] = {
        "paddleocr": PaddleOCREngine
    }
    
    @classmethod
    def register_engine(cls, name: str, engine_class: Type[OCRInterface]) -> None:
        """
        注册新的OCR引擎
        
        Args:
            name: 引擎名称
            engine_class: 引擎类（必须实现OCRInterface接口）
        """
        cls._engines[name.lower()] = engine_class
        logger.info(f"Registered OCR engine: {name}")
    
    @classmethod
    def create_engine(cls, engine_type: Optional[str] = None, **kwargs) -> OCRInterface:
        """
        创建OCR引擎实例
        
        Args:
            engine_type: 引擎类型名称，如果为None则使用配置中的默认引擎
            **kwargs: 传递给引擎构造函数的参数
            
        Returns:
            OCRInterface: OCR引擎实例
            
        Raises:
            ValueError: 如果指定的引擎类型不存在
        """
        # 如果没有指定引擎类型，使用配置中的默认值
        if engine_type is None:
            engine_type = settings.OCR_ENGINE
        
        engine_type = engine_type.lower()
        
        # 检查引擎是否已注册
        if engine_type not in cls._engines:
            # 尝试动态导入引擎模块（用于后续可能添加的其他引擎）
            try:
                module_path = f"app.core.ocr.{engine_type.lower().replace('-', '_')}_ocr"
                module = importlib.import_module(module_path)
                
                # 假设模块中有一个与引擎同名的类（首字母大写）
                class_name = ''.join(word.capitalize() for word in engine_type.split('-')) + 'Engine'
                if hasattr(module, class_name):
                    engine_class = getattr(module, class_name)
                    cls.register_engine(engine_type, engine_class)
                else:
                    logger.error(f"Could not find OCR engine class {class_name} in module {module_path}")
                    raise ValueError(f"Unsupported OCR engine: {engine_type}")
            except ImportError:
                logger.error(f"Could not import OCR engine module for engine type: {engine_type}")
                raise ValueError(f"Unsupported OCR engine: {engine_type}")
        
        # 创建引擎实例
        engine_class = cls._engines[engine_type]
        logger.info(f"Creating OCR engine: {engine_type}")
        
        return engine_class(**kwargs)
    
    @classmethod
    def list_available_engines(cls) -> Dict[str, str]:
        """
        列出所有可用的OCR引擎
        
        Returns:
            Dict[str, str]: 引擎名称及其描述的字典
        """
        return {name: engine.__doc__ or "" for name, engine in cls._engines.items()}