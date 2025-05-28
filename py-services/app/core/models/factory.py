from typing import Dict, Type, Any, Optional
import importlib
from enum import Enum

from app.core.models.interface import MultiModalModelInterface
from app.core.models.qwen_vl import QwenVLModel
from app.config.settings import settings
from app.utils.logger import logger

class ModelType(Enum):
    """多模态模型类型枚举"""
    QWEN_VL = "qwen-vl"


class ModelFactory:
    """
    多模态模型工厂类
    负责创建不同类型的多模态模型实例
    """
    
    # 注册可用的模型类
    _models: Dict[str, Type[MultiModalModelInterface]] = {
        ModelType.QWEN_VL.value: QwenVLModel,
    }
    
    @classmethod
    def register_model(cls, model_type: str, model_class: Type[MultiModalModelInterface]) -> None:
        """
        注册新的模型类型
        
        Args:
            model_type: 模型类型名称
            model_class: 模型类（必须实现MultiModalModelInterface接口）
        """
        cls._models[model_type.lower()] = model_class
        logger.info(f"Registered model type: {model_type}")
    
    @classmethod
    def create_model(cls, 
                    model_type: Optional[str] = None, 
                    **kwargs) -> MultiModalModelInterface:
        """
        创建多模态模型实例
        
        Args:
            model_type: 模型类型名称，如果为None则使用配置中的默认模型
            **kwargs: 传递给模型构造函数的参数
            
        Returns:
            MultiModalModelInterface: 模型实例
            
        Raises:
            ValueError: 如果指定的模型类型不存在
        """
        # 如果没有指定模型类型，使用配置中的默认值
        if model_type is None:
            model_type = settings.LLM_MODEL
        
        model_type = model_type.lower()
        
        # 检查模型是否已注册
        if model_type not in cls._models:
            # 尝试动态导入模型模块
            try:
                module_path = f"app.core.models.{model_type.replace('-', '_')}"
                module = importlib.import_module(module_path)
                
                # 假设模块中有一个与模型同名的类（首字母大写，下划线转驼峰）
                class_name = ''.join(word.capitalize() for word in model_type.replace('-', '_').split('_')) + 'Model'
                if hasattr(module, class_name):
                    model_class = getattr(module, class_name)
                    cls.register_model(model_type, model_class)
                else:
                    logger.error(f"Could not find model class {class_name} in module {module_path}")
                    raise ValueError(f"Unsupported model type: {model_type}")
            except ImportError:
                logger.error(f"Could not import model module for model type: {model_type}")
                raise ValueError(f"Unsupported model type: {model_type}")
        
        # 根据模型类型选择合适的API密钥
        if model_type.startswith("qwen"):
            # 如果没有在kwargs中提供api_key，从设置中获取
            if 'api_key' not in kwargs:
                kwargs['api_key'] = settings.DASHSCOPE_API_KEY
                
            # 设置模型名称（如果没有提供）
            if 'model_name' not in kwargs and settings.LLM_MODEL_PATH:
                kwargs['model_name'] = settings.LLM_MODEL_PATH
        
        # 创建模型实例
        model_class = cls._models[model_type]
        logger.info(f"Creating model instance: {model_type}")
        
        try:
            model_instance = model_class(**kwargs)
            return model_instance
        except Exception as e:
            logger.error(f"Failed to create model instance for {model_type}: {str(e)}")
            raise
    
    @classmethod
    def list_available_models(cls) -> Dict[str, str]:
        """
        列出所有可用的模型类型
        
        Returns:
            Dict[str, str]: 模型类型及其描述的字典
        """
        return {name: model_class.__doc__ or "" for name, model_class in cls._models.items()}
    
    @classmethod
    def get_model_class(cls, model_type: str) -> Type[MultiModalModelInterface]:
        """
        获取指定类型的模型类
        
        Args:
            model_type: 模型类型名称
            
        Returns:
            Type[MultiModalModelInterface]: 模型类
            
        Raises:
            ValueError: 如果指定的模型类型不存在
        """
        model_type = model_type.lower()
        if model_type in cls._models:
            return cls._models[model_type]
        else:
            logger.error(f"Unknown model type: {model_type}")
            raise ValueError(f"Unknown model type: {model_type}")
    
    @classmethod
    def create_model_with_settings(cls, settings_key: str, **kwargs) -> MultiModalModelInterface:
        """
        根据配置项创建模型实例，便于从配置文件中选择模型
        
        Args:
            settings_key: 配置项键名，例如 'LLM_MODEL'
            **kwargs: 传递给模型构造函数的参数
            
        Returns:
            MultiModalModelInterface: 模型实例
        """
        # 从settings获取配置值
        if hasattr(settings, settings_key):
            model_type = getattr(settings, settings_key)
            return cls.create_model(model_type, **kwargs)
        else:
            logger.error(f"Settings key not found: {settings_key}")
            raise ValueError(f"Settings key not found: {settings_key}")


# 用于创建模型实例的简便函数
def create_model(model_type: Optional[str] = None, **kwargs) -> MultiModalModelInterface:
    """
    创建多模态模型实例的便捷函数
    
    Args:
        model_type: 模型类型名称，如果为None则使用配置中的默认模型
        **kwargs: 传递给模型构造函数的参数
        
    Returns:
        MultiModalModelInterface: 模型实例
    """
    return ModelFactory.create_model(model_type, **kwargs)