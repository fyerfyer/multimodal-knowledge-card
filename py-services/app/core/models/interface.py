from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Optional
from pathlib import Path
import numpy as np

class MultiModalModelInterface(ABC):
    """
    多模态模型接口
    定义所有多模态大语言模型实现必须提供的方法
    """
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        初始化模型
        
        Returns:
            bool: 是否成功初始化
        """
        pass
    
    @abstractmethod
    def generate_response(
        self, 
        prompt: str,
        image: Optional[Union[str, Path, np.ndarray]] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        生成模型响应
        
        Args:
            prompt: 文本提示词
            image: 可选的图像输入，可以是图像路径字符串、Path对象或numpy数组
            options: 可选的生成参数
            
        Returns:
            Dict[str, Any]: 包含生成结果的字典
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 包含模型名称、参数量等信息的字典
        """
        pass
    
    @abstractmethod
    def is_initialized(self) -> bool:
        """
        检查模型是否已初始化
        
        Returns:
            bool: 模型是否已初始化
        """
        pass

class ModelResponse:
    """
    模型响应封装类
    用于统一处理各种多模态模型的响应格式
    """
    
    def __init__(
        self, 
        text: str, 
        success: bool = True,
        error: Optional[str] = None,
        usage: Optional[Dict[str, int]] = None,
        model_name: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        初始化模型响应对象
        
        Args:
            text: 生成的文本内容
            success: 是否成功生成
            error: 错误信息（如果有）
            usage: 资源使用情况（如token数量）
            model_name: 模型名称
            metadata: 其他元数据
        """
        self.text = text
        self.success = success
        self.error = error
        self.usage = usage or {}
        self.model_name = model_name
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将响应转换为字典
        
        Returns:
            Dict[str, Any]: 响应字典
        """
        result = {
            "success": self.success,
            "text": self.text,
            "model_name": self.model_name,
            "usage": self.usage,
            "metadata": self.metadata
        }
        
        if self.error:
            result["error"] = self.error
            
        return result