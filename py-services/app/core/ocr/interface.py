from abc import ABC, abstractmethod
from typing import List, Dict, Union, Any, Optional
from pathlib import Path
import numpy as np

class OCRResult:
    """OCR结果数据结构，保存检测到的文本及其位置信息"""
    
    def __init__(self, text: str, confidence: float, box: Optional[List[List[int]]] = None):
        """
        初始化OCR结果
        
        Args:
            text: 识别到的文本
            confidence: 识别置信度
            box: 文本框坐标，格式为[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        self.text = text
        self.confidence = confidence
        self.box = box
    
    def __str__(self) -> str:
        return f"Text: {self.text} (Confidence: {self.confidence:.2f})"

class OCRInterface(ABC):
    """
    OCR服务接口抽象类
    定义所有OCR实现必须提供的方法
    """
    
    @abstractmethod
    def initialize(self) -> None:
        """初始化OCR引擎"""
        pass
    
    @abstractmethod
    def process_image(self, image: Union[str, Path, np.ndarray]) -> List[OCRResult]:
        """
        处理图像并提取文本
        
        Args:
            image: 输入图像，可以是图像路径字符串、Path对象或numpy数组
            
        Returns:
            List[OCRResult]: OCR结果列表
        """
        pass
    
    @abstractmethod
    def extract_text(self, image: Union[str, Path, np.ndarray]) -> str:
        """
        从图像中提取纯文本（不含位置信息）
        
        Args:
            image: 输入图像，可以是图像路径字符串、Path对象或numpy数组
            
        Returns:
            str: 提取的文本内容
        """
        pass
    
    @abstractmethod
    def get_engine_info(self) -> Dict[str, Any]:
        """
        获取OCR引擎信息
        
        Returns:
            Dict[str, Any]: 包含引擎名称、版本等信息的字典
        """
        pass