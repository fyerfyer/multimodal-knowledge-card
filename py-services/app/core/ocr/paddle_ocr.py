import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Union, Any, Optional
from paddleocr import PaddleOCR

from app.core.ocr.interface import OCRInterface, OCRResult
from app.config.settings import settings
from app.utils.logger import logger

class PaddleOCREngine(OCRInterface):
    """
    PaddleOCR引擎实现
    集成PaddleOCR引擎提供OCR服务
    """
    
    def __init__(
        self, 
        lang: str = settings.OCR_LANG, 
        use_gpu: bool = False,  # 注意：此参数在当前版本的PaddleOCR中未使用
        enable_mkldnn: bool = True  # 注意：此参数在当前版本的PaddleOCR中未使用
    ):
        """
        初始化PaddleOCR引擎
        
        Args:
            lang: 识别语言，默认为配置中设置的语言
            use_gpu: 是否使用GPU加速
            enable_mkldnn: 是否启用MKLDNN加速
        """
        self.lang = lang
        self.use_gpu = use_gpu
        self.enable_mkldnn = enable_mkldnn
        self.ocr_engine = None
        self.initialize()
    
    def initialize(self) -> None:
        """
        初始化PaddleOCR引擎
        """
        try:
            logger.info(f"Initializing PaddleOCR with language: {self.lang}")
            # 创建PaddleOCR实例，配置检测、识别和方向分类器
            self.ocr_engine = PaddleOCR(
                use_textline_orientation=True,  # 替换use_angle_cls，使用方向分类器
                lang=self.lang,                 # 语言设置
                # use_gpu=self.use_gpu, 
                # enable_mkldnn=self.enable_mkldnn
            )
            logger.info("PaddleOCR engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {str(e)}")
            raise
    
    def process_image(self, image: Union[str, Path, np.ndarray]) -> List[OCRResult]:
        """
        处理图像并提取文本及位置信息
        
        Args:
            image: 输入图像，可以是图像路径或numpy数组
            
        Returns:
            List[OCRResult]: OCR结果列表
        """
        if self.ocr_engine is None:
            logger.error("OCR engine not initialized")
            raise RuntimeError("OCR engine not initialized")
        
        try:
            # 将Path对象转换为字符串
            if isinstance(image, Path):
                image = str(image)
            
            logger.debug(f"Processing image with PaddleOCR")
            # 执行OCR识别
            ocr_result = self.ocr_engine.predict(image)
            
            # PaddleOCR 3.x版本返回格式可能有区别，需要适配
            if len(ocr_result) > 0 and isinstance(ocr_result[0], list):
                results = []
                # 处理结果格式
                for line in ocr_result[0]:
                    # PaddleOCR结果格式: [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], [text, confidence]]
                    box = line[0]  # 文本框坐标
                    text = line[1][0]  # 文本内容
                    confidence = float(line[1][1])  # 置信度
                    
                    # 创建OCRResult对象并添加到结果列表
                    results.append(OCRResult(text=text, confidence=confidence, box=box))
                    
                logger.info(f"OCR completed. Extracted {len(results)} text regions")
                return results
            else:
                logger.warning("No text detected in the image or unexpected result format")
                return []
                
        except Exception as e:
            logger.error(f"Error processing image with PaddleOCR: {str(e)}")
            raise
    
    def extract_text(self, image: Union[str, Path, np.ndarray]) -> str:
        """
        从图像中提取纯文本
        
        Args:
            image: 输入图像，可以是图像路径或numpy数组
            
        Returns:
            str: 提取的文本内容，以换行符分隔
        """
        results = self.process_image(image)
        
        # 将所有文本合并，按照识别置信度排序
        texts = [result.text for result in sorted(
            results, 
            key=lambda x: (0 if x.box is None else x.box[0][1], 0 if x.box is None else x.box[0][0])
        )]
        
        return "\n".join(texts)
    
    def get_engine_info(self) -> Dict[str, Any]:
        """
        获取OCR引擎信息
        
        Returns:
            Dict[str, Any]: 包含引擎名称、版本等信息的字典
        """
        import paddleocr
        
        return {
            "name": "PaddleOCR",
            "version": paddleocr.__version__,
            "language": self.lang,
            "use_gpu": self.use_gpu,
            "enable_mkldnn": self.enable_mkldnn
        }