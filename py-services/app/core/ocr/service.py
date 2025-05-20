from typing import Dict, Union, Optional, Any
from pathlib import Path
import numpy as np
import io
import os
import time
from PIL import Image

from app.core.ocr.factory import OCRFactory
from app.core.ocr.postprocess import OCRPostProcessor
from app.config.settings import settings
from app.utils.logger import logger


class OCRService:
    """
    OCR服务类，提供图像文本识别的完整流程
    包括OCR引擎管理、图像处理、结果后处理等功能
    """
    
    def __init__(self, engine_type: Optional[str] = None, post_processor: Optional[OCRPostProcessor] = None):
        """
        初始化OCR服务
        
        Args:
            engine_type: OCR引擎类型，默认使用配置中指定的引擎
            post_processor: OCR结果后处理器，如果为None则创建默认后处理器
        """
        self.engine = OCRFactory.create_engine(engine_type)
        self.post_processor = post_processor if post_processor else OCRPostProcessor()
        logger.info(f"OCR service initialized with engine: {self.engine.get_engine_info()['name']}")
    
    def process_image_file(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        处理图像文件并提取文本
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            Dict[str, Any]: 包含OCR结果的字典
        """
        try:
            start_time = time.time()
            
            # 确保路径是字符串类型
            if isinstance(image_path, Path):
                image_path = str(image_path)
                
            # 检查文件是否存在
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # 获取原始OCR结果
            logger.info(f"Processing image file: {image_path}")
            ocr_results = self.engine.process_image(image_path)
            
            # 后处理OCR结果
            paragraphs, full_text = self.post_processor.process(ocr_results)
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "processing_time": processing_time,
                "text_count": len(ocr_results),
                "paragraph_count": len(paragraphs),
                "paragraphs": paragraphs,
                "full_text": full_text,
                "raw_results": [
                    {"text": r.text, "confidence": r.confidence, "box": r.box} 
                    for r in ocr_results
                ],
                "engine_info": self.engine.get_engine_info()
            }
        
        except Exception as e:
            logger.error(f"Error processing image file: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def process_image_data(self, image_data: np.ndarray) -> Dict[str, Any]:
        """
        处理图像数据（numpy数组）并提取文本
        
        Args:
            image_data: 图像数据（numpy数组）
            
        Returns:
            Dict[str, Any]: 包含OCR结果的字典
        """
        try:
            start_time = time.time()
            
            # 检查图像数据
            if image_data is None or image_data.size == 0:
                # 直接抛出异常，而不是记录错误后继续
                logger.error("Invalid image data")
                raise ValueError("Invalid image data")
            
            # 获取原始OCR结果
            logger.info("Processing image data array")
            ocr_results = self.engine.process_image(image_data)
            
            # 后处理OCR结果
            paragraphs, full_text = self.post_processor.process(ocr_results)
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "processing_time": processing_time,
                "text_count": len(ocr_results),
                "paragraph_count": len(paragraphs),
                "paragraphs": paragraphs,
                "full_text": full_text,
                "raw_results": [
                    {"text": r.text, "confidence": r.confidence, "box": r.box} 
                    for r in ocr_results
                ],
                "engine_info": self.engine.get_engine_info()
            }
            
        except Exception as e:
            logger.error(f"Error processing image data: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def process_image_bytes(self, image_bytes: bytes, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        处理图像字节数据并提取文本
        
        Args:
            image_bytes: 图像字节数据
            filename: 原始文件名（可选，仅用于日志记录）
            
        Returns:
            Dict[str, Any]: 包含OCR结果的字典
        """
        try:
            start_time = time.time()
            
            # 将字节数据转换为numpy数组
            logger.info(f"Processing image from bytes{f' (filename: {filename})' if filename else ''}")
            image = Image.open(io.BytesIO(image_bytes))
            image_np = np.array(image)
            
            # 获取原始OCR结果
            ocr_results = self.engine.process_image(image_np)
            
            # 后处理OCR结果
            paragraphs, full_text = self.post_processor.process(ocr_results)
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "processing_time": processing_time,
                "text_count": len(ocr_results),
                "paragraph_count": len(paragraphs),
                "paragraphs": paragraphs,
                "full_text": full_text,
                "raw_results": [
                    {"text": r.text, "confidence": r.confidence, "box": r.box} 
                    for r in ocr_results
                ],
                "engine_info": self.engine.get_engine_info()
            }
            
        except Exception as e:
            logger.error(f"Error processing image bytes: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def save_image_for_ocr(self, image_bytes: bytes, filename: Optional[str] = None) -> Path:
        """
        保存上传的图像以供OCR处理
        
        Args:
            image_bytes: 图像字节数据
            filename: 原始文件名（可选）
            
        Returns:
            Path: 保存的图像文件路径
        """
        # 确保上传目录存在
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        
        # 生成唯一文件名
        if filename:
            # 确保文件名安全
            safe_filename = os.path.basename(filename)
            timestamp = int(time.time())
            file_path = settings.UPLOAD_DIR / f"{timestamp}_{safe_filename}"
        else:
            timestamp = int(time.time())
            file_path = settings.UPLOAD_DIR / f"{timestamp}_upload.png"
        
        # 保存图像
        with open(file_path, "wb") as f:
            f.write(image_bytes)
        
        logger.info(f"Saved image for OCR processing: {file_path}")
        return file_path
    
    def change_engine(self, engine_type: str) -> bool:
        """
        更改OCR引擎
        
        Args:
            engine_type: 新的OCR引擎类型
            
        Returns:
            bool: 是否成功更改引擎
        """
        try:
            # 创建新的引擎
            new_engine = OCRFactory.create_engine(engine_type)
            
            # 切换到新引擎
            self.engine = new_engine
            
            logger.info(f"OCR engine changed to: {engine_type}")
            return True
        except Exception as e:
            logger.error(f"Failed to change OCR engine: {str(e)}")
            return False
    
    @staticmethod
    def list_available_engines() -> Dict[str, str]:
        """
        列出所有可用的OCR引擎
        
        Returns:
            Dict[str, str]: 引擎名称及其描述
        """
        return OCRFactory.list_available_engines()


# 单例实例，方便直接导入使用
ocr_service = OCRService()