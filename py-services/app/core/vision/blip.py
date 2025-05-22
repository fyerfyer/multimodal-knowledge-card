from typing import Union, List, Dict, Any, Optional
import torch
import os
from pathlib import Path
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering

from app.utils.logger import logger
from app.config.settings import settings
from app.utils.image_utils import read_image, convert_to_rgb

class BLIPModel:
    """
    BLIP模型包装器，用于图像理解、描述生成和VQA
    """
    
    def __init__(
        self, 
        model_name: str = "Salesforce/blip-image-captioning-base",
        device: Optional[str] = None,
        vqa_model_name: Optional[str] = None
    ):
        """
        初始化BLIP模型
        
        Args:
            model_name: 模型名称或路径
            device: 运行设备 ('cuda' 或 'cpu')，如果为None则自动选择
            vqa_model_name: VQA模型名称，如果为None则不加载VQA模型
        """
        self.model_name = model_name
        # 确保device字段为合法值
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vqa_model_name = vqa_model_name or "Salesforce/blip-vqa-base"
        
        self.processor = None
        self.model = None
        self.vqa_model = None
        
        # 设置HuggingFace镜像站点
        if settings.USE_HF_MIRROR and os.environ.get('HF_ENDPOINT') is None:
            os.environ['HF_ENDPOINT'] = settings.HF_MIRROR
            logger.info(f"Using HuggingFace mirror: {settings.HF_MIRROR}")
            
        # 初始化模型
        self._load_captioning_model()
        
        logger.info(f"BLIP model initialized with device: {self.device}")
    
    def _load_captioning_model(self) -> None:
        """
        加载图像描述生成模型，优先使用本地模型
        """
        try:
            # 优先使用配置中指定的本地模型路径
            model_path = settings.VISION_MODEL_PATH or self.model_name
            logger.info(f"Loading BLIP captioning model: {model_path}")
            
            try:
                # 尝试从本地加载
                self.processor = BlipProcessor.from_pretrained(model_path)
                self.model = BlipForConditionalGeneration.from_pretrained(model_path).to(self.device)
                logger.info("BLIP captioning model loaded successfully")
            except Exception as local_err:
                # 如果本地加载失败，尝试从HuggingFace加载
                if settings.VISION_MODEL_PATH:
                    logger.warning(f"Failed to load local model from {model_path}, trying HuggingFace: {local_err}")
                
                self.processor = BlipProcessor.from_pretrained(self.model_name)
                self.model = BlipForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
                logger.info("BLIP captioning model loaded from HuggingFace successfully")
            
        except Exception as e:
            logger.error(f"Failed to load BLIP captioning model: {str(e)}")
            # 出错时不急于崩溃，允许后续重试或使用其他模型
            self.model = None
            self.processor = None
    
    def _load_vqa_model(self) -> bool:
        """
        按需加载VQA模型，优先使用本地模型
        
        Returns:
            bool: 加载是否成功
        """
        if self.vqa_model is not None:
            return True  # 已经加载
            
        try:
            # 优先使用配置中指定的本地VQA模型路径
            model_path = settings.VISION_VQA_MODEL_PATH or self.vqa_model_name
            logger.info(f"Loading BLIP VQA model: {model_path}")
            
            try:
                # 尝试从本地加载
                self.vqa_model = BlipForQuestionAnswering.from_pretrained(model_path).to(self.device)
                
                # VQA模型使用相同的处理器或加载新的处理器
                if self.processor is None:
                    self.processor = BlipProcessor.from_pretrained(model_path)
                    
                logger.info("BLIP VQA model loaded successfully from local path")
                
            except Exception as local_err:
                # 如果本地加载失败，尝试从HuggingFace加载
                if settings.VISION_VQA_MODEL_PATH:
                    logger.warning(f"Failed to load local VQA model from {model_path}, trying HuggingFace: {local_err}")
                
                self.vqa_model = BlipForQuestionAnswering.from_pretrained(self.vqa_model_name).to(self.device)
                
                if self.processor is None:
                    self.processor = BlipProcessor.from_pretrained(self.vqa_model_name)
                
                logger.info("BLIP VQA model loaded from HuggingFace successfully")
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to load BLIP VQA model: {str(e)}")
            return False  # 加载失败
    
    def generate_caption(
        self, 
        image: Union[str, Path, np.ndarray], 
        max_length: int = 30,
        num_beams: int = 4,
        min_length: int = 5,
        temperature: float = 1.0,
        conditional_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        生成图像描述
        
        Args:
            image: 图像路径或numpy数组
            max_length: 生成描述的最大长度
            num_beams: beam search的beam数量
            min_length: 生成描述的最小长度
            temperature: 生成多样性参数，越高则结果越多样
            conditional_prompt: 条件提示词（可选），如"一张图片，展示了"
        
        Returns:
            Dict[str, Any]: 包含生成描述和额外信息的字典
        """
        if self.model is None or self.processor is None:
            logger.error("BLIP model not properly initialized")
            return {"success": False, "error": "BLIP model not initialized"}
        
        try:
            # 预处理图像
            if isinstance(image, (str, Path)):
                # 读取图像
                raw_image = read_image(image)
                raw_image = convert_to_rgb(raw_image)
                # 转换为PIL格式供BLIP处理器使用
                pil_image = Image.fromarray(raw_image)
            elif isinstance(image, np.ndarray):
                # 如果已经是numpy数组，确保是RGB格式
                if len(image.shape) == 3 and image.shape[2] == 3:
                    pil_image = Image.fromarray(image)
                else:
                    logger.error(f"Invalid image format: {image.shape}")
                    return {"success": False, "error": "Invalid image format"}
            else:
                logger.error(f"Unsupported image type: {type(image)}")
                return {"success": False, "error": "Unsupported image type"}
            
            # 使用BLIP处理器处理图像和文本
            inputs = self.processor(
                pil_image, 
                text=conditional_prompt if conditional_prompt else "",  # 可以为空
                return_tensors="pt"
            ).to(self.device)
            
            logger.debug("Running BLIP caption generation")
            # 生成描述
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    min_length=min_length,
                    temperature=temperature
                )
            
            # 解码生成的标记为文本
            generated_caption = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            logger.info(f"Caption generated successfully: {generated_caption[:30]}...")
            return {
                "success": True,
                "caption": generated_caption,
                "parameters": {
                    "max_length": max_length,
                    "num_beams": num_beams,
                    "min_length": min_length,
                    "temperature": temperature,
                    "conditional_prompt": conditional_prompt
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating caption: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def answer_question(
        self, 
        image: Union[str, Path, np.ndarray], 
        question: str,
        max_length: int = 30,
        num_beams: int = 4,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """
        针对图像回答问题 (VQA功能)
        
        Args:
            image: 图像路径或numpy数组
            question: 关于图像的问题
            max_length: 生成回答的最大长度
            num_beams: beam search的beam数量
            temperature: 生成多样性参数
            
        Returns:
            Dict[str, Any]: 包含回答和额外信息的字典
        """
        # 确保VQA模型已加载
        if not self._load_vqa_model():
            return {"success": False, "error": "Failed to load VQA model"}
        
        try:
            # 预处理图像
            if isinstance(image, (str, Path)):
                raw_image = read_image(image)
                raw_image = convert_to_rgb(raw_image)
                pil_image = Image.fromarray(raw_image)
            elif isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 3:
                    pil_image = Image.fromarray(image)
                else:
                    return {"success": False, "error": "Invalid image format"}
            else:
                return {"success": False, "error": "Unsupported image type"}
            
            # 使用BLIP处理器处理图像和问题
            inputs = self.processor(pil_image, question, return_tensors="pt").to(self.device)
            
            logger.debug(f"Running VQA for question: {question}")
            # 生成回答
            with torch.no_grad():
                outputs = self.vqa_model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    temperature=temperature
                )
            
            # 解码生成的标记为文本
            answer = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            logger.info(f"Question answered: Q: {question}, A: {answer}")
            return {
                "success": True,
                "question": question,
                "answer": answer,
                "parameters": {
                    "max_length": max_length,
                    "num_beams": num_beams,
                    "temperature": temperature
                }
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def analyze_image_content(
        self, 
        image: Union[str, Path, np.ndarray]
    ) -> Dict[str, Any]:
        """
        分析图像内容，组合多种信息
        
        Args:
            image: 图像路径或numpy数组
            
        Returns:
            Dict[str, Any]: 包含分析结果的字典
        """
        # 生成基本描述
        caption_result = self.generate_caption(image)
        if not caption_result["success"]:
            return caption_result
        
        # 尝试回答一些常见问题来丰富分析
        questions = [
            "What type of image is this?",
            "Is this image from a textbook or educational material?",
            "Are there any diagrams or charts in this image?"
        ]
        
        qa_results = {}
        for question in questions:
            qa_result = self.answer_question(image, question)
            if qa_result["success"]:
                qa_results[question] = qa_result["answer"]
        
        # 组合结果
        return {
            "success": True,
            "caption": caption_result["caption"],
            "analysis": qa_results,
            "image_type": self._determine_image_type(caption_result["caption"], qa_results)
        }
    
    def _determine_image_type(self, caption: str, qa_results: Dict[str, str]) -> str:
        """
        基于生成的描述和问答结果确定图像类型
        
        Args:
            caption: 图像描述
            qa_results: 问答结果
            
        Returns:
            str: 图像类型
        """
        # 这里是简化实现，根据关键词匹配图像类型
        caption = caption.lower()
        
        # 检查是否包含图表相关关键词
        if any(word in caption for word in ["chart", "graph", "diagram", "plot"]):
            return "chart"
        
        # 检查是否包含公式相关关键词
        if any(word in caption for word in ["equation", "formula", "mathematical"]):
            return "formula"
            
        # 检查是否包含表格相关关键词
        if any(word in caption for word in ["table", "grid", "row", "column"]):
            return "table"
        
        # 检查问答结果
        if qa_results:
            qa_text = " ".join(qa_results.values()).lower()
            if any(word in qa_text for word in ["chart", "graph", "diagram", "plot"]):
                return "chart"
            if any(word in qa_text for word in ["equation", "formula", "mathematical"]):
                return "formula"
            if any(word in qa_text for word in ["table", "grid", "row", "column"]):
                return "table"
            if any(word in qa_text for word in ["textbook", "educational", "academic", "lecture"]):
                return "educational_material"
                
        # 默认为普通图像
        return "general_image"

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "vqa_model_name": self.vqa_model_name,
            "vqa_model_loaded": self.vqa_model is not None,
            "captioning_model_loaded": self.model is not None and self.processor is not None,
        }