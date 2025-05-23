import os
from typing import Dict, Any, Union, List, Optional
import time
from pathlib import Path
import numpy as np

from app.utils.logger import logger
from app.core.vision.blip import BLIPModel
from app.config.settings import settings

class ImageCaptioner:
    """
    图像描述生成器
    负责生成图像内容的自然语言描述
    """
    
    def __init__(
        self, 
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        caption_styles: Optional[Dict[str, str]] = None
    ):
        """
        初始化图像描述生成器
        
        Args:
            model_name: 模型名称或路径
            device: 运行设备 ('cuda' 或 'cpu')，如果为None则自动选择
            caption_styles: 可选的描述风格模板字典
        """
        # 如果未指定设备，则使用自动选择
        self.device = device

        # print(f"[DEBUG] Initializing ImageCaptioner")
        # print(f"[DEBUG] os.environ has VISION_MODEL_PATH: {'VISION_MODEL_PATH' in os.environ}")
        # print(f"[DEBUG] Settings dir: {dir(settings)}")
        
        # 使用配置的模型路径或默认路径
        model_path = model_name or settings.VISION_MODEL_PATH or "Salesforce/blip-image-captioning-base"
        if settings.USE_HF_MIRROR and os.environ.get('HF_ENDPOINT') is None:
            os.environ['HF_ENDPOINT'] = settings.HF_MIRROR
            logger.info(f"Using HuggingFace mirror for captioner: {settings.HF_MIRROR}")

        # 初始化BLIP模型
        self.blip_model = BLIPModel(model_name=model_path, device=self.device)
        
        # 定义不同的描述风格模板
        self.caption_styles = caption_styles or {
            "default": "",  # 默认风格，无条件提示词
            "detailed": "a detailed image of",  # 详细描述风格
            "educational": "an educational material showing",  # 教育内容风格
            "diagram": "a diagram illustrating",  # 图表描述风格
            "formula": "a mathematical formula representing",  # 公式描述风格
        }
        
        logger.info(f"ImageCaptioner initialized with model: {model_path}")
        
    def generate_caption(
        self, 
        image: Union[str, Path, np.ndarray],
        style: str = "default",
        max_length: int = 50,
        num_beams: int = 5,
        min_length: int = 5,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """
        生成图像描述
        
        Args:
            image: 图像路径或numpy数组
            style: 描述风格，对应caption_styles中的键
            max_length: 生成描述的最大长度
            num_beams: beam search的beam数量
            min_length: 生成描述的最小长度
            temperature: 生成多样性参数
            
        Returns:
            Dict[str, Any]: 包含描述结果的字典
        """
        try:
            # 获取选定风格的条件提示词
            conditional_prompt = self.caption_styles.get(style, self.caption_styles["default"])
            
            # 调用BLIP模型生成描述
            start_time = time.time()
            caption_result = self.blip_model.generate_caption(
                image=image,
                max_length=max_length,
                num_beams=num_beams,
                min_length=min_length,
                temperature=temperature,
                conditional_prompt=conditional_prompt
            )
            processing_time = time.time() - start_time
            
            # 如果生成成功，添加额外信息
            if caption_result["success"]:
                return {
                    "success": True,
                    "caption": caption_result["caption"],
                    "style": style,
                    "processing_time": processing_time,
                    "parameters": caption_result["parameters"]
                }
            else:
                logger.error(f"Failed to generate caption: {caption_result.get('error', 'Unknown error')}")
                return caption_result
                
        except Exception as e:
            logger.error(f"Error in generate_caption: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def generate_multi_style_captions(
        self, 
        image: Union[str, Path, np.ndarray],
        styles: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        使用多种风格生成图像描述
        
        Args:
            image: 图像路径或numpy数组
            styles: 要使用的风格列表，如果为None则使用所有可用风格
            **kwargs: 传递给generate_caption的其他参数
            
        Returns:
            Dict[str, Any]: 包含多种风格描述结果的字典
        """
        # 如果未指定风格，使用所有可用风格
        if styles is None:
            styles = list(self.caption_styles.keys())
        
        results = {}
        success_count = 0
        
        # 为每种风格生成描述
        for style in styles:
            result = self.generate_caption(image=image, style=style, **kwargs)
            results[style] = result
            if result["success"]:
                success_count += 1
        
        # 返回综合结果
        return {
            "success": success_count > 0,
            "styles_count": len(styles),
            "success_count": success_count,
            "captions": results
        }
    
    def analyze_educational_content(
        self, 
        image: Union[str, Path, np.ndarray],
        **kwargs
    ) -> Dict[str, Any]:
        """
        分析教育内容图像，针对教材、讲义等内容进行优化的描述生成
        
        Args:
            image: 图像路径或numpy数组
            **kwargs: 传递给generate_caption的其他参数
            
        Returns:
            Dict[str, Any]: 包含教育内容分析结果的字典
        """
        # 首先生成常规描述
        regular_caption = self.generate_caption(image=image, style="default", **kwargs)
        if not regular_caption["success"]:
            return regular_caption
            
        # 生成教育风格的描述
        edu_caption = self.generate_caption(image=image, style="educational", **kwargs)
        
        # 获取BLIP模型的VQA回答
        questions = [
            "What subject is this educational material about?",
            "Is this a textbook, lecture notes, or diagram?",
            "What grade level or education level is this material for?"
        ]
        
        qa_results = {}
        for question in questions:
            result = self.blip_model.answer_question(image, question)
            if result["success"]:
                qa_results[question] = result["answer"]
        
        # 返回组合结果
        return {
            "success": True,
            "general_caption": regular_caption["caption"],
            "educational_caption": edu_caption["caption"] if edu_caption["success"] else None,
            "subject_analysis": qa_results,
            "processing_time": regular_caption.get("processing_time", 0) + 
                             (edu_caption.get("processing_time", 0) if edu_caption["success"] else 0)
        }
        
    def add_caption_style(self, name: str, template: str) -> None:
        """
        添加新的描述风格模板
        
        Args:
            name: 风格名称
            template: 风格模板字符串
        """
        self.caption_styles[name] = template
        logger.info(f"Added caption style: {name}")
    
    def get_available_styles(self) -> Dict[str, str]:
        """
        获取所有可用的描述风格
        
        Returns:
            Dict[str, str]: 风格名称和模板的字典
        """
        return self.caption_styles


# 创建单例实例，方便直接导入使用
captioner = ImageCaptioner()