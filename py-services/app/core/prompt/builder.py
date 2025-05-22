from typing import Dict, Any, List, Optional, Union
import time
from pathlib import Path
import numpy as np

from app.utils.logger import logger
from app.core.prompt.templates import prompt_template_manager, TemplateType
from app.core.vision.service import vision_service
from app.core.ocr.service import ocr_service

class PromptBuilder:
    """
    Prompt构建器
    负责根据图像内容和文本分析生成适合多模态大语言模型的提示词
    """
    
    def __init__(self, template_manager=None):
        """
        初始化Prompt构建器
        
        Args:
            template_manager: 模板管理器实例，如果为None则使用默认实例
        """
        # 使用提供的模板管理器或默认单例
        self.template_manager = template_manager or prompt_template_manager
        logger.info("Prompt builder initialized")
    
    def build_prompt_from_image(
        self, 
        image: Union[str, Path, np.ndarray],
        content_type: Optional[str] = None,
        custom_template: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        从图像构建完整的提示词
        
        Args:
            image: 图像路径或numpy数组
            content_type: 内容类型，如果为None则自动检测
            custom_template: 自定义模板名称，如果提供则使用该模板
            additional_context: 附加的上下文信息，将合并到提示词变量中
            
        Returns:
            Dict[str, Any]: 构建结果，包含提示词和其他元数据
        """
        start_time = time.time()
        
        try:
            # 步骤1: 获取图像描述（使用vision服务）
            prompt_description = vision_service.describe_for_prompt(image)
            
            if not prompt_description["success"]:
                return {
                    "success": False,
                    "error": f"Failed to analyze image: {prompt_description.get('error', 'Unknown error')}"
                }
            
            # 步骤2: 如果未指定内容类型，使用检测到的类型
            if content_type is None:
                content_type = prompt_description["content_type"]
            
            # 步骤3: 执行OCR识别文本
            if isinstance(image, (str, Path)):
                ocr_result = ocr_service.process_image_file(image)
            else:
                ocr_result = ocr_service.process_image_data(image)
                
            if not ocr_result["success"]:
                return {
                    "success": False,
                    "error": f"Failed to perform OCR: {ocr_result.get('error', 'Unknown error')}"
                }
            
            # 步骤4: 准备模板变量
            template_vars = {
                "image_caption": prompt_description.get("caption", ""),
                "ocr_text": ocr_result.get("full_text", ""),
                "content_type": content_type
            }
            
            # 添加内容类型特定的额外信息
            template_vars.update(self._get_content_specific_info(
                content_type, 
                prompt_description
            ))
            
            # 合并附加上下文（如果有）
            if additional_context:
                template_vars.update(additional_context)
            
            # 步骤5: 选择适当的模板
            if custom_template:
                formatted_prompt = self.template_manager.format_template(
                    custom_template, 
                    **template_vars
                )
            else:
                formatted_prompt = self.template_manager.format_template(
                    content_type, 
                    **template_vars
                )
            
            # 计算处理时间
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "prompt": formatted_prompt,
                "content_type": content_type,
                "processing_time": processing_time,
                "template_variables": template_vars,
                "ocr_stats": {
                    "text_length": len(ocr_result.get("full_text", "")),
                    "paragraph_count": ocr_result.get("paragraph_count", 0)
                },
                "vision_info": {
                    "caption": prompt_description.get("caption", ""),
                    "focus_points": prompt_description.get("focus_points", {})
                }
            }
            
        except Exception as e:
            logger.error(f"Error building prompt from image: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _get_content_specific_info(
        self, 
        content_type: str, 
        vision_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        根据内容类型获取特定的附加信息
        
        Args:
            content_type: 内容类型
            vision_data: 图像分析数据
            
        Returns:
            Dict[str, Any]: 特定于内容类型的附加信息
        """
        additional_info = {}
        
        # 根据内容类型添加特定信息
        if content_type == "formula":
            additional_info["formula_description"] = vision_data.get("caption", "")
            additional_info["formula_focus"] = vision_data.get("focus_points", {}).get(
                "What is the most important element in this image?", ""
            )
            
        elif content_type in ["chart", "diagram"]:
            focus_points = vision_data.get("focus_points", {})
            additional_info["diagram_topic"] = focus_points.get(
                "What is the main topic of this image?", ""
            )
            
        elif content_type == "table":
            additional_info["table_description"] = vision_data.get("caption", "")
            # 表格行列信息可能需要从OCR结果中进一步分析
            
        return additional_info
    
    def build_prompt_from_text_and_caption(
        self, 
        ocr_text: str, 
        image_caption: str,
        content_type: str = "general",
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        从OCR文本和图像描述构建提示词（不需要图像）
        
        Args:
            ocr_text: OCR提取的文本
            image_caption: 图像描述
            content_type: 内容类型
            additional_context: 附加的上下文信息
            
        Returns:
            Dict[str, Any]: 构建结果，包含提示词和其他元数据
        """
        try:
            # 准备模板变量
            template_vars = {
                "image_caption": image_caption,
                "ocr_text": ocr_text,
                "content_type": content_type
            }
            
            # 合并附加上下文（如果有）
            if additional_context:
                template_vars.update(additional_context)
            
            # 选择和格式化模板
            formatted_prompt = self.template_manager.format_template(
                content_type, 
                **template_vars
            )
            
            return {
                "success": True,
                "prompt": formatted_prompt,
                "content_type": content_type,
                "template_variables": template_vars
            }
            
        except Exception as e:
            logger.error(f"Error building prompt from text and caption: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def enhance_prompt_with_context(
        self, 
        base_prompt: str,
        context_info: Dict[str, Any]
    ) -> str:
        """
        使用上下文信息增强已有提示词
        
        Args:
            base_prompt: 基础提示词
            context_info: 上下文信息
            
        Returns:
            str: 增强后的提示词
        """
        # 从上下文中提取关键信息
        subject = context_info.get("subject", "")
        educational_level = context_info.get("educational_level", "")
        focus = context_info.get("focus", "")
        
        # 构建上下文前缀
        context_prefix = ""
        if subject:
            context_prefix += f"This content is about {subject}. "
        if educational_level:
            context_prefix += f"It's suitable for {educational_level} level. "
        if focus:
            context_prefix += f"Focus on {focus}. "
        
        # 如果有前缀，添加到提示词前面
        if context_prefix:
            enhanced_prompt = f"{context_prefix}\n\n{base_prompt}"
            logger.info(f"Enhanced prompt with context information")
            return enhanced_prompt
        
        # 否则返回原始提示词
        return base_prompt
    
    def build_multi_part_prompt(
        self, 
        parts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        构建多部分提示词（适用于复杂图像，如多图表或多段落）
        
        Args:
            parts: 提示词部分列表，每个部分包含text, caption, type等
            
        Returns:
            Dict[str, Any]: 构建结果
        """
        try:
            combined_prompt = "I will analyze this content in multiple parts:\n\n"
            
            for i, part in enumerate(parts):
                # 获取部分标题
                part_title = part.get("title", f"Part {i+1}")
                ocr_text = part.get("text", "")
                caption = part.get("caption", "")
                content_type = part.get("type", "general")
                
                # 构建此部分的提示词
                part_result = self.build_prompt_from_text_and_caption(
                    ocr_text=ocr_text,
                    image_caption=caption,
                    content_type=content_type
                )
                
                if not part_result["success"]:
                    continue
                    
                # 添加到组合提示词
                combined_prompt += f"--- {part_title} ---\n"
                combined_prompt += part_result["prompt"]
                combined_prompt += "\n\n"
                
            combined_prompt += "\nPlease integrate information from all parts in your response."
            
            return {
                "success": True,
                "prompt": combined_prompt,
                "part_count": len(parts)
            }
            
        except Exception as e:
            logger.error(f"Error building multi-part prompt: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_available_templates(self) -> Dict[str, str]:
        """
        获取所有可用的提示词模板
        
        Returns:
            Dict[str, str]: 模板名称及内容
        """
        return self.template_manager.list_templates()


# 创建单例实例，方便直接导入使用
prompt_builder = PromptBuilder()