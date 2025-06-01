import json
import markdown
import time
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import numpy as np

from app.utils.logger import logger
from app.core.formatter.json_parser import llm_output_parser
from app.core.formatter.markdown import markdown_formatter
from app.core.formatter.quality import quality_controller
from app.core.models.inference import inference_service

class FormatterService:
    """
    卡片格式化服务
    整合解析、格式化和质量控制功能，提供完整的卡片处理服务
    """
    
    def __init__(
        self,
        parser=None,
        markdown_formatter=None,
        quality_controller=None
    ):
        """
        初始化格式化服务
        
        Args:
            parser: LLM输出解析器实例
            markdown_formatter: Markdown格式化器实例
            quality_controller: 质量控制器实例
        """
        # 使用提供的实例或默认单例
        self.parser = parser or llm_output_parser
        self.markdown_formatter = markdown_formatter or markdown_formatter
        self.quality_controller = quality_controller or quality_controller
        logger.info("Formatter service initialized")
    
    def process_llm_output(
        self, 
        llm_output: str,
        improve_quality: bool = True,
        format_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        处理LLM输出，转换为结构化知识卡片
        
        Args:
            llm_output: LLM生成的原始输出文本
            improve_quality: 是否尝试改进卡片质量
            format_options: 格式化选项
            
        Returns:
            Dict[str, Any]: 处理结果，包含结构化卡片和不同格式的输出
        """
        try:
            start_time = time.time()
            
            # 解析LLM输出为结构化卡片
            logger.info("Processing LLM output to create knowledge card")
            parse_result = self.parser.parse(llm_output)
            
            if not parse_result.get("success", False):
                logger.error(f"Failed to parse LLM output: {parse_result.get('error', 'Unknown error')}")
                return {
                    "success": False,
                    "error": parse_result.get('error', 'Failed to parse LLM output'),
                    "raw_output": llm_output
                }
            
            card_data = parse_result["card_data"]
            
            # 验证卡片质量
            validation = self.quality_controller.validate_card(card_data)
            
            # 如果需要并且有质量问题，尝试改进卡片
            improved_card = card_data
            improvement_info = None
            
            if improve_quality and not validation["valid"]:
                logger.info(f"Attempting to improve card quality (original score: {validation['quality_score']})")
                improve_result = self.quality_controller.improve_card(card_data)
                
                if improve_result.get("improved", False):
                    improved_card = improve_result["improved_card"]
                    improvement_info = {
                        "original_score": improve_result.get("original_score"),
                        "improved_score": improve_result.get("improved_score"),
                        "improvements": improve_result["improvements"],
                        "remaining_issues": improve_result.get("remaining_issues", [])
                    }
                    logger.info(f"Card quality improved: {improvement_info['original_score']} -> {improvement_info['improved_score']}")
            
            # 生成不同格式的输出
            format_options = format_options or {}
            formatted_outputs = self._generate_formatted_outputs(improved_card, format_options)
            
            # 计算处理时间
            processing_time = time.time() - start_time
            
            # 构建结果
            result = {
                "success": True,
                "processing_time": processing_time,
                "parsing": {
                    "success": parse_result["success"],
                    "format": parse_result.get("format", "unknown")
                },
                "quality": {
                    "valid": validation["valid"],
                    "score": validation["quality_score"],
                    "issues": validation.get("issues", []),
                },
                "card_data": improved_card,
                "formats": formatted_outputs
            }
            
            # 如果进行了改进，添加改进信息
            if improvement_info:
                result["improvements"] = improvement_info
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing LLM output: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "raw_output": llm_output
            }
    
    def format_card(
        self, 
        card_data: Dict[str, Any],
        target_format: str = "markdown",
        format_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        将卡片数据转换为指定格式
        
        Args:
            card_data: 知识卡片数据
            target_format: 目标格式，支持"json", "markdown", "text", "html"
            format_options: 格式化选项
            
        Returns:
            Dict[str, Any]: 格式化结果
        """
        try:
            format_options = format_options or {}
            
            if target_format == "json":
                return self._format_as_json(card_data, format_options)
            elif target_format == "markdown":
                return self._format_as_markdown(card_data, format_options)
            elif target_format == "text":
                return self._format_as_text(card_data, format_options)
            elif target_format == "html":
                return self._format_as_html(card_data, format_options)
            else:
                logger.error(f"Unsupported format: {target_format}")
                return {
                    "success": False,
                    "error": f"Unsupported format: {target_format}"
                }
                
        except Exception as e:
            logger.error(f"Error formatting card as {target_format}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def validate_and_improve_card(
        self, 
        card_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        验证并改进卡片质量
        
        Args:
            card_data: 知识卡片数据
            
        Returns:
            Dict[str, Any]: 验证和改进结果
        """
        try:
            # 验证卡片
            validation = self.quality_controller.validate_card(card_data)
            
            # 如果有质量问题，尝试改进
            if not validation["valid"]:
                improve_result = self.quality_controller.improve_card(card_data)
                
                return {
                    "success": True,
                    "validation": validation,
                    "improved": improve_result["improved"],
                    "original_card": card_data,
                    "improved_card": improve_result["improved_card"] if improve_result["improved"] else card_data,
                    "improvements": improve_result.get("improvements", []),
                    "remaining_issues": improve_result.get("remaining_issues", [])
                }
            else:
                # 卡片已经符合质量标准
                return {
                    "success": True,
                    "validation": validation,
                    "improved": False,
                    "card_data": card_data,
                    "message": "Card already meets quality standards"
                }
                
        except Exception as e:
            logger.error(f"Error validating and improving card: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def generate_card_from_prompt(
        self, 
        prompt: str,
        image: Optional[Union[str, bytes, Path, np.ndarray]] = None,
        model_type: Optional[str] = None,
        improve_quality: bool = True
    ) -> Dict[str, Any]:
        """
        通过提示词生成知识卡片，直接调用推理服务
        
        Args:
            prompt: 提示词
            image: 可选的图像输入
            model_type: 可选的模型类型
            improve_quality: 是否改进卡片质量
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        try:
            # 调用推理服务
            logger.info("Generating knowledge card using inference service")
            response = inference_service.generate_response(
                prompt=prompt,
                image=image,
                model_type=model_type
            )
            
            if not response.get("success", False):
                logger.error(f"Inference service error: {response.get('error', 'Unknown error')}")
                return {
                    "success": False,
                    "error": response.get("error", "Inference service failed"),
                    "inference_result": response
                }
            
            # 获取LLM生成的文本
            llm_output = response.get("text", "")
            
            if not llm_output.strip():
                logger.error("Inference service returned empty output")
                return {
                    "success": False,
                    "error": "Empty response from inference service",
                    "inference_result": response
                }
            
            # 处理LLM输出
            result = self.process_llm_output(
                llm_output=llm_output,
                improve_quality=improve_quality
            )
            
            # 添加推理元数据
            result["inference_metadata"] = {
                "model_name": response.get("model_name", "unknown"),
                "processing_time": response.get("processing_time", 0),
                "token_usage": response.get("usage", {})
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating card from prompt: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _generate_formatted_outputs(
        self, 
        card_data: Dict[str, Any],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        生成不同格式的输出
        
        Args:
            card_data: 知识卡片数据
            options: 格式化选项
            
        Returns:
            Dict[str, Any]: 不同格式的输出
        """
        formats = {}
        
        # JSON格式
        json_result = self._format_as_json(card_data, options)
        if json_result["success"]:
            formats["json"] = json_result["json"]
        
        # Markdown格式
        md_result = self._format_as_markdown(card_data, options)
        if md_result["success"]:
            formats["markdown"] = md_result["markdown"]
        
        # 纯文本格式
        text_result = self._format_as_text(card_data, options)
        if text_result["success"]:
            formats["text"] = text_result["text"]
        
        # HTML格式 (如果支持)
        if "html" in options and options["html"]:
            html_result = self._format_as_html(card_data, options)
            if html_result["success"]:
                formats["html"] = html_result["html"]
        
        return formats
    
    def _format_as_json(
        self, 
        card_data: Dict[str, Any],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        将卡片数据格式化为JSON
        
        Args:
            card_data: 知识卡片数据
            options: 格式化选项
            
        Returns:
            Dict[str, Any]: JSON格式化结果
        """
        try:
            # 默认格式化选项
            indent = options.get("indent", 2)
            pretty = options.get("pretty", True)
            
            # 将字典转换为JSON字符串
            if pretty:
                json_str = json.dumps(card_data, indent=indent, ensure_ascii=False)
            else:
                json_str = json.dumps(card_data, ensure_ascii=False)
                
            return {
                "success": True,
                "json": json_str,
                "format": "json"
            }
            
        except Exception as e:
            logger.error(f"Error formatting as JSON: {str(e)}")
            return {
                "success": False,
                "error": f"JSON formatting failed: {str(e)}",
                "format": "json"
            }
    
    def _format_as_markdown(
        self, 
        card_data: Dict[str, Any],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        将卡片数据格式化为Markdown
        
        Args:
            card_data: 知识卡片数据
            options: 格式化选项
            
        Returns:
            Dict[str, Any]: Markdown格式化结果
        """
        try:
            # 使用Markdown格式化器
            md_result = self.markdown_formatter.card_to_markdown(card_data)
            
            if not md_result.get("success", False):
                return {
                    "success": False,
                    "error": md_result.get("error", "Markdown formatting failed"),
                    "format": "markdown"
                }
                
            return {
                "success": True,
                "markdown": md_result["markdown"],
                "format": "markdown"
            }
            
        except Exception as e:
            logger.error(f"Error formatting as Markdown: {str(e)}")
            return {
                "success": False,
                "error": f"Markdown formatting failed: {str(e)}",
                "format": "markdown"
            }
    
    def _format_as_text(
        self, 
        card_data: Dict[str, Any],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        将卡片数据格式化为纯文本
        
        Args:
            card_data: 知识卡片数据
            options: 格式化选项
            
        Returns:
            Dict[str, Any]: 纯文本格式化结果
        """
        try:
            # 构建纯文本版本
            text_parts = []
            
            # 添加标题
            if "title" in card_data:
                title = card_data["title"]
                text_parts.append(title)
                text_parts.append("=" * len(title))
                text_parts.append("")
            
            # 添加摘要
            if "summary" in card_data and card_data["summary"]:
                text_parts.append(card_data["summary"])
                text_parts.append("")
            
            # 添加知识点
            if "key_points" in card_data and card_data["key_points"]:
                text_parts.append("Key Points:")
                text_parts.append("")
                
                for i, point in enumerate(card_data["key_points"], 1):
                    if isinstance(point, dict):
                        content = point.get("content", "")
                        formula = point.get("formula")
                        reference = point.get("reference")
                        
                        text_parts.append(f"{i}. {content}")
                        
                        if formula:
                            text_parts.append(f"   Formula: {formula}")
                            
                        if reference:
                            text_parts.append(f"   Reference: {reference}")
                    else:
                        text_parts.append(f"{i}. {point}")
                        
                text_parts.append("")
            
            # 添加练习题
            if "quiz" in card_data:
                quiz = card_data["quiz"]
                text_parts.append("Quiz:")
                text_parts.append("")
                
                if isinstance(quiz, dict):
                    if "question" in quiz:
                        text_parts.append(f"Question: {quiz['question']}")
                        
                    if "answer" in quiz:
                        text_parts.append(f"Answer: {quiz['answer']}")
                else:
                    text_parts.append(str(quiz))
            
            # 添加引用
            if "references" in card_data and card_data["references"]:
                text_parts.append("")
                text_parts.append("References:")
                
                refs = card_data["references"]
                if isinstance(refs, dict):
                    for key, value in refs.items():
                        text_parts.append(f"- {key}: {value}")
                else:
                    text_parts.append(f"- {refs}")
            
            # 合并文本
            text = "\n".join(text_parts)
            
            return {
                "success": True,
                "text": text,
                "format": "text"
            }
            
        except Exception as e:
            logger.error(f"Error formatting as text: {str(e)}")
            return {
                "success": False,
                "error": f"Text formatting failed: {str(e)}",
                "format": "text"
            }
    
    def _format_as_html(
        self, 
        card_data: Dict[str, Any],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        将卡片数据格式化为HTML (通过Markdown转换)
        
        Args:
            card_data: 知识卡片数据
            options: 格式化选项
            
        Returns:
            Dict[str, Any]: HTML格式化结果
        """
        try:
            # 先转换为Markdown
            md_result = self._format_as_markdown(card_data, options)
            if not md_result["success"]:
                return {
                    "success": False,
                    "error": md_result.get("error", "Markdown conversion failed"),
                    "format": "html"
                }
                
            markdown_content = md_result["markdown"]
            
            html_content = markdown.markdown(markdown_content, extensions=['tables', 'fenced_code'])
                
            html = f"""
<!DOCTYPE html>
<html>
<head>
  <title>{card_data.get("title", "Knowledge Card")}</title>
  <meta charset="UTF-8">
  <style>
    body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }}
    h1 {{ color: #2c3e50; }}
    .key-point {{ margin-bottom: 15px; }}
    .quiz {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 20px; }}
    .formula {{ font-family: monospace; background-color: #f0f0f0; padding: 2px 5px; }}
  </style>
</head>
<body>
  <div class="knowledge-card">
    {html_content}
  </div>
</body>
</html>
"""
           
            return {
                "success": True,
                "html": html,
                "format": "html"
            }
            
        except Exception as e:
            logger.error(f"Error formatting as HTML: {str(e)}")
            return {
                "success": False,
                "error": f"HTML formatting failed: {str(e)}",
                "format": "html"
            }


# 创建单例实例，方便直接导入使用
formatter_service = FormatterService()