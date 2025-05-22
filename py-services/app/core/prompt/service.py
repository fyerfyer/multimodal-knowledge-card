from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import time
import numpy as np

from app.utils.logger import logger
from app.core.prompt.templates import prompt_template_manager, TemplateType
from app.core.prompt.builder import prompt_builder
from app.core.prompt.multimodal_fusion import multimodal_fusion
from app.core.prompt.optimizer import prompt_optimizer


class PromptService:
    """
    提示词服务
    整合模板管理、提示词构建、多模态融合和优化功能，提供统一的提示词生成服务
    """
    
    def __init__(
        self,
        template_manager=None,
        builder=None,
        fusion=None,
        optimizer=None
    ):
        """
        初始化提示词服务
        
        Args:
            template_manager: 模板管理器实例，如果为None则使用默认实例
            builder: 提示词构建器实例，如果为None则使用默认实例
            fusion: 多模态融合实例，如果为None则使用默认实例
            optimizer: 提示词优化器实例，如果为None则使用默认实例
        """
        # 使用提供的组件实例或默认单例
        self.template_manager = template_manager or prompt_template_manager
        self.builder = builder or prompt_builder
        self.fusion = fusion or multimodal_fusion
        self.optimizer = optimizer or prompt_optimizer
        logger.info("Prompt service initialized")
    
    def create_prompt_from_image(
        self,
        image: Union[str, Path, np.ndarray],
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        从图像创建提示词，包含完整的处理流程
        
        Args:
            image: 图像路径或numpy数组
            options: 可选配置，包含以下可能的键：
                - content_type: 内容类型，如果未指定则自动检测
                - template_type: 模板类型，如果未指定则根据内容类型选择
                - optimize: 是否优化提示词
                - optimize_rules: 要应用的优化规则列表
                - target_length: 目标提示词长度
                - additional_context: 附加上下文信息
                
        Returns:
            Dict[str, Any]: 包含生成的提示词和相关元数据的字典
        """
        try:
            start_time = time.time()
            
            # 提取选项，使用默认值
            options = options or {}
            content_type = options.get("content_type")
            template_type = options.get("template_type")
            optimize = options.get("optimize", False)
            optimize_rules = options.get("optimize_rules")
            target_length = options.get("target_length")
            additional_context = options.get("additional_context", {})
            
            # 1. 多模态融合处理
            logger.info(f"Processing image with multimodal fusion")
            fusion_result = self.fusion.fuse_content(
                image=image,
                include_vision=True,
                include_ocr=True
            )
            
            if not fusion_result["success"]:
                return {
                    "success": False,
                    "error": fusion_result.get("error", "Failed in multimodal fusion stage")
                }
            
            # 确定内容类型（如果未指定）
            if content_type is None:
                content_type = fusion_result["content_type"]
                
            # 2. 基于融合结果构建提示词
            logger.info(f"Building prompt for content type: {content_type}")
            
            # 准备构建提示词的参数
            build_params = {
                "ocr_text": fusion_result.get("ocr_text", ""),
                "image_caption": fusion_result.get("caption", ""),
                "content_type": content_type,
                "additional_context": additional_context
            }
            
            # 根据内容类型添加特定信息
            if content_type in ["chart", "diagram"] and "diagram_type" in fusion_result:
                build_params["additional_context"]["diagram_type"] = fusion_result["diagram_type"]
                
            if content_type == "formula" and "formula_text" in fusion_result:
                build_params["additional_context"]["formula_text"] = fusion_result["formula_text"]
                
            if content_type == "table" and "estimated_rows" in fusion_result:
                build_params["additional_context"]["table_rows"] = fusion_result["estimated_rows"]
            
            # 构建提示词
            prompt_result = self.builder.build_prompt_from_text_and_caption(
                **build_params
            )
            
            if not prompt_result["success"]:
                return {
                    "success": False,
                    "error": prompt_result.get("error", "Failed to build prompt")
                }
            
            prompt = prompt_result["prompt"]
            
            # 3. 如果需要，优化提示词
            if optimize:
                logger.info("Optimizing prompt")
                optimization_context = {"content_type": content_type}
                
                if target_length:
                    optimization_context["target_length"] = target_length
                    
                optimize_result = self.optimizer.optimize(
                    prompt=prompt,
                    context=optimization_context,
                    apply_rules=optimize_rules,
                    target_length=target_length
                )
                
                if optimize_result["success"]:
                    prompt = optimize_result["optimized_prompt"]
            
            # 计算处理时间
            processing_time = time.time() - start_time
            
            # 构建返回结果
            result = {
                "success": True,
                "prompt": prompt,
                "content_type": content_type,
                "processing_time": processing_time,
                "metadata": {
                    "ocr_text_length": len(fusion_result.get("ocr_text", "")),
                    "caption": fusion_result.get("caption", ""),
                    "vision_included": fusion_result.get("vision_included", False),
                    "ocr_included": fusion_result.get("ocr_included", False)
                }
            }
            
            # 如果优化了提示词，添加优化信息
            if optimize and optimize_result["success"]:
                result["optimization"] = {
                    "applied_rules": optimize_result["applied_rules"],
                    "stats": optimize_result["stats"]
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating prompt from image: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def create_text_prompt(
        self,
        text: str,
        template_type: Union[str, TemplateType] = TemplateType.GENERAL,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        从纯文本创建提示词
        
        Args:
            text: 输入文本
            template_type: 模板类型，可以是字符串或TemplateType枚举
            options: 可选配置
                - optimize: 是否优化提示词
                - optimize_rules: 要应用的优化规则列表
                - target_length: 目标提示词长度
                - variables: 模板变量
                
        Returns:
            Dict[str, Any]: 包含生成的提示词和相关元数据的字典
        """
        try:
            start_time = time.time()
            
            # 提取选项，使用默认值
            options = options or {}
            optimize = options.get("optimize", False)
            optimize_rules = options.get("optimize_rules")
            target_length = options.get("target_length")
            variables = options.get("variables", {})
            
            # 准备模板变量
            template_vars = {
                "text": text,
                "ocr_text": text,  # 兼容现有模板
                "image_caption": ""  # 兼容现有模板
            }
            
            # 合并自定义变量
            template_vars.update(variables)
            
            # 格式化模板
            prompt = self.template_manager.format_template(template_type, **template_vars)
            
            # 如果需要，优化提示词
            if optimize:
                logger.info("Optimizing text prompt")
                
                # 准备优化上下文
                opt_context = {}
                if isinstance(template_type, TemplateType):
                    opt_context["content_type"] = template_type.value
                else:
                    opt_context["content_type"] = template_type
                
                optimize_result = self.optimizer.optimize(
                    prompt=prompt,
                    context=opt_context,
                    apply_rules=optimize_rules,
                    target_length=target_length
                )
                
                if optimize_result["success"]:
                    prompt = optimize_result["optimized_prompt"]
            
            # 计算处理时间
            processing_time = time.time() - start_time
            
            # 构建返回结果
            result = {
                "success": True,
                "prompt": prompt,
                "template_type": template_type.value if isinstance(template_type, TemplateType) else template_type,
                "processing_time": processing_time,
            }
            
            # 如果优化了提示词，添加优化信息
            if optimize and optimize_result["success"]:
                result["optimization"] = {
                    "applied_rules": optimize_result["applied_rules"],
                    "stats": optimize_result["stats"]
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating text prompt: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def enhance_prompt(
        self, 
        prompt: str, 
        enhancement_type: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        增强现有提示词（重写、优化或添加上下文）
        
        Args:
            prompt: 原始提示词
            enhancement_type: 增强类型，可选值：
                - "optimize": 应用优化规则
                - "rewrite": 按指定风格重写
                - "context": 添加上下文信息
            options: 增强选项，根据增强类型不同而不同
                - optimize: optimize_rules, target_length, context
                - rewrite: style (concise/detailed/technical/educational)
                - context: context_info (字典，包含subject, educational_level, focus等)
                
        Returns:
            Dict[str, Any]: 包含增强后的提示词和相关元数据的字典
        """
        try:
            start_time = time.time()
            options = options or {}
            
            if enhancement_type == "optimize":
                # 优化提示词
                optimize_rules = options.get("optimize_rules")
                target_length = options.get("target_length")
                context = options.get("context", {})
                
                result = self.optimizer.optimize(
                    prompt=prompt,
                    context=context,
                    apply_rules=optimize_rules,
                    target_length=target_length
                )
                
                if result["success"]:
                    return {
                        "success": True,
                        "enhanced_prompt": result["optimized_prompt"],
                        "enhancement_type": "optimize",
                        "processing_time": time.time() - start_time,
                        "enhancement_details": {
                            "applied_rules": result["applied_rules"],
                            "stats": result["stats"]
                        }
                    }
                else:
                    return {
                        "success": False,
                        "error": result.get("error", "Failed to optimize prompt")
                    }
                    
            elif enhancement_type == "rewrite":
                # 重写提示词
                style = options.get("style", "concise")
                
                result = self.optimizer.rewrite_prompt(prompt=prompt, style=style)
                
                if result["success"]:
                    return {
                        "success": True,
                        "enhanced_prompt": result["rewritten_prompt"],
                        "enhancement_type": "rewrite",
                        "processing_time": time.time() - start_time,
                        "enhancement_details": {
                            "style": style,
                            "style_description": result.get("style_description", "")
                        }
                    }
                else:
                    return {
                        "success": False,
                        "error": result.get("error", "Failed to rewrite prompt")
                    }
                    
            elif enhancement_type == "context":
                # 添加上下文信息
                context_info = options.get("context_info", {})
                
                enhanced_prompt = self.builder.enhance_prompt_with_context(
                    base_prompt=prompt,
                    context_info=context_info
                )
                
                return {
                    "success": True,
                    "enhanced_prompt": enhanced_prompt,
                    "enhancement_type": "context",
                    "processing_time": time.time() - start_time,
                    "enhancement_details": {
                        "context_keys": list(context_info.keys())
                    }
                }
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown enhancement type: {enhancement_type}"
                }
                
        except Exception as e:
            logger.error(f"Error enhancing prompt: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def create_multi_part_prompt(
        self,
        parts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        创建多部分提示词，适用于分析复杂内容
        
        Args:
            parts: 提示词部分列表，每个部分是一个字典，包含：
                - text: 文本内容
                - caption: 图像描述（可选）
                - type: 内容类型（可选，默认为"general"）
                - title: 部分标题（可选）
                
        Returns:
            Dict[str, Any]: 包含生成的多部分提示词和相关元数据的字典
        """
        try:
            start_time = time.time()
            
            # 验证部分
            if not parts or not isinstance(parts, list):
                return {
                    "success": False,
                    "error": "Parts must be a non-empty list"
                }
            
            # 使用builder创建多部分提示词
            result = self.builder.build_multi_part_prompt(parts)
            
            if not result["success"]:
                return {
                    "success": False,
                    "error": result.get("error", "Failed to build multi-part prompt")
                }
            
            # 添加处理时间
            processing_time = time.time() - start_time
            result["processing_time"] = processing_time
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating multi-part prompt: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def manage_templates(
        self,
        action: str,
        template_name: str,
        template_content: Optional[str] = None,
        file_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        管理提示词模板（添加、删除、保存）
        
        Args:
            action: 操作类型，可选值：add, remove, save
            template_name: 模板名称
            template_content: 模板内容，仅在添加模板时需要
            file_path: 保存模板的文件路径，仅在保存模板时需要
            
        Returns:
            Dict[str, Any]: 操作结果
        """
        try:
            if action == "add":
                if not template_content:
                    return {"success": False, "error": "Template content required for add action"}
                
                self.template_manager.add_template(template_name, template_content)
                
                return {
                    "success": True,
                    "action": "add",
                    "template_name": template_name,
                    "message": f"Template '{template_name}' added successfully"
                }
                
            elif action == "remove":
                result = self.template_manager.remove_template(template_name)
                
                if result:
                    return {
                        "success": True,
                        "action": "remove",
                        "template_name": template_name,
                        "message": f"Template '{template_name}' removed successfully"
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Template '{template_name}' not found"
                    }
                    
            elif action == "save":
                result = self.template_manager.save_template_to_file(template_name, file_path)
                
                if result:
                    return {
                        "success": True,
                        "action": "save",
                        "template_name": template_name,
                        "file_path": str(file_path) if file_path else "default path",
                        "message": f"Template '{template_name}' saved successfully"
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Failed to save template '{template_name}'"
                    }
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown action: {action}"
                }
                
        except Exception as e:
            logger.error(f"Error managing templates: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_content_specific_template(self, content_type: str) -> Dict[str, Any]:
        """
        获取特定内容类型的模板
        
        Args:
            content_type: 内容类型
            
        Returns:
            Dict[str, Any]: 包含模板的字典
        """
        try:
            template = self.template_manager.get_template_for_content_type(content_type)
            
            return {
                "success": True,
                "content_type": content_type,
                "template": template
            }
            
        except Exception as e:
            logger.error(f"Error getting template for content type: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_available_templates(self) -> Dict[str, Any]:
        """
        获取所有可用的模板
        
        Returns:
            Dict[str, Any]: 包含模板字典的字典
        """
        try:
            templates = self.template_manager.list_templates()
            
            return {
                "success": True,
                "template_count": len(templates),
                "templates": templates
            }
            
        except Exception as e:
            logger.error(f"Error getting available templates: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_service_info(self) -> Dict[str, Any]:
        """
        获取服务信息
        
        Returns:
            Dict[str, Any]: 服务信息
        """
        return {
            "service": "Prompt Service",
            "components": {
                "template_manager": "PromptTemplateManager",
                "builder": "PromptBuilder",
                "fusion": "MultiModalFusion",
                "optimizer": "PromptOptimizer"
            },
            "available_enhancements": ["optimize", "rewrite", "context"],
            "available_rewrite_styles": ["concise", "detailed", "technical", "educational"],
            "template_count": len(self.template_manager.list_templates())
        }


# 创建单例实例，方便直接导入使用
prompt_service = PromptService()