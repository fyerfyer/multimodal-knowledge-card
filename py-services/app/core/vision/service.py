from typing import Dict, Any, Union, List, Optional, Tuple
import time
from pathlib import Path
import numpy as np

from app.utils.logger import logger
from app.config.settings import settings
from app.core.vision.blip import BLIPModel
from app.core.vision.captioner import captioner, ImageCaptioner
from app.core.vision.classifier import classifier, ContentClassifier, ContentType
from app.core.ocr.service import ocr_service


class VisionService:
    """
    图像理解服务
    整合多种视觉分析功能，提供图像内容理解的统一接口
    """
    
    def __init__(
        self,
        captioner_instance: Optional[ImageCaptioner] = None,
        classifier_instance: Optional[ContentClassifier] = None,
        blip_model_name: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        初始化图像理解服务
        
        Args:
            captioner_instance: 图像描述生成器实例，如果为None则使用默认实例
            classifier_instance: 内容分类器实例，如果为None则使用默认实例
            blip_model_name: BLIP模型名称或路径，如果为None则使用配置
            device: 运行设备 ('cuda' 或 'cpu')，如果为None则自动选择
        """
        # 使用提供的实例或默认单例
        self.captioner = captioner_instance or captioner
        self.classifier = classifier_instance or classifier
        
        # 根据配置创建BLIP模型（可能在captioner和classifier中已经创建）
        # 但保留直接访问的能力以支持某些特殊操作
        self.blip_model_name = blip_model_name or settings.VISION_MODEL_PATH or "Salesforce/blip-image-captioning-base"
        self.device = device
        self.blip_model = None  # 延迟加载
        
        logger.info(f"Vision service initialized")
    
    def _ensure_blip_model(self) -> None:
        """
        确保BLIP模型已加载
        """
        if self.blip_model is None:
            logger.info(f"Initializing BLIP model on demand: {self.blip_model_name}")
            self.blip_model = BLIPModel(model_name=self.blip_model_name, device=self.device)
    
    def analyze_image(
        self, 
        image: Union[str, Path, np.ndarray],
        include_caption: bool = True,
        include_classification: bool = True,
        include_ocr: bool = False,
        detailed_analysis: bool = False
    ) -> Dict[str, Any]:
        """
        综合分析图像内容
        
        Args:
            image: 图像路径或numpy数组
            include_caption: 是否包含图像描述
            include_classification: 是否包含内容类型分类
            include_ocr: 是否包含OCR文本提取（需要OCR服务）
            detailed_analysis: 是否包含详细分析
            
        Returns:
            Dict[str, Any]: 包含分析结果的字典
        """
        try:
            start_time = time.time()
            
            results = {
                "success": True,
                "timestamp": start_time
            }
            
            # 图像描述生成
            caption_success = False
            if include_caption:
                caption_result = self.captioner.generate_caption(image)
                if caption_result["success"]:
                    caption_success = True
                    results["caption"] = caption_result["caption"]
                    
                    # 只有在需要详细分析时才包含教育内容分析
                    if detailed_analysis:
                        try:
                            edu_analysis = self.captioner.analyze_educational_content(image)
                            if edu_analysis["success"]:
                                results["educational_analysis"] = {
                                    "is_educational": "educational" in edu_analysis.get("subject_analysis", {}).values(),
                                    "subject": self._extract_subject_from_qa(edu_analysis.get("subject_analysis", {})),
                                    "education_level": self._extract_education_level(edu_analysis.get("subject_analysis", {}))
                                }
                        except Exception as e:
                            logger.warning(f"Detailed analysis failed but continuing: {str(e)}")
                            # Don't fail the whole request for this
                else:
                    # 修改错误处理逻辑
                    if include_caption:  
                        return {
                            "success": False,
                            "error": caption_result.get("error", "Failed to generate caption")
                        }
                    logger.warning(f"Caption generation failed: {caption_result.get('error', 'Unknown error')}")
            
            # 内容分类
            classification_success = False
            if include_classification:
                classification_result = self.classifier.classify_content(
                    image, 
                    detailed=detailed_analysis
                )
                
                if classification_result["success"]:
                    classification_success = True
                    results["content_type"] = classification_result["content_type"]
                    results["content_confidence"] = classification_result["confidence"]
                    
                    # 根据内容类型可能需要特殊处理
                    if detailed_analysis:
                        # 如果是教育图表，添加额外分析
                        if results["content_type"] in ["diagram", "chart", "formula"]:
                            diagram_info = self.classifier.detect_educational_diagram(image)
                            if diagram_info["success"]:
                                results["diagram_analysis"] = {
                                    "is_educational": diagram_info["is_educational"],
                                    "subject": diagram_info["subject"],
                                    "educational_level": diagram_info["educational_level"]
                                }
                else:
                    # 如果分类失败且是必须的，则整个分析也应该失败
                    if include_classification and not caption_success:
                        return {
                            "success": False,
                            "error": classification_result.get("error", "Failed to classify content")
                        }
                    logger.warning(f"Content classification failed: {classification_result.get('error', 'Unknown error')}")
            
            # 如果两个主要分析都失败了，返回失败结果
            if include_caption and include_classification and not (caption_success or classification_success):
                return {
                    "success": False, 
                    "error": "Both caption generation and content classification failed"
                }
                
            # OCR文本提取（可选）
            if include_ocr:
                try:
                    if isinstance(image, (str, Path)):
                        ocr_result = ocr_service.process_image_file(image)
                    else:  # numpy数组
                        ocr_result = ocr_service.process_image_data(image)
                        
                    if ocr_result["success"]:
                        results["ocr"] = {
                            "full_text": ocr_result["full_text"],
                            "paragraph_count": ocr_result["paragraph_count"],
                            "paragraphs": ocr_result["paragraphs"][:3] if detailed_analysis else []  # 仅在详细模式下包含段落内容
                        }
                except Exception as e:
                    logger.error(f"Error performing OCR: {str(e)}")
                    results["ocr_error"] = str(e)
            
            # 综合评估
            if caption_success and classification_success:
                results["summary"] = self._generate_summary(results)
            else:
                results["summary"] = ""
            
            # 处理时间
            processing_time = time.time() - start_time
            results["processing_time"] = processing_time
            
            logger.info(f"Image analysis completed in {processing_time:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def describe_for_prompt(
        self, 
        image: Union[str, Path, np.ndarray]
    ) -> Dict[str, Any]:
        """
        生成适合多模态Prompt的图像描述
        这个方法专门为多模态Prompt构建器提供更适合的输出格式
        
        Args:
            image: 图像路径或numpy数组
            
        Returns:
            Dict[str, Any]: 为Prompt构建优化的图像描述
        """
        try:
            # 首先执行一个基本分析
            basic_analysis = self.analyze_image(
                image, 
                include_caption=True, 
                include_classification=True,
                include_ocr=False
            )
            
            if not basic_analysis["success"]:
                return basic_analysis
            
            # 根据内容类型生成不同的描述风格
            content_type = basic_analysis.get("content_type", "unknown")
            
            # 针对不同类型的内容使用不同风格的描述
            if content_type == "chart" or content_type == "diagram":
                caption_result = self.captioner.generate_caption(image, style="diagram")
            elif content_type == "formula":
                caption_result = self.captioner.generate_caption(image, style="formula")
            elif content_type == "text":
                # 对于文本密集型图像，使用默认风格
                caption_result = basic_analysis.get("caption", "An image containing text")
            else:
                # 对于教育内容，尝试提供更详细的描述
                edu_analysis = self.captioner.analyze_educational_content(image)
                if edu_analysis["success"] and edu_analysis.get("educational_caption"):
                    caption_result = edu_analysis["educational_caption"]
                else:
                    caption_result = basic_analysis.get("caption", "")
            
            # 额外的主题和焦点分析
            focus_questions = [
                "What is the main topic of this image?",
                "What is the most important element in this image?"
            ]
            
            focus_analysis = {}
            self._ensure_blip_model()  # 确保模型已加载
            
            for question in focus_questions:
                qa_result = self.blip_model.answer_question(image, question)
                if qa_result["success"]:
                    focus_analysis[question] = qa_result["answer"]
            
            # 构建优化的Prompt描述
            return {
                "success": True,
                "content_type": content_type,
                "caption": caption_result if isinstance(caption_result, str) else caption_result["caption"],
                "focus_points": focus_analysis,
                "prompt_suggestion": self._generate_prompt_suggestion(content_type, basic_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error generating prompt description: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def analyze_image_batch(
        self, 
        images: List[Union[str, Path, np.ndarray]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        批量分析多个图像
        
        Args:
            images: 图像路径或numpy数组的列表
            **kwargs: 传递给analyze_image的参数
            
        Returns:
            List[Dict[str, Any]]: 每个图像的分析结果列表
        """
        results = []
        
        for idx, image in enumerate(images):
            try:
                logger.info(f"Processing image {idx+1}/{len(images)}")
                result = self.analyze_image(image, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing image {idx+1}: {str(e)}")
                results.append({
                    "success": False,
                    "error": str(e),
                    "image_index": idx
                })
        
        logger.info(f"Batch analysis completed for {len(images)} images")
        return results
    
    def compare_images(
        self, 
        image1: Union[str, Path, np.ndarray],
        image2: Union[str, Path, np.ndarray]
    ) -> Dict[str, Any]:
        """
        比较两个图像的内容相似性
        
        Args:
            image1: 第一个图像路径或numpy数组
            image2: 第二个图像路径或numpy数组
            
        Returns:
            Dict[str, Any]: 比较结果
        """
        # TODO: 实现后续可能需要的图像比较功能
        # 需要实现特征提取和比较逻辑
        logger.warning("Image comparison functionality not fully implemented yet")
        
        # 分析两个图像
        analysis1 = self.analyze_image(image1)
        analysis2 = self.analyze_image(image2)
        
        if not analysis1["success"] or not analysis2["success"]:
            return {"success": False, "error": "Failed to analyze one or both images"}
        
        # 简单比较内容类型
        same_content_type = analysis1.get("content_type") == analysis2.get("content_type")
        
        return {
            "success": True,
            "same_content_type": same_content_type,
            "content_type1": analysis1.get("content_type"),
            "content_type2": analysis2.get("content_type"),
            "caption1": analysis1.get("caption"),
            "caption2": analysis2.get("caption"),
            "note": "Full image comparison not implemented yet"
        }
    
    def get_service_info(self) -> Dict[str, Any]:
        """
        获取服务信息
        
        Returns:
            Dict[str, Any]: 服务信息
        """
        # 获取图像描述生成器信息
        captioner_info = {
            "available_styles": self.captioner.get_available_styles()
        }
        
        # 获取分类器信息
        classifier_info = {
            "confidence_threshold": self.classifier.confidence_threshold,
            "content_types": [ct.value for ct in ContentType]
        }
        
        return {
            "service": "Vision Analysis Service",
            "captioner": captioner_info,
            "classifier": classifier_info,
            "model_path": self.blip_model_name
        }
    
    def _extract_subject_from_qa(self, qa_results: Dict[str, str]) -> str:
        """
        从问答结果中提取学科信息
        
        Args:
            qa_results: 问答结果字典
            
        Returns:
            str: 提取的学科
        """
        subject = "general"
        
        # 尝试从"What subject is this educational material about?"问题的回答中提取
        subject_question = "What subject is this educational material about?"
        if subject_question in qa_results:
            answer = qa_results[subject_question].lower()
            
            # 常见学科关键词
            subjects = {
                "math": ["math", "mathematics", "algebra", "calculus", "geometry"],
                "physics": ["physics", "mechanics", "electromagnetism", "optics"],
                "chemistry": ["chemistry", "chemical", "molecule", "atom", "reaction"],
                "biology": ["biology", "cell", "organism", "anatomy", "genetic"],
                "computer science": ["computer", "programming", "algorithm", "data structure"],
                "history": ["history", "historical", "century", "ancient", "medieval"],
                "geography": ["geography", "map", "terrain", "climate", "continent"]
            }
            
            # 检查关键词匹配
            for subj, keywords in subjects.items():
                if any(keyword in answer for keyword in keywords):
                    subject = subj
                    break
        
        return subject
    
    def _extract_education_level(self, qa_results: Dict[str, str]) -> str:
        """
        从问答结果中提取教育级别
        
        Args:
            qa_results: 问答结果字典
            
        Returns:
            str: 提取的教育级别
        """
        level = "unknown"
        
        # 尝试从"What grade level or education level is this material for?"问题的回答中提取
        level_question = "What grade level or education level is this material for?"
        if level_question in qa_results:
            answer = qa_results[level_question].lower()
            
            # 教育级别关键词
            levels = {
                "elementary": ["elementary", "primary", "basic", "beginner"],
                "middle school": ["middle school", "junior high"],
                "high school": ["high school", "secondary"],
                "college": ["college", "undergraduate", "university", "bachelor"],
                "graduate": ["graduate", "master", "phd", "doctoral", "advanced"]
            }
            
            # 检查关键词匹配
            for lvl, keywords in levels.items():
                if any(keyword in answer for keyword in keywords):
                    level = lvl
                    break
        
        return level
    
    def _generate_summary(self, analysis_results: Dict[str, Any]) -> str:
        """
        根据分析结果生成摘要描述
        
        Args:
            analysis_results: 分析结果字典
            
        Returns:
            str: 摘要描述
        """
        content_type = analysis_results.get("content_type", "unknown")
        caption = analysis_results.get("caption", "")
        
        # 根据内容类型生成不同的摘要
        if content_type == "chart":
            return f"This image contains a chart or diagram. {caption}"
        elif content_type == "formula":
            return f"This image contains mathematical or scientific formula. {caption}"
        elif content_type == "table":
            return f"This image contains a table or grid. {caption}"
        elif content_type == "text":
            return f"This image contains text content. {caption}"
        elif content_type == "handwritten":
            return f"This image contains handwritten content. {caption}"
        elif content_type == "mixed":
            return f"This image contains mixed content types. {caption}"
        else:
            return caption
    
    def _generate_prompt_suggestion(self, content_type: str, analysis: Dict[str, Any]) -> str:
        """
        根据内容类型生成提示词建议
        
        Args:
            content_type: 内容类型
            analysis: 分析结果
            
        Returns:
            str: 提示词建议
        """
        caption = analysis.get("caption", "")
        
        if content_type == "chart" or content_type == "diagram":
            return f"A diagram showing {caption}. Please explain what this diagram illustrates."
        elif content_type == "formula":
            return f"A mathematical or scientific formula. Please explain what this formula represents."
        elif content_type == "table":
            return f"A table containing data or information. Please analyze the content of this table."
        elif content_type == "text":
            return f"Text content from an educational material. Please summarize the main points."
        elif content_type == "handwritten":
            return f"Handwritten notes or content. Please analyze and explain these notes."
        else:
            return f"An educational image. {caption}"


# 创建单例实例，方便直接导入使用
vision_service = VisionService()