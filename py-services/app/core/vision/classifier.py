from typing import Dict, Any, Union, Optional, Tuple
import time
from pathlib import Path
import numpy as np
from enum import Enum

from app.utils.logger import logger
from app.core.vision.blip import BLIPModel

class ContentType(Enum):
    """图像内容类型枚举"""
    UNKNOWN = "unknown"
    TEXT = "text"
    CHART = "chart"
    DIAGRAM = "diagram"
    FORMULA = "formula"
    TABLE = "table"
    PHOTO = "photo"
    HANDWRITTEN = "handwritten"
    MIXED = "mixed"


class ContentClassifier:
    """
    图像内容分类器
    识别图像内容类型（如文本、图表、公式等）
    """
    
    def __init__(
        self, 
        model_name: str = "Salesforce/blip-vqa-base",
        device: Optional[str] = None,
        confidence_threshold: float = 0.7,
    ):
        """
        初始化内容分类器
        
        Args:
            model_name: VQA模型名称或路径
            device: 运行设备 ('cuda' 或 'cpu')，如果为None则自动选择
            confidence_threshold: 分类置信度阈值
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        # 初始化BLIP模型（使用VQA功能）
        self.blip_model = BLIPModel(
            model_name="Salesforce/blip-image-captioning-base",
            device=self.device,
            vqa_model_name=model_name
        )
        
        # 针对不同内容类型的特征关键词
        self.content_keywords = {
            ContentType.TEXT: ["text", "paragraph", "article", "passage", "document"],
            ContentType.CHART: ["chart", "graph", "plot", "histogram", "pie chart", "bar chart"],
            ContentType.DIAGRAM: ["diagram", "flowchart", "illustration", "schematic", "circuit"],
            ContentType.FORMULA: ["formula", "equation", "mathematical", "algebra", "calculus"],
            ContentType.TABLE: ["table", "grid", "rows", "columns", "spreadsheet"],
            ContentType.PHOTO: ["photograph", "picture", "photo", "image", "realistic"],
            ContentType.HANDWRITTEN: ["handwritten", "notes", "handwriting", "written by hand"],
        }
        
        # 分类问题模板
        self.classification_questions = [
            "What type of content is shown in this image?",
            "Is this image a chart, diagram, formula, table, or plain text?",
            "Does this image contain mathematical formulas?",
            "Does this image show a chart or graph?",
            "Does this image contain a table or grid?",
            "Is this handwritten content or printed text?"
        ]
        
        logger.info(f"ContentClassifier initialized with model: {model_name}")
    
    def classify_content(
        self, 
        image: Union[str, Path, np.ndarray],
        detailed: bool = False
    ) -> Dict[str, Any]:
        """
        对图像内容进行分类
        
        Args:
            image: 图像路径或numpy数组
            detailed: 是否返回详细的分类信息
            
        Returns:
            Dict[str, Any]: 分类结果
        """
        try:
            start_time = time.time()
            
            # 生成图像描述以获取初步内容理解
            caption_result = self.blip_model.generate_caption(image)
            if not caption_result["success"]:
                return {"success": False, "error": "Failed to generate image caption"}
            
            caption = caption_result["caption"].lower()
            
            # 对图像提问以进一步确定内容类型
            qa_results = {}
            for question in self.classification_questions:
                result = self.blip_model.answer_question(image, question)
                if result["success"]:
                    qa_results[question] = result["answer"].lower()
            
            # 基于描述和问答进行内容类型判断
            content_type, confidence = self._determine_content_type(caption, qa_results)
            
            processing_time = time.time() - start_time
            
            result = {
                "success": True,
                "content_type": content_type.value,
                "confidence": confidence,
                "processing_time": processing_time,
            }
            
            # 如果需要详细信息，添加额外的分析结果
            if detailed:
                result.update({
                    "caption": caption_result["caption"],
                    "qa_results": qa_results,
                    "content_analysis": self._analyze_content_features(content_type, caption, qa_results)
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in classify_content: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _determine_content_type(
        self, 
        caption: str, 
        qa_results: Dict[str, str]
    ) -> Tuple[ContentType, float]:
        """
        根据描述和问答结果确定内容类型
        
        Args:
            caption: 图像描述
            qa_results: 问答结果
            
        Returns:
            Tuple[ContentType, float]: 内容类型和置信度
        """
        # 合并所有文本以进行分析
        combined_text = caption + " " + " ".join(qa_results.values())
        combined_text = combined_text.lower()
        
        # 计算每种内容类型的匹配分数
        type_scores = {}
        for content_type, keywords in self.content_keywords.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            normalized_score = score / len(keywords) if keywords else 0
            type_scores[content_type] = normalized_score
        
        # 特殊规则处理
        # 1. 检测公式优先级高
        if type_scores[ContentType.FORMULA] > 0.3 or "formula" in combined_text or "equation" in combined_text:
            type_scores[ContentType.FORMULA] += 0.2
            
        # 2. 图表/表格检测特殊处理
        if any(keyword in combined_text for keyword in ["chart", "graph", "plot"]):
            type_scores[ContentType.CHART] += 0.3
            
        if any(keyword in combined_text for keyword in ["table", "grid", "row", "column"]):
            type_scores[ContentType.TABLE] += 0.3
        
        # 3. 混合内容检测
        content_types_present = sum(1 for score in type_scores.values() if score > 0.2)
        if content_types_present > 1:
            type_scores[ContentType.MIXED] = 0.7  # 假设有多种内容类型存在
        
        # 找出得分最高的内容类型
        if not type_scores:
            return ContentType.UNKNOWN, 0.0
            
        best_type = max(type_scores, key=type_scores.get)
        confidence = type_scores[best_type]
        
        # 如果置信度低于阈值，则归类为未知
        if confidence < self.confidence_threshold:
            return ContentType.UNKNOWN, confidence
            
        return best_type, confidence
    
    def _analyze_content_features(
        self, 
        content_type: ContentType, 
        caption: str, 
        qa_results: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        分析内容特征，根据内容类型提取相关信息
        
        Args:
            content_type: 内容类型
            caption: 图像描述
            qa_results: 问答结果
            
        Returns:
            Dict[str, Any]: 内容特征分析结果
        """
        features = {"type_specific_info": {}}
        
        # 根据内容类型进行特定分析
        if content_type == ContentType.CHART:
            # 分析图表类型
            chart_type = "unknown"
            if any(x in caption.lower() for x in ["bar", "histogram"]):
                chart_type = "bar_chart"
            elif any(x in caption.lower() for x in ["line", "trend"]):
                chart_type = "line_chart"
            elif any(x in caption.lower() for x in ["pie", "circle"]):
                chart_type = "pie_chart"
            elif any(x in caption.lower() for x in ["scatter"]):
                chart_type = "scatter_plot"
                
            features["type_specific_info"]["chart_type"] = chart_type
            
        elif content_type == ContentType.FORMULA:
            # 检测是否可能是物理、化学或数学公式
            domain = "unknown"
            if any(x in caption.lower() for x in ["physics", "motion", "force", "energy"]):
                domain = "physics"
            elif any(x in caption.lower() for x in ["chemistry", "molecule", "reaction"]):
                domain = "chemistry"
            elif any(x in caption.lower() for x in ["math", "algebra", "calculus", "geometry"]):
                domain = "mathematics"
                
            features["type_specific_info"]["domain"] = domain
            
        elif content_type == ContentType.TABLE:
            # 尝试估计表格规模
            table_size = "unknown"
            if any(x in caption.lower() for x in ["small", "simple"]):
                table_size = "small"
            elif any(x in caption.lower() for x in ["large", "complex"]):
                table_size = "large"
                
            features["type_specific_info"]["table_size"] = table_size
        
        return features
    
    def detect_educational_diagram(
        self, 
        image: Union[str, Path, np.ndarray]
    ) -> Dict[str, Any]:
        """
        特别检测教育类图表
        
        Args:
            image: 图像路径或numpy数组
            
        Returns:
            Dict[str, Any]: 教育图表检测结果
        """
        try:
            # 先进行一般分类
            classify_result = self.classify_content(image, detailed=True)
            if not classify_result["success"]:
                return classify_result
                
            # 如果已经被识别为图表或图解，进行进一步分析
            is_diagram = classify_result["content_type"] in ["chart", "diagram"]
            
            # 专门针对教育图表的问题
            edu_questions = [
                "Is this a diagram from an educational textbook?",
                "What subject does this diagram illustrate?",
                "Is this diagram showing a scientific concept?",
                "What educational level is this diagram appropriate for?"
            ]
            
            edu_qa_results = {}
            for question in edu_questions:
                result = self.blip_model.answer_question(image, question)
                if result["success"]:
                    edu_qa_results[question] = result["answer"]
            
            # 判断是否是教育图表
            is_educational = False
            subject = "unknown"
            educational_level = "unknown"
            
            combined_text = " ".join(edu_qa_results.values()).lower()
            
            if "yes" in combined_text[:10] or "textbook" in combined_text or "educational" in combined_text:
                is_educational = True
                
            # 尝试确定学科
            subjects = ["math", "physics", "chemistry", "biology", "history", "geography", "computer"]
            for s in subjects:
                if s in combined_text:
                    subject = s
                    break
            
            # 尝试确定教育级别
            levels = ["elementary", "primary", "middle school", "high school", "college", "university", "graduate"]
            for l in levels:
                if l in combined_text:
                    educational_level = l
                    break
            
            return {
                "success": True,
                "is_diagram": is_diagram,
                "is_educational": is_educational,
                "content_type": classify_result["content_type"],
                "subject": subject,
                "educational_level": educational_level,
                "qa_results": edu_qa_results
            }
            
        except Exception as e:
            logger.error(f"Error in detect_educational_diagram: {str(e)}")
            return {"success": False, "error": str(e)}


# 创建单例实例，方便直接导入使用
classifier = ContentClassifier()