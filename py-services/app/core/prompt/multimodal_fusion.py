from typing import Dict, Any, List, Optional, Union
import re
from pathlib import Path
import numpy as np

from app.utils.logger import logger
from app.core.vision.service import vision_service
from app.core.ocr.service import ocr_service


class MultiModalFusion:
    """
    多模态融合处理类
    负责组合和优化来自图像理解和OCR文本识别的结果，创建更高质量的多模态输入
    """
    
    def __init__(self, 
                 vision_threshold: float = 0.6,
                 max_ocr_text_length: int = 4000):
        """
        初始化多模态融合处理器
        
        Args:
            vision_threshold: 视觉内容有效性阈值
            max_ocr_text_length: OCR文本最大长度限制
        """
        self.vision_threshold = vision_threshold
        self.max_ocr_text_length = max_ocr_text_length
        logger.info("MultiModal fusion processor initialized")
    
    def fuse_content(self, 
                     image: Union[str, Path, np.ndarray],
                     include_vision: bool = True,
                     include_ocr: bool = True,
                     vision_result: Optional[Dict[str, Any]] = None,
                     ocr_result: Optional[Dict[str, Any]] = None
                    ) -> Dict[str, Any]:
        """
        融合图像理解和OCR结果
        
        Args:
            image: 图像路径或numpy数组
            include_vision: 是否包含图像理解结果
            include_ocr: 是否包含OCR文本结果
            vision_result: 预先计算的图像理解结果（如果为None则重新计算）
            ocr_result: 预先计算的OCR结果（如果为None则重新计算）
            
        Returns:
            Dict[str, Any]: 融合后的内容
        """
        try:
            # 如果没有提供预计算结果，则计算图像理解结果
            if include_vision and vision_result is None:
                vision_result = vision_service.describe_for_prompt(image)
                
                if not vision_result["success"]:
                    logger.warning(f"Vision analysis failed: {vision_result.get('error', 'Unknown error')}")
                    include_vision = False
                    vision_result = None
            
            # 如果没有提供预计算结果，则计算OCR结果
            if include_ocr and ocr_result is None:
                if isinstance(image, (str, Path)):
                    ocr_result = ocr_service.process_image_file(image)
                else:
                    ocr_result = ocr_service.process_image_data(image)
                    
                if not ocr_result["success"]:
                    logger.warning(f"OCR processing failed: {ocr_result.get('error', 'Unknown error')}")
                    include_ocr = False
                    ocr_result = None
            
            # 如果两个分析都失败，返回错误
            if not include_vision and not include_ocr:
                return {
                    "success": False,
                    "error": "Both vision analysis and OCR processing failed"
                }
            
            # 确定内容类型（优先使用vision结果）
            content_type = "unknown"
            if include_vision and vision_result:
                content_type = vision_result.get("content_type", "unknown")
            
            # 组合结果
            fusion_result = {
                "success": True,
                "content_type": content_type,
                "vision_included": include_vision,
                "ocr_included": include_ocr
            }
            
            # 应用不同的融合策略，基于内容类型
            if content_type in ["chart", "diagram"]:
                fusion_result.update(self._fuse_diagram_content(vision_result, ocr_result))
            elif content_type == "formula":
                fusion_result.update(self._fuse_formula_content(vision_result, ocr_result))
            elif content_type == "table":
                fusion_result.update(self._fuse_table_content(vision_result, ocr_result))
            else:
                # 默认策略 (text, mixed等)
                fusion_result.update(self._fuse_general_content(vision_result, ocr_result))
                
            return fusion_result
            
        except Exception as e:
            logger.error(f"Error in content fusion: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _fuse_general_content(self, 
                             vision_result: Optional[Dict[str, Any]], 
                             ocr_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        通用内容融合策略
        
        Args:
            vision_result: 图像理解结果
            ocr_result: OCR结果
            
        Returns:
            Dict[str, Any]: 融合结果
        """
        result = {}
        
        # 提取图像描述
        if vision_result:
            result["caption"] = vision_result.get("caption", "")
            result["focus_points"] = vision_result.get("focus_points", {})
        else:
            result["caption"] = ""
            result["focus_points"] = {}
        
        # 提取OCR文本
        if ocr_result:
            # 限制OCR文本长度
            full_text = ocr_result.get("full_text", "")
            if len(full_text) > self.max_ocr_text_length:
                # 如果文本过长，进行截断
                full_text = full_text[:self.max_ocr_text_length] + "... [text truncated due to length]"
                logger.info(f"OCR text truncated from {len(ocr_result.get('full_text', ''))} to {len(full_text)} characters")
                
            result["ocr_text"] = full_text
            result["paragraph_count"] = ocr_result.get("paragraph_count", 0)
        else:
            result["ocr_text"] = ""
            result["paragraph_count"] = 0
            
        return result
    
    def _fuse_diagram_content(self, 
                             vision_result: Optional[Dict[str, Any]], 
                             ocr_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        图表类内容融合策略
        
        Args:
            vision_result: 图像理解结果
            ocr_result: OCR结果
            
        Returns:
            Dict[str, Any]: 融合结果
        """
        # 基础融合
        result = self._fuse_general_content(vision_result, ocr_result)
        
        # 图表特定处理
        if vision_result:
            # 尝试提取图表类型
            chart_type = "unknown"
            caption = vision_result.get("caption", "").lower()
            
            if "bar" in caption or "histogram" in caption:
                chart_type = "bar_chart"
            elif "line" in caption or "trend" in caption:
                chart_type = "line_chart"
            elif "pie" in caption:
                chart_type = "pie_chart"
            elif "scatter" in caption:
                chart_type = "scatter_plot"
            elif "flow" in caption or "diagram" in caption:
                chart_type = "flow_diagram"
            
            result["diagram_type"] = chart_type
            
            # 提取关键元素
            focus_points = vision_result.get("focus_points", {})
            main_topic = focus_points.get("What is the main topic of this image?", "")
            
            result["main_topic"] = main_topic
            
        # 处理图表中的文字（可能是图例、轴标签等）
        if ocr_result:
            # 尝试识别轴标签和图例
            text_items = []
            for paragraph in ocr_result.get("paragraphs", []):
                text_items.append(paragraph.strip())
            
            result["text_elements"] = text_items
            
        return result
    
    def _fuse_formula_content(self, 
                             vision_result: Optional[Dict[str, Any]], 
                             ocr_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        数学公式内容融合策略
        
        Args:
            vision_result: 图像理解结果
            ocr_result: OCR结果
            
        Returns:
            Dict[str, Any]: 融合结果
        """
        # 基础融合
        result = self._fuse_general_content(vision_result, ocr_result)
        
        # 公式特定处理
        if ocr_result and ocr_result.get("full_text"):
            # 尝试清理和格式化公式文本
            formula_text = ocr_result.get("full_text", "")
            # 简单清理一些OCR可能错误识别的字符
            formula_text = re.sub(r'[^\w\s\.\,\?\!\:\;\'\"\(\)\[\]\{\}\-\+\=\*\/\\]', '', formula_text)
            
            result["formula_text"] = formula_text
        
        if vision_result:
            # 尝试获取公式的主题领域
            caption = vision_result.get("caption", "").lower()
            formula_domain = "general"
            
            if any(kw in caption for kw in ["physics", "mechanics", "force", "energy"]):
                formula_domain = "physics"
            elif any(kw in caption for kw in ["math", "algebra", "calculus"]):
                formula_domain = "mathematics"
            elif any(kw in caption for kw in ["chemistry", "reaction", "compound"]):
                formula_domain = "chemistry"
                
            result["formula_domain"] = formula_domain
            
        return result
    
    def _fuse_table_content(self, 
                           vision_result: Optional[Dict[str, Any]], 
                           ocr_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        表格内容融合策略
        
        Args:
            vision_result: 图像理解结果
            ocr_result: OCR结果
            
        Returns:
            Dict[str, Any]: 融合结果
        """
        # 基础融合
        result = self._fuse_general_content(vision_result, ocr_result)
        
        # 表格特定处理：尝试识别表格结构
        if ocr_result:
            full_text = ocr_result.get("full_text", "")
            
            # 估计表格大小（行数）
            estimated_rows = full_text.count('\n') + 1
            
            # 简单启发式：检测表头
            paragraphs = ocr_result.get("paragraphs", [])
            header = paragraphs[0] if paragraphs else ""
            
            result["estimated_rows"] = estimated_rows
            result["possible_header"] = header
            
        return result
    
    def chunk_content(self, 
                      ocr_text: str, 
                      max_chunk_size: int = 1000, 
                      overlap: int = 100) -> List[str]:
        """
        将长文本切分成较小的块，适用于处理长文档
        
        Args:
            ocr_text: 要切分的OCR文本
            max_chunk_size: 每个块的最大字符数
            overlap: 相邻块之间的重叠字符数
            
        Returns:
            List[str]: 文本块列表
        """
        # 如果文本长度小于最大块大小，直接返回原文本
        if len(ocr_text) <= max_chunk_size:
            return [ocr_text]
        
        chunks = []
        start = 0
        
        while start < len(ocr_text):
            # 确定当前块的结束位置
            end = start + max_chunk_size
            
            # 如果不是最后一块，尝试在句子或段落边界切分
            if end < len(ocr_text):
                # 尝试在段落边界切分
                paragraph_end = ocr_text.rfind('\n\n', start, end)
                if paragraph_end != -1 and paragraph_end > start + max_chunk_size // 2:
                    end = paragraph_end + 2  # 包含段落分隔符
                else:
                    # 尝试在句子边界切分
                    sentence_end = ocr_text.rfind('. ', start, end)
                    if sentence_end != -1 and sentence_end > start + max_chunk_size // 2:
                        end = sentence_end + 2  # 包含句号和空格
            
            # 添加当前块
            chunks.append(ocr_text[start:end])
            
            # 更新下一块的起始位置（考虑重叠）
            start = end - overlap if end < len(ocr_text) else len(ocr_text)
        
        logger.info(f"Split content into {len(chunks)} chunks")
        return chunks
    
    def prioritize_content(self, 
                          ocr_text: str, 
                          vision_caption: str, 
                          max_length: int = 2000) -> str:
        """
        当内容过长时，基于相关性进行优先级排序和截断
        
        Args:
            ocr_text: OCR提取的文本
            vision_caption: 图像描述
            max_length: 返回文本的最大长度
            
        Returns:
            str: 优先级排序后的文本
        """
        if len(ocr_text) <= max_length:
            return ocr_text
        
        # 将文本分段
        paragraphs = ocr_text.split('\n\n')
        
        # 创建基于与图像描述相关性的段落分数
        scored_paragraphs = []
        for paragraph in paragraphs:
            # 简单的相关性评分：段落中包含多少图像描述中的关键词
            caption_words = set(re.findall(r'\b\w+\b', vision_caption.lower()))
            paragraph_words = set(re.findall(r'\b\w+\b', paragraph.lower()))
            
            # 计算相似词汇比例
            common_words = caption_words.intersection(paragraph_words)
            relevance_score = len(common_words) / max(1, len(caption_words))
            
            # 考虑段落长度（较短段落可能是标题或重要说明）
            length_factor = 1.0
            if len(paragraph) < 50:  # 短段落
                length_factor = 1.5
                
            final_score = relevance_score * length_factor
            scored_paragraphs.append((paragraph, final_score))
        
        # 根据分数排序段落（高分在前）
        sorted_paragraphs = sorted(scored_paragraphs, key=lambda x: x[1], reverse=True)
        
        # 保留高分段落直到达到长度限制
        prioritized_text = ""
        for paragraph, _ in sorted_paragraphs:
            if len(prioritized_text + paragraph + "\n\n") <= max_length:
                prioritized_text += paragraph + "\n\n"
            else:
                break
                
        logger.info(f"Prioritized content: kept {len(prioritized_text)} out of {len(ocr_text)} characters")
        return prioritized_text.strip()


# 创建单例实例，方便直接导入使用
multimodal_fusion = MultiModalFusion()