from typing import Dict, Any, Union, Optional
import time

from app.utils.logger import logger
from app.core.formatter.schema import KnowledgeCard


class MarkdownFormatter:
    """
    Markdown格式化器
    负责将结构化的知识卡片转换为Markdown格式
    """
    
    def __init__(self):
        """初始化Markdown格式化器"""
        logger.info("Markdown formatter initialized")
    
    def card_to_markdown(self, card_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        将知识卡片数据转换为Markdown格式
        
        Args:
            card_data: 知识卡片数据
            
        Returns:
            Dict[str, Any]: 包含Markdown内容和元数据的字典
        """
        try:
            # 确保关键字段存在
            if not self._validate_card_data(card_data):
                return {
                    "success": False,
                    "error": "Invalid card data: missing required fields",
                    "markdown": ""
                }
            
            start_time = time.time()
            
            # 构建Markdown内容
            markdown = self._build_markdown(card_data)
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "markdown": markdown,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error formatting card to markdown: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "markdown": ""
            }
    
    def _validate_card_data(self, card_data: Dict[str, Any]) -> bool:
        """
        验证卡片数据是否包含必要字段
        
        Args:
            card_data: 知识卡片数据
            
        Returns:
            bool: 是否包含所有必要字段
        """
        return all(key in card_data for key in ["title", "key_points", "quiz"])
    
    def _build_markdown(self, card_data: Dict[str, Any]) -> str:
        """
        构建Markdown内容
        
        Args:
            card_data: 知识卡片数据
            
        Returns:
            str: Markdown格式的内容
        """
        # 开始构建Markdown
        md_parts = []
        
        # 标题
        md_parts.append(f"# {card_data['title']}")
        md_parts.append("")
        
        # 摘要（如果存在）
        if "summary" in card_data and card_data["summary"]:
            md_parts.append(f"*{card_data['summary']}*")
            md_parts.append("")
        
        # 关键知识点
        md_parts.append("## 关键知识点")
        
        for idx, point in enumerate(card_data["key_points"], 1):
            if isinstance(point, dict):
                content = point.get("content", "")
                formula = point.get("formula")
                reference = point.get("reference")
                
                point_md = f"{idx}. {content}"
                
                if formula:
                    point_md += f"\n   - 公式: `{formula}`"
                    
                if reference:
                    point_md += f"\n   - 来源: {reference}"
                    
                md_parts.append(point_md)
            else:
                md_parts.append(f"{idx}. {point}")
        
        md_parts.append("")
        
        # 练习题
        md_parts.append("## 练习题")
        
        quiz = card_data["quiz"]
        if isinstance(quiz, dict):
            question = quiz.get("question", "")
            answer = quiz.get("answer", "")
            
            md_parts.append(f"**问题**: {question}")
            md_parts.append("")
            md_parts.append(f"**答案**: {answer}")
        else:
            md_parts.append(str(quiz))
            
        md_parts.append("")
        
        # 引用（如果存在）
        if "references" in card_data and card_data["references"]:
            md_parts.append("## 引用")
            
            refs = card_data["references"]
            if isinstance(refs, dict):
                for key, value in refs.items():
                    md_parts.append(f"- {key}: {value}")
            else:
                md_parts.append(f"- {refs}")
                
            md_parts.append("")
        
        # 元数据（如果存在）
        if "metadata" in card_data and card_data["metadata"]:
            md_parts.append("<details>")
            md_parts.append("<summary>元数据</summary>")
            md_parts.append("")
            
            metadata = card_data["metadata"]
            for key, value in metadata.items():
                md_parts.append(f"- {key}: {value}")
                
            md_parts.append("")
            md_parts.append("</details>")
        
        # 合并所有部分
        return "\n".join(md_parts)
    
    def render_card_preview(self, card_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成知识卡片预览（简化版Markdown）
        
        Args:
            card_data: 知识卡片数据
            
        Returns:
            Dict[str, Any]: 包含预览内容的字典
        """
        try:
            # 验证数据
            if not self._validate_card_data(card_data):
                return {
                    "success": False,
                    "error": "Invalid card data for preview",
                    "preview": ""
                }
            
            # 构建简单预览
            preview = f"# {card_data['title']}\n\n"
            
            # 添加简短摘要
            if "summary" in card_data and card_data["summary"]:
                preview += f"{card_data['summary']}\n\n"
            
            # 添加知识点数量
            key_points_count = len(card_data.get("key_points", []))
            preview += f"包含 {key_points_count} 个知识点和 1 道练习题\n"
            
            return {
                "success": True,
                "preview": preview
            }
            
        except Exception as e:
            logger.error(f"Error rendering card preview: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "preview": ""
            }
    
    def markdown_to_html(self, markdown: str) -> Dict[str, Any]:
        """
        将Markdown转换为HTML
        
        Args:
            markdown: Markdown内容
            
        Returns:
            Dict[str, Any]: 包含HTML内容的字典
        """
        try:
            # 这里可以使用markdown库来转换，但需要在requirements.txt中添加依赖
            # TODO: 实现Markdown到HTML的转换，在添加markdown库后完成
            return {
                "success": False,
                "error": "Markdown to HTML conversion not implemented yet",
                "html": ""
            }
            
        except Exception as e:
            logger.error(f"Error converting markdown to HTML: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "html": ""
            }


# 创建单例实例，方便直接导入使用
markdown_formatter = MarkdownFormatter()