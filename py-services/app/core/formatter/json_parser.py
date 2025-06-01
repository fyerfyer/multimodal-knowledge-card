import json
import re
from typing import Dict, Any, List, Optional, Tuple, Union
import time

from app.utils.logger import logger
from app.core.formatter.schema import KnowledgeCard, SimpleKnowledgeCard


class LLMOutputParser:
    """
    LLM输出解析器
    负责将大语言模型生成的文本解析为结构化的知识卡片JSON格式
    """
    
    def __init__(self):
        """初始化LLM输出解析器"""
        # 用于识别JSON块的模式
        self.json_block_pattern = re.compile(r'```(?:json)?\s*([\s\S]*?)\s*```')
        
        # 用于提取部分的模式
        self.section_patterns = {
            'title': re.compile(r'(?:^|\n)(?:#\s*|标题[:：]\s*)(.+)(?:\n|$)'),
            'summary': re.compile(r'(?:^|\n)(?:##\s*摘要|摘要[:：])\s*(.+?)(?:\n|$)'),
            'key_points': re.compile(r'(?:^|\n)(?:##\s*(?:关键|核心)?知识点|知识点[:：])\s*([\s\S]+?)(?:\n##|\n\d+\.|$)'),
            'quiz': re.compile(r'(?:^|\n)(?:##\s*(?:练习题|问题)|问题[:：])\s*([\s\S]+?)(?:\n##|$)'),
        }
        
    def parse(self, llm_output: str) -> Dict[str, Any]:
        """
        解析LLM输出文本为结构化的知识卡片
        
        Args:
            llm_output: LLM生成的原始输出文本
            
        Returns:
            Dict[str, Any]: 结构化的知识卡片数据
        """
        start_time = time.time()
        
        try:
            logger.info("Parsing LLM output to structured knowledge card")
            
            # 首先尝试提取JSON块
            json_match = self.json_block_pattern.search(llm_output)
            
            if json_match:
                # 如果找到JSON块，尝试解析
                return self.parse_json_block(json_match.group(1))
            else:
                # 如果没有JSON块，尝试从文本中解析结构
                return self.extract_from_text(llm_output)
                
        except Exception as e:
            logger.error(f"Error parsing LLM output: {str(e)}")
            # 返回解析失败的结果
            return {
                "success": False,
                "error": f"Failed to parse output: {str(e)}",
                "raw_output": llm_output
            }
        finally:
            processing_time = time.time() - start_time
            logger.debug(f"Parsing completed in {processing_time:.3f}s")
    
    def parse_json_block(self, json_str: str) -> Dict[str, Any]:
        """
        解析JSON块字符串
        
        Args:
            json_str: 提取的JSON字符串
            
        Returns:
            Dict[str, Any]: 解析的JSON对象，如果解析失败则返回错误信息
        """
        try:
            # 修正常见的JSON错误
            corrected_json = self._correct_common_json_errors(json_str)
            
            # 解析JSON
            card_data = json.loads(corrected_json)
            
            # 验证基本结构
            if not self._validate_basic_structure(card_data):
                logger.warning("Parsed JSON missing required fields")
                return {
                    "success": False,
                    "error": "Parsed JSON does not contain required fields (title, key_points, quiz)",
                    "partial_data": card_data
                }
            
            # 验证并补全字段
            card_data = self._normalize_card_data(card_data)
            
            # 构造成功响应
            return {
                "success": True,
                "card_data": card_data,
                "format": "json"
            }
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error: {str(e)}")
            # 如果JSON解析失败，尝试从文本中提取
            return self.extract_from_text(json_str)
    
    def extract_from_text(self, text: str) -> Dict[str, Any]:
        """
        从非JSON文本中提取知识卡片结构
        
        Args:
            text: LLM生成的文本
            
        Returns:
            Dict[str, Any]: 提取的知识卡片数据
        """
        try:
            # 提取各个部分
            title = self._extract_title(text)
            summary = self._extract_summary(text)
            key_points = self._extract_key_points(text)
            quiz = self._extract_quiz(text)
            
            # 检查是否至少提取到了必要字段
            if not (title and key_points and quiz):
                # 必要字段缺失，返回失败
                missing = []
                if not title: missing.append("title")
                if not key_points: missing.append("key_points")
                if not quiz: missing.append("quiz")
                
                logger.warning(f"Failed to extract required fields: {', '.join(missing)}")
                return {
                    "success": False,
                    "error": f"Could not extract required fields: {', '.join(missing)}",
                    "partial_data": {
                        "title": title,
                        "key_points": key_points,
                        "quiz": quiz,
                        "summary": summary
                    },
                    "raw_text": text
                }
            
            # 构建卡片数据
            card_data = {
                "title": title,
                "key_points": key_points,
                "quiz": quiz
            }
            
            if summary:
                card_data["summary"] = summary
                
            # 添加元数据
            card_data["metadata"] = {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "extraction_method": "text_pattern"
            }
                
            # 规范化数据
            normalized_data = self._normalize_card_data(card_data)
            
            return {
                "success": True,
                "card_data": normalized_data,
                "format": "extracted_text"
            }
            
        except Exception as e:
            logger.error(f"Error extracting from text: {str(e)}")
            return {
                "success": False,
                "error": f"Text extraction failed: {str(e)}",
                "raw_text": text
            }
    
    def _extract_title(self, text: str) -> str:
        """从文本中提取标题"""
        match = self.section_patterns['title'].search(text)
        if match:
            return match.group(1).strip()
        
        # 如果上面的模式没找到，尝试取第一个非空行
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if lines:
            return lines[0]
            
        return ""
    
    def _extract_summary(self, text: str) -> str:
        """从文本中提取摘要"""
        match = self.section_patterns['summary'].search(text)
        return match.group(1).strip() if match else ""
    
    def _extract_key_points(self, text: str) -> List[Dict[str, str]]:
        """从文本中提取关键知识点"""
        match = self.section_patterns['key_points'].search(text)
        if not match:
            return []
            
        key_points_text = match.group(1).strip()
        
        # 按列表项分割
        items = re.split(r'(?:^|\n)[\*\-•]|\d+\.', key_points_text)
        # 过滤空项并清理空白
        items = [item.strip() for item in items if item.strip()]
        
        # 转换为标准格式
        result = []
        for item in items:
            # 检查是否包含公式
            formula_match = re.search(r'`(.*?)`|公式[:：](.*?)($|\n)', item)
            formula = formula_match.group(1) if formula_match else None
            
            result.append({
                "content": item,
                "formula": formula
            })
            
        return result
    
    def _extract_quiz(self, text: str) -> Dict[str, str]:
        """从文本中提取练习题和答案"""
        match = self.section_patterns['quiz'].search(text)
        if not match:
            return {"question": "", "answer": ""}
            
        quiz_text = match.group(1).strip()
        
        # 尝试分离问题和答案
        answer_patterns = [
            r'(?:^|\n)答案[:：]?\s*([\s\S]+)',
            r'(?:^|\n)Answer[:：]?\s*([\s\S]+)',
            r'(?:^|\n)解答[:：]?\s*([\s\S]+)',
            r'(?:^|\n)解析[:：]?\s*([\s\S]+)'
        ]
        
        answer = ""
        for pattern in answer_patterns:
            answer_match = re.search(pattern, quiz_text)
            if answer_match:
                answer = answer_match.group(1).strip()
                # 从问题文本中删除答案部分
                question = re.sub(pattern, '', quiz_text).strip()
                break
        else:
            # 如果没有明确的答案标记，假设前一半是问题，后一半是答案
            lines = quiz_text.split('\n')
            if len(lines) >= 2:
                half_point = max(1, len(lines) // 2)
                question = '\n'.join(lines[:half_point]).strip()
                answer = '\n'.join(lines[half_point:]).strip()
            else:
                question = quiz_text
                answer = ""
        
        return {"question": question, "answer": answer}
    
    def _correct_common_json_errors(self, json_str: str) -> str:
        """
        修正常见的JSON格式错误
        
        Args:
            json_str: 原始JSON字符串
            
        Returns:
            str: 修正后的JSON字符串
        """
        # 移除 Python 风格单引号并替换为双引号
        corrected = re.sub(r'(?<![\\])\'', '"', json_str)
        
        # 修复没有使用双引号的键
        corrected = re.sub(r'(\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*):', r'\1"\2"\3:', corrected)
        
        # 修复尾随逗号
        corrected = re.sub(r',(\s*[}\]])', r'\1', corrected)
        
        # 修复缺少逗号的问题
        corrected = re.sub(r'(["\d])\s*\n\s*(["{])', r'\1,\n\2', corrected)
        
        # 确保布尔值和空值为小写
        corrected = re.sub(r'\bTrue\b', 'true', corrected)
        corrected = re.sub(r'\bFalse\b', 'false', corrected)
        corrected = re.sub(r'\bNone\b', 'null', corrected)
        
        return corrected
    
    def _validate_basic_structure(self, data: Dict[str, Any]) -> bool:
        """
        验证解析后的数据是否包含必要字段
        
        Args:
            data: 解析的JSON对象
            
        Returns:
            bool: 是否包含所有必要字段
        """
        required_fields = ["title", "key_points", "quiz"]
        return all(field in data for field in required_fields)
    
    def _normalize_card_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        规范化卡片数据，确保格式一致
        
        Args:
            data: 解析的卡片数据
            
        Returns:
            Dict[str, Any]: 规范化后的数据
        """
        # 复制数据以避免修改原始数据
        normalized = dict(data)
        
        # 确保key_points是列表对象
        if "key_points" in normalized and not isinstance(normalized["key_points"], list):
            if isinstance(normalized["key_points"], str):
                normalized["key_points"] = [{"content": normalized["key_points"]}]
            else:
                normalized["key_points"] = []
        
        # 如果key_points是字符串列表，转换为标准格式
        if "key_points" in normalized:
            for i, point in enumerate(normalized["key_points"]):
                if isinstance(point, str):
                    normalized["key_points"][i] = {"content": point}
        
        # 确保quiz是包含question和answer字段的对象
        if "quiz" in normalized:
            if isinstance(normalized["quiz"], str):
                normalized["quiz"] = {"question": normalized["quiz"], "answer": ""}
            elif isinstance(normalized["quiz"], list) and len(normalized["quiz"]) > 0:
                if isinstance(normalized["quiz"][0], str):
                    normalized["quiz"] = {
                        "question": normalized["quiz"][0],
                        "answer": normalized["quiz"][1] if len(normalized["quiz"]) > 1 else ""
                    }
                elif isinstance(normalized["quiz"][0], dict) and "question" in normalized["quiz"][0]:
                    normalized["quiz"] = normalized["quiz"][0]
            
            # 确保quiz包含必要的字段
            if "question" not in normalized["quiz"]:
                normalized["quiz"]["question"] = ""
            if "answer" not in normalized["quiz"]:
                normalized["quiz"]["answer"] = ""
        
        # 确保metadata存在
        if "metadata" not in normalized:
            normalized["metadata"] = {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
        
        return normalized
    
    def format_as_knowledge_card(self, data: Dict[str, Any]) -> Union[Dict[str, Any], KnowledgeCard]:
        """
        将通用字典格式化为KnowledgeCard模型实例
        
        Args:
            data: 卡片数据字典
            
        Returns:
            Union[Dict[str, Any], KnowledgeCard]: 格式化后的知识卡片，如果验证失败则返回错误字典
        """
        try:
            # 首先尝试使用SimpleKnowledgeCard进行验证和转换
            if all(key in data for key in ["title", "key_points", "quiz"]):
                try:
                    # 使用简单模式进行初始验证
                    simple_card = SimpleKnowledgeCard(
                        title=data["title"],
                        key_points=[kp["content"] if isinstance(kp, dict) else kp for kp in data["key_points"]],
                        quiz=data["quiz"] if isinstance(data["quiz"], dict) else {"question": data["quiz"], "answer": ""},
                        additional_info={k: v for k, v in data.items() if k not in ["title", "key_points", "quiz"]}
                    )
                    # 简单验证通过
                except Exception as e:
                    logger.warning(f"Simple validation failed: {str(e)}")
                
            # 尝试构造完整的KnowledgeCard
            return KnowledgeCard(**data)
            
        except Exception as e:
            logger.warning(f"Failed to validate as KnowledgeCard: {str(e)}")
            # 返回原始数据和错误
            return {
                "data": data,
                "validation_error": str(e)
            }


# 创建单例实例，方便直接导入使用
llm_output_parser = LLMOutputParser()