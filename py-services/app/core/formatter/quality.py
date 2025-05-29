from typing import Dict, Any, List, Optional
import re

from app.utils.logger import logger
from app.core.formatter.schema import KnowledgeCard, KeyPoint

class QualityController:
    """
    卡片质量控制器
    负责验证和提升知识卡片的质量
    """
    
    def __init__(self, 
                 min_title_length: int = 3,
                 max_title_length: int = 100,
                 min_key_points: int = 1,
                 min_key_point_length: int = 10,
                 max_key_point_length: int = 500,
                 min_quiz_question_length: int = 10,
                 min_quiz_answer_length: int = 5):
        """
        初始化质量控制器
        
        Args:
            min_title_length: 标题最小长度
            max_title_length: 标题最大长度
            min_key_points: 最少知识点数量
            min_key_point_length: 知识点最小长度
            max_key_point_length: 知识点最大长度
            min_quiz_question_length: 问题最小长度
            min_quiz_answer_length: 答案最小长度
        """
        self.min_title_length = min_title_length
        self.max_title_length = max_title_length
        self.min_key_points = min_key_points
        self.min_key_point_length = min_key_point_length
        self.max_key_point_length = max_key_point_length
        self.min_quiz_question_length = min_quiz_question_length
        self.min_quiz_answer_length = min_quiz_answer_length
        
        logger.info("Quality controller initialized")
    
    def validate_card(self, card_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证知识卡片质量
        
        Args:
            card_data: 知识卡片数据
            
        Returns:
            Dict[str, Any]: 验证结果，包含问题列表和质量评分
        """
        issues = []
        quality_score = 100  # 满分100
        
        try:
            # 检查必要字段
            if "title" not in card_data:
                issues.append("Missing title")
                quality_score -= 30  # 缺少标题是严重问题
            
            if "key_points" not in card_data or not card_data["key_points"]:
                issues.append("Missing key points")
                quality_score -= 30  # 缺少知识点是严重问题
            
            if "quiz" not in card_data:
                issues.append("Missing quiz")
                quality_score -= 20  # 缺少练习题是重要问题
            
            # 如果缺少基本字段，直接返回
            if issues:
                return {
                    "valid": False,
                    "issues": issues,
                    "quality_score": max(0, quality_score),
                    "improvements": ["Ensure all required fields (title, key_points, quiz) are present"]
                }
            
            # 验证标题
            title = card_data["title"]
            if len(title) < self.min_title_length:
                issues.append(f"Title too short (min: {self.min_title_length})")
                quality_score -= 10
            elif len(title) > self.max_title_length:
                issues.append(f"Title too long (max: {self.max_title_length})")
                quality_score -= 5
            
            # 验证知识点
            key_points = card_data["key_points"]
            if isinstance(key_points, list):
                if len(key_points) < self.min_key_points:
                    issues.append(f"Too few key points (min: {self.min_key_points})")
                    quality_score -= 15
                
                for i, point in enumerate(key_points):
                    if isinstance(point, dict) and "content" in point:
                        content = point["content"]
                    elif isinstance(point, str):
                        content = point
                    else:
                        issues.append(f"Invalid key point format at index {i}")
                        quality_score -= 5
                        continue
                    
                    if len(content) < self.min_key_point_length:
                        issues.append(f"Key point {i+1} too short (min: {self.min_key_point_length})")
                        quality_score -= 5
                    elif len(content) > self.max_key_point_length:
                        issues.append(f"Key point {i+1} too long (max: {self.max_key_point_length})")
                        quality_score -= 3
            else:
                issues.append("Key points must be a list")
                quality_score -= 15
            
            # 验证练习题
            quiz = card_data.get("quiz", {})
            if isinstance(quiz, dict):
                if "question" not in quiz or not quiz["question"]:
                    issues.append("Missing quiz question")
                    quality_score -= 10
                elif len(quiz["question"]) < self.min_quiz_question_length:
                    issues.append(f"Quiz question too short (min: {self.min_quiz_question_length})")
                    quality_score -= 5
                
                if "answer" not in quiz or not quiz["answer"]:
                    issues.append("Missing quiz answer")
                    quality_score -= 10
                elif len(quiz["answer"]) < self.min_quiz_answer_length:
                    issues.append(f"Quiz answer too short (min: {self.min_quiz_answer_length})")
                    quality_score -= 5
            else:
                issues.append("Quiz must be an object with question and answer")
                quality_score -= 15
            
            # 创建改进建议
            improvements = self._generate_improvements(issues)
            
            # 确保质量分数在0-100之间
            quality_score = max(0, min(100, quality_score))
            
            return {
                "valid": len(issues) == 0,
                "issues": issues,
                "quality_score": quality_score,
                "improvements": improvements
            }
            
        except Exception as e:
            logger.error(f"Error validating card: {str(e)}")
            return {
                "valid": False,
                "issues": ["Error during validation: " + str(e)],
                "quality_score": 0,
                "improvements": ["Fix card structure to allow proper validation"]
            }
    
    def improve_card(self, card_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        尝试改进知识卡片质量
        
        Args:
            card_data: 知识卡片数据
            
        Returns:
            Dict[str, Any]: 改进后的知识卡片和改进信息
        """
        # 首先验证卡片
        validation = self.validate_card(card_data)
        
        # 如果没有问题，直接返回原卡片
        if validation["valid"]:
            return {
                "improved": False,
                "original": card_data,
                "improved_card": card_data,
                "improvements": []
            }
        
        try:
            # 创建卡片副本进行修改
            improved_card = card_data.copy()
            improvements = []
            
            # 1. 修复标题
            if "title" in improved_card:
                original_title = improved_card["title"]
                title = self._fix_title(original_title)
                if title != original_title:
                    improved_card["title"] = title
                    improvements.append(f"Fixed title: '{original_title}' -> '{title}'")
            
            # 2. 修复知识点
            if "key_points" in improved_card and isinstance(improved_card["key_points"], list):
                fixed_points = []
                
                for i, point in enumerate(improved_card["key_points"]):
                    # 统一知识点格式
                    if isinstance(point, str):
                        fixed_point = {"content": point}
                        improvements.append(f"Standardized format for key point {i+1}")
                    elif isinstance(point, dict):
                        fixed_point = point.copy()
                    else:
                        continue
                    
                    # 确保有content字段
                    if "content" not in fixed_point or not fixed_point["content"]:
                        continue
                    
                    # 修复内容
                    original_content = fixed_point["content"]
                    fixed_content = self._fix_text(original_content, 
                                                 min_length=self.min_key_point_length, 
                                                 max_length=self.max_key_point_length)
                    
                    if fixed_content != original_content:
                        fixed_point["content"] = fixed_content
                        improvements.append(f"Fixed content for key point {i+1}")
                    
                    fixed_points.append(fixed_point)
                
                # 如果没有足够的知识点，尝试从现有知识点中分割
                if len(fixed_points) < self.min_key_points and fixed_points:
                    # 找到最长的知识点
                    longest_idx = max(range(len(fixed_points)), 
                                    key=lambda i: len(fixed_points[i]["content"]))
                    
                    # 尝试分割它
                    long_content = fixed_points[longest_idx]["content"]
                    if len(long_content) > self.min_key_point_length * 2:
                        split_point = len(long_content) // 2
                        # 找到附近的句号或分号
                        for i in range(split_point, min(len(long_content), split_point + 50)):
                            if long_content[i] in ['.', '。', ';', '；']:
                                split_point = i + 1
                                break
                        
                        if split_point < len(long_content):
                            part1 = long_content[:split_point].strip()
                            part2 = long_content[split_point:].strip()
                            
                            fixed_points[longest_idx]["content"] = part1
                            fixed_points.append({"content": part2})
                            
                            improvements.append(f"Split long key point into two separate points")
                
                improved_card["key_points"] = fixed_points
            
            # 3. 修复练习题
            if "quiz" in improved_card:
                original_quiz = improved_card["quiz"]
                fixed_quiz = {}
                
                # 统一格式
                if isinstance(original_quiz, str):
                    # 尝试将其分为问题和答案
                    parts = original_quiz.split('?', 1)
                    if len(parts) > 1:
                        fixed_quiz = {
                            "question": parts[0] + '?',
                            "answer": parts[1].strip()
                        }
                        improvements.append("Converted quiz string to question-answer format")
                    else:
                        fixed_quiz = {
                            "question": original_quiz,
                            "answer": "Please provide an answer."
                        }
                        improvements.append("Created quiz structure with placeholder answer")
                elif isinstance(original_quiz, dict):
                    fixed_quiz = original_quiz.copy()
                
                # 修复问题
                if "question" in fixed_quiz:
                    original_question = fixed_quiz["question"]
                    fixed_question = self._fix_text(original_question, 
                                                  min_length=self.min_quiz_question_length)
                    
                    if fixed_question != original_question:
                        fixed_quiz["question"] = fixed_question
                        improvements.append("Fixed quiz question")
                else:
                    fixed_quiz["question"] = "Based on the key points, what concept is being discussed?"
                    improvements.append("Added default quiz question")
                
                # 修复答案
                if "answer" in fixed_quiz:
                    original_answer = fixed_quiz["answer"]
                    fixed_answer = self._fix_text(original_answer, 
                                                min_length=self.min_quiz_answer_length)
                    
                    if fixed_answer != original_answer:
                        fixed_quiz["answer"] = fixed_answer
                        improvements.append("Fixed quiz answer")
                else:
                    fixed_quiz["answer"] = "Please refer to the key points for the answer."
                    improvements.append("Added default quiz answer")
                
                improved_card["quiz"] = fixed_quiz
            
            # 重新验证改进后的卡片
            revalidation = self.validate_card(improved_card)
            
            return {
                "improved": revalidation["valid"] or revalidation["quality_score"] > validation["quality_score"],
                "original": card_data,
                "improved_card": improved_card,
                "original_score": validation["quality_score"],
                "improved_score": revalidation["quality_score"],
                "improvements": improvements,
                "remaining_issues": revalidation["issues"]
            }
            
        except Exception as e:
            logger.error(f"Error improving card: {str(e)}")
            return {
                "improved": False,
                "original": card_data,
                "improved_card": card_data,
                "error": str(e),
                "improvements": []
            }
    
    def _fix_title(self, title: str) -> str:
        """
        修复标题格式和长度
        
        Args:
            title: 原始标题
            
        Returns:
            str: 修复后的标题
        """
        if not title:
            return "Knowledge Card"
        
        # 去除多余空格和标点
        title = title.strip()
        title = re.sub(r'\s+', ' ', title)
        
        # 去除可能的前置标记，例如"标题："或"Title:"
        title = re.sub(r'^(标题[:：]|Title:)\s*', '', title)
        
        # 截断过长标题
        if len(title) > self.max_title_length:
            title = title[:self.max_title_length] + "..."
        
        # 补充过短标题
        if len(title) < self.min_title_length:
            title = title + " Knowledge Card"
        
        # 确保首字母大写
        if title and title[0].isalpha():
            title = title[0].upper() + title[1:]
        
        return title
    
    def _fix_text(self, text: str, min_length: int, max_length: Optional[int] = None) -> str:
        """
        修复文本内容和长度
        
        Args:
            text: 原始文本
            min_length: 最小长度
            max_length: 最大长度
            
        Returns:
            str: 修复后的文本
        """
        if not text:
            return "Content unavailable"
        
        # 去除多余空格
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        
        # 截断过长文本
        if max_length and len(text) > max_length:
            # 尝试在句子结束处截断
            last_period = text[:max_length].rfind('.')
            if last_period > max_length * 0.5:  # 如果在后半部分有句号，在句号处截断
                text = text[:last_period + 1]
            else:  # 否则直接截断并添加省略号
                text = text[:max_length] + "..."
        
        # 补充过短文本
        if len(text) < min_length:
            if text.endswith('.'):
                text = text[:-1]
            text = text + ". Please refer to the source material for more details."
        
        return text
    
    def _generate_improvements(self, issues: List[str]) -> List[str]:
        """
        根据问题生成改进建议
        
        Args:
            issues: 问题列表
            
        Returns:
            List[str]: 改进建议列表
        """
        improvements = []
        
        for issue in issues:
            if "title too short" in issue.lower():
                improvements.append("Make the title more descriptive and specific")
            
            elif "title too long" in issue.lower():
                improvements.append("Shorten the title to be more concise while retaining key information")
            
            elif "too few key points" in issue.lower():
                improvements.append("Add more key points to cover the topic comprehensively")
            
            elif "key point too short" in issue.lower():
                improvements.append("Expand the short key points with more details or examples")
            
            elif "key point too long" in issue.lower():
                improvements.append("Break long key points into smaller, focused points")
            
            elif "missing quiz question" in issue.lower():
                improvements.append("Add a quiz question that tests understanding of the key points")
            
            elif "quiz question too short" in issue.lower():
                improvements.append("Make the quiz question more specific and clear")
            
            elif "missing quiz answer" in issue.lower():
                improvements.append("Provide a detailed answer to the quiz question")
            
            elif "quiz answer too short" in issue.lower():
                improvements.append("Expand the quiz answer with more explanation or reasoning")
        
        # 如果没有特定建议，添加通用建议
        if not improvements and issues:
            improvements.append("Review and address the identified issues to improve card quality")
        
        return improvements


# 创建单例实例，方便直接导入使用
quality_controller = QualityController()