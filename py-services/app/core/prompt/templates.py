from typing import Dict, Any, List, Optional, Union
from enum import Enum
import os
import json
from pathlib import Path

from app.utils.logger import logger
from app.config.settings import settings

class TemplateType(Enum):
    """提示词模板类型枚举"""
    GENERAL = "general"  # 通用模板
    EDUCATIONAL = "educational"  # 教育内容模板
    FORMULA = "formula"  # 公式模板
    DIAGRAM = "diagram"  # 图表模板
    CHART = "chart"  # 统计图模板
    TABLE = "table"  # 表格模板
    TEXT_HEAVY = "text_heavy"  # 文本密集型模板


class PromptTemplateManager:
    """
    提示词模板管理器
    负责加载、存储、选择和格式化提示词模板
    """
    
    def __init__(self, 
                 template_dir: Optional[Path] = None,
                 custom_templates: Optional[Dict[str, str]] = None):
        """
        初始化提示词模板管理器
        
        Args:
            template_dir: 模板文件目录，如果为None则使用默认目录
            custom_templates: 自定义模板字典，可以覆盖或添加模板
        """
        # 使用用户指定目录或默认目录
        self.template_dir = template_dir or (settings.BASE_DIR / "app" / "core" / "prompt" / "templates")
        
        # 默认模板
        self.templates = self._get_default_templates()
        
        # 加载自定义模板
        if custom_templates:
            self.templates.update(custom_templates)
            logger.info(f"Added {len(custom_templates)} custom templates")
        
        # 尝试从文件加载模板
        self._load_templates_from_files()
        
        logger.info(f"Prompt template manager initialized with {len(self.templates)} templates")
    
    def _get_default_templates(self) -> Dict[str, str]:
        """
        获取默认模板
        
        Returns:
            Dict[str, str]: 默认模板字典
        """
        return {
            # 通用知识卡片模板
            TemplateType.GENERAL.value: """
你现在是一个教学知识总结助手。
以下是教材图像内容和图像中提取的文字，请生成一张知识卡片：

图像内容说明：
{image_caption}

OCR提取文字：
{ocr_text}

请生成结构化知识卡片，包含：
- 标题
- 2~3个关键知识点（简短）
- 一道相关的练习题（问答形式）
""",

            # 教育内容特定模板
            TemplateType.EDUCATIONAL.value: """
你是一位专业的教育内容总结专家。
请根据以下教材内容，生成一张结构化的知识卡片：

教材图像内容：
{image_caption}

文本内容：
{ocr_text}

请按以下格式生成知识卡片：
1. 标题：简洁明了地概括主题
2. 核心概念：列出3-5个关键概念或术语，并附简短解释
3. 主要知识点：用简明的语言总结2-3个主要知识点
4. 练习题：设计一道能够检验对这些知识理解的问题，并提供参考答案

注意：内容应该准确、简洁、结构清晰。
""",

            # 公式特定模板
            TemplateType.FORMULA.value: """
你是一位数学公式专家。
请分析以下数学或科学公式，并创建一张详细的知识卡片：

公式图像描述：
{image_caption}

OCR识别内容：
{ocr_text}

请按照以下格式生成公式知识卡片：
1. 公式名称
2. 公式表达式（使用文本形式准确表示）
3. 公式中各变量含义
4. 公式的应用场景或意义
5. 一个使用该公式的简单示例问题与解答

确保卡片内容准确、简洁，并突出公式的核心概念。
""",

            # 图表特定模板
            TemplateType.DIAGRAM.value: """
你是一位科学图表分析专家。
请解读以下教育图表，并生成相应的知识卡片：

图表描述：
{image_caption}

图表中的文字：
{ocr_text}

请按照以下结构生成图表知识卡片：
1. 图表标题/主题
2. 图表类型与用途
3. 图表展示的主要关系或过程
4. 关键要点（3点以内）
5. 一个检验读者理解的思考题

注意：请确保解释清晰，突出图表传达的核心信息。
""",

            # 统计图表特定模板
            TemplateType.CHART.value: """
你是一位数据可视化专家。
请分析以下统计图表，并生成一张信息卡片：

图表描述：
{image_caption}

图表上的文字：
{ocr_text}

请按照以下格式生成统计图知识卡片：
1. 图表主题
2. 图表类型（如折线图、柱状图、饼图等）
3. 数据来源（如有）
4. 主要趋势或发现（2-3点）
5. 数据背后的意义或应用
6. 一个基于图表的分析问题

注意：请尽量量化描述，使用精确的数字和比例。
""",

            # 表格特定模板
            TemplateType.TABLE.value: """
你是一位数据分析专家。
请解析以下表格内容，并生成一张知识卡片：

表格描述：
{image_caption}

表格内容（OCR识别）：
{ocr_text}

请按照以下格式生成表格知识卡片：
1. 表格主题/标题
2. 表格结构（行列组织方式）
3. 关键数据点或分类（3-5个）
4. 表格数据的主要发现或结论
5. 一个基于表格数据的应用问题

注意：请尽量保留表格的结构化信息，并突出数据间的关系。
""",

            # 文本密集型内容模板
            TemplateType.TEXT_HEAVY.value: """
你是一位专业的教材内容总结专家。
请对以下教材文本内容进行分析，并生成一张简洁的知识卡片：

内容主题：
{image_caption}

文本内容：
{ocr_text}

请按照以下格式生成文本知识卡片：
1. 标题（概括核心主题）
2. 核心概念（3-5个关键词或短语）
3. 主要观点摘要（不超过3点，每点30字以内）
4. 一个理解检测问题（针对核心内容，附带简要答案）

注意：请保持简洁性，突出最重要的信息，避免冗余内容。
"""
        }
    
    def _load_templates_from_files(self) -> None:
        """
        从文件加载模板，支持json和txt格式
        """
        try:
            # 确保模板目录存在
            os.makedirs(self.template_dir, exist_ok=True)
            
            # 扫描目录中的json文件
            for file_path in self.template_dir.glob("*.json"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        templates_data = json.load(f)
                        self.templates.update(templates_data)
                    logger.info(f"Loaded templates from {file_path}")
                except Exception as e:
                    logger.error(f"Error loading template file {file_path}: {str(e)}")
            
            # 加载单一模板文件（每个文件一个模板）
            for file_path in self.template_dir.glob("*.txt"):
                try:
                    template_name = file_path.stem  # 使用文件名作为模板名
                    with open(file_path, "r", encoding="utf-8") as f:
                        template_content = f.read()
                        self.templates[template_name] = template_content
                    logger.info(f"Loaded template '{template_name}' from {file_path}")
                except Exception as e:
                    logger.error(f"Error loading template file {file_path}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error loading templates from directory: {str(e)}")
    
    def get_template(self, template_type: Union[str, TemplateType]) -> str:
        """
        获取指定类型的模板
        
        Args:
            template_type: 模板类型，可以是TemplateType枚举或字符串
            
        Returns:
            str: 模板内容
            
        Raises:
            ValueError: 如果模板类型不存在
        """
        # 如果是枚举类型，转换为字符串
        if isinstance(template_type, TemplateType):
            template_type = template_type.value
            
        # 尝试获取模板
        template = self.templates.get(template_type)
        if template is None:
            logger.warning(f"Template '{template_type}' not found, using general template")
            template = self.templates.get(TemplateType.GENERAL.value)
            
            # 如果连通用模板都不存在，抛出异常
            if template is None:
                logger.error(f"General template not found")
                raise ValueError("Template not found and no default template available")
                
        return template
    
    def format_template(self, template_type: Union[str, TemplateType], **kwargs) -> str:
        """
        获取并格式化模板
        
        Args:
            template_type: 模板类型
            **kwargs: 要替换的变量
            
        Returns:
            str: 格式化后的模板
        """
        template = self.get_template(template_type)
        
        try:
            # 使用字符串格式化替换变量
            formatted_template = template.format(**kwargs)
            return formatted_template
        except KeyError as e:
            logger.error(f"Missing required parameter in template: {str(e)}")
            # 返回带有错误提示的模板
            return f"{template}\n\nERROR: Missing required parameter: {str(e)}"
        except Exception as e:
            logger.error(f"Error formatting template: {str(e)}")
            return template
    
    def add_template(self, name: str, content: str) -> None:
        """
        添加或更新模板
        
        Args:
            name: 模板名称
            content: 模板内容
        """
        self.templates[name] = content
        logger.info(f"Added template: {name}")
    
    def remove_template(self, name: str) -> bool:
        """
        删除模板
        
        Args:
            name: 模板名称
            
        Returns:
            bool: 是否成功删除
        """
        if name in self.templates:
            del self.templates[name]
            logger.info(f"Removed template: {name}")
            return True
        return False
    
    def list_templates(self) -> Dict[str, str]:
        """
        获取所有模板
        
        Returns:
            Dict[str, str]: 模板字典
        """
        return self.templates
    
    def save_template_to_file(self, name: str, file_path: Optional[Path] = None) -> bool:
        """
        将模板保存到文件
        
        Args:
            name: 模板名称
            file_path: 文件路径，如果为None则使用默认目录和模板名称
            
        Returns:
            bool: 是否成功保存
        """
        if name not in self.templates:
            logger.error(f"Template '{name}' not found")
            return False
            
        try:
            # 如果没有指定路径，使用默认路径
            if file_path is None:
                file_path = self.template_dir / f"{name}.txt"
                
            # 确保目录存在
            os.makedirs(file_path.parent, exist_ok=True)
                
            # 写入文件
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(self.templates[name])
                
            logger.info(f"Template '{name}' saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving template '{name}' to file: {str(e)}")
            return False
    
    def get_template_for_content_type(self, content_type: str) -> str:
        """
        根据内容类型获取合适的模板
        
        Args:
            content_type: 内容类型
            
        Returns:
            str: 合适的模板
        """
        content_type = content_type.lower()
        
        # 内容类型到模板类型的映射
        type_mapping = {
            "chart": TemplateType.CHART.value,
            "diagram": TemplateType.DIAGRAM.value,
            "formula": TemplateType.FORMULA.value,
            "table": TemplateType.TABLE.value,
            "text": TemplateType.TEXT_HEAVY.value,
            "handwritten": TemplateType.TEXT_HEAVY.value,
            "unknown": TemplateType.GENERAL.value,
            "mixed": TemplateType.GENERAL.value,
        }
        
        template_type = type_mapping.get(content_type, TemplateType.GENERAL.value)
        return self.get_template(template_type)


# 创建单例实例，方便直接导入使用
prompt_template_manager = PromptTemplateManager()