from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class QuizItem(BaseModel):
    """测验项模型，包含问题和答案"""
    question: str = Field(..., description="问题内容")
    answer: str = Field(..., description="问题答案")


class KeyPoint(BaseModel):
    """知识点模型，包含内容和可选的公式或引用"""
    content: str = Field(..., description="知识点内容")
    formula: Optional[str] = Field(None, description="可选的公式")
    reference: Optional[str] = Field(None, description="可选的引用或来源")


class KnowledgeCard(BaseModel):
    """知识卡片完整模型"""
    title: str = Field(..., description="卡片标题")
    summary: Optional[str] = Field(None, description="内容摘要")
    key_points: List[KeyPoint] = Field(..., description="关键知识点列表")
    quiz: QuizItem = Field(..., description="练习题")
    references: Optional[Dict[str, Any]] = Field(None, description="引用和来源")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据，如生成时间、使用的模型等")

    class Config:
        """模型配置"""
        schema_extra = {
            "example": {
                "title": "牛顿运动定律",
                "summary": "描述物体运动与力之间的关系的基本物理定律",
                "key_points": [
                    {
                        "content": "物体在没有外力作用下保持静止或匀速直线运动状态",
                        "formula": None,
                        "reference": None
                    },
                    {
                        "content": "力等于质量乘以加速度",
                        "formula": "F = ma",
                        "reference": None
                    }
                ],
                "quiz": {
                    "question": "一个质量为5kg的物体受到10N的恒力作用，求其加速度大小。",
                    "answer": "根据牛顿第二定律 F = ma，a = F/m = 10N/5kg = 2m/s²"
                },
                "references": {
                    "source": "大学物理教材第三章",
                    "page": "42"
                },
                "metadata": {
                    "generated_at": "2023-05-29T10:30:00Z",
                    "model_used": "Qwen-VL-Plus"
                }
            }
        }


class SimpleKnowledgeCard(BaseModel):
    """简化版知识卡片模型，用于灵活处理LLM输出"""
    title: str = Field(..., description="卡片标题")
    key_points: List[str] = Field(..., description="关键知识点列表（简化为字符串列表）")
    quiz: Dict[str, str] = Field(..., description="练习题（简化为问题和答案的字典）")
    additional_info: Optional[Dict[str, Any]] = Field(None, description="任何额外信息")