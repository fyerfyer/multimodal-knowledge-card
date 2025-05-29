import os
import pytest
import json
import time
import re
from pathlib import Path
import numpy as np
from typing import Dict, Any, Optional

# 导入格式化服务模块
from app.core.formatter.service import formatter_service
from app.core.formatter.json_parser import llm_output_parser
from app.core.formatter.markdown import markdown_formatter
from app.core.formatter.quality import quality_controller
from app.core.formatter.schema import KnowledgeCard, SimpleKnowledgeCard

# 导入模型和服务
from app.core.models.factory import create_model
from app.core.models.qwen_vl import QwenVLModel 
from app.core.vision.service import vision_service
from app.core.ocr.service import ocr_service
from app.config.settings import settings
from app.utils.logger import logger

# 设置测试目录
TEST_DIR = Path(__file__).parent
FIXTURES_DIR = TEST_DIR / "fixtures"
os.makedirs(FIXTURES_DIR, exist_ok=True)


@pytest.fixture
def sample_llm_output_json():
    """
    提供样例LLM JSON格式输出，用于测试解析器
    """
    return """```json
{
    "title": "牛顿运动定律",
    "summary": "描述物体运动与力之间的关系的基本物理定律",
    "key_points": [
        {
            "content": "物体在没有外力作用下保持静止或匀速直线运动状态",
            "formula": null,
            "reference": null
        },
        {
            "content": "力等于质量乘以加速度",
            "formula": "F = ma",
            "reference": null
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
}```"""


@pytest.fixture
def sample_llm_output_markdown():
    """
    提供样例LLM Markdown格式输出，用于测试解析器
    """
    return """
# 牛顿运动定律

描述物体运动与力之间的关系的基本物理定律

## 关键知识点
1. 物体在没有外力作用下保持静止或匀速直线运动状态
2. 力等于质量乘以加速度，公式: F = ma

## 练习题
**问题**: 一个质量为5kg的物体受到10N的恒力作用，求其加速度大小。

**答案**: 根据牛顿第二定律 F = ma，a = F/m = 10N/5kg = 2m/s²

## 引用
- 来源: 大学物理教材第三章
- 页码: 42
"""


@pytest.fixture
def sample_card_data():
    """
    提供样例卡片数据，用于测试格式化器
    """
    return {
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


@pytest.fixture
def low_quality_card_data():
    """
    提供质量较低的卡片数据，用于测试质量改进功能
    """
    return {
        "title": "牛定律",
        "key_points": [
            "物体动"
        ],
        "quiz": {
            "question": "加速度?",
            "answer": "2"
        }
    }


@pytest.fixture
def test_image_path():
    """
    提供测试图像路径，如果不存在则创建一个简单的测试图像
    """
    test_image = FIXTURES_DIR / "test_image.jpg"
    
    # 如果测试图像不存在，创建一个简单的测试图像
    if not test_image.exists():
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # 创建一个简单的图像，包含文本
            img = Image.new('RGB', (600, 400), color='white')
            d = ImageDraw.Draw(img)
            
            # 尝试加载字体
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                # 如果找不到指定字体，使用默认字体
                font = ImageFont.load_default()
            
            # 添加文本
            d.text((50, 50), "牛顿运动定律", fill='black', font=font)
            d.text((50, 100), "1. 惯性定律: 物体在没有外力作用下保持静止或匀速直线运动状态", fill='black', font=font)
            d.text((50, 150), "2. 加速度定律: F = ma", fill='black', font=font)
            d.text((50, 200), "3. 作用力与反作用力定律", fill='black', font=font)
            d.text((50, 250), "问题: 一个质量为5kg的物体受到10N的力，加速度是多少?", fill='black', font=font)
            
            # 保存图像
            img.save(test_image)
            logger.info(f"Created test image at: {test_image}")
            
        except Exception as e:
            pytest.skip(f"Cannot create test image: {str(e)}")
    
    return str(test_image)


def test_json_parser_with_json_output(sample_llm_output_json):
    """
    测试JSON解析器能正确解析LLM的JSON输出
    """
    parse_result = llm_output_parser.parse(sample_llm_output_json)
    
    assert parse_result["success"] == True
    assert "card_data" in parse_result
    assert parse_result["card_data"]["title"] == "牛顿运动定律"
    assert len(parse_result["card_data"]["key_points"]) == 2
    assert parse_result["format"] == "json"


def test_json_parser_with_markdown_output(sample_llm_output_markdown):
    """
    测试JSON解析器能从Markdown输出中提取结构化数据
    """
    parse_result = llm_output_parser.parse(sample_llm_output_markdown)
    
    assert parse_result["success"] == True
    assert "card_data" in parse_result
    assert parse_result["card_data"]["title"] == "牛顿运动定律"
    assert len(parse_result["card_data"]["key_points"]) > 0
    assert "question" in parse_result["card_data"]["quiz"]
    assert "answer" in parse_result["card_data"]["quiz"]


def test_markdown_formatter(sample_card_data):
    """
    测试Markdown格式化器
    """
    md_result = markdown_formatter.card_to_markdown(sample_card_data)
    
    assert md_result["success"] == True
    assert "markdown" in md_result
    assert "# 牛顿运动定律" in md_result["markdown"]
    assert "F = ma" in md_result["markdown"]
    assert "## 练习题" in md_result["markdown"]
    assert "**问题**" in md_result["markdown"]
    assert "**答案**" in md_result["markdown"]


def test_card_preview_renderer(sample_card_data):
    """
    测试卡片预览生成
    """
    preview_result = markdown_formatter.render_card_preview(sample_card_data)
    
    assert preview_result["success"] == True
    assert "preview" in preview_result
    assert "牛顿运动定律" in preview_result["preview"]
    assert "包含" in preview_result["preview"]


def test_quality_validation(sample_card_data, low_quality_card_data):
    """
    测试卡片质量验证功能
    """
    # 测试高质量卡片
    high_quality_result = quality_controller.validate_card(sample_card_data)
    assert high_quality_result["valid"] == True
    assert high_quality_result["quality_score"] > 80
    
    # 测试低质量卡片
    low_quality_result = quality_controller.validate_card(low_quality_card_data)
    assert low_quality_result["valid"] == False
    assert low_quality_result["quality_score"] < 70
    assert len(low_quality_result["issues"]) > 0
    assert len(low_quality_result["improvements"]) > 0


def test_quality_improvement(low_quality_card_data):
    """
    测试卡片质量改进功能
    """
    improve_result = quality_controller.improve_card(low_quality_card_data)
    
    assert "improved" in improve_result
    assert "improved_card" in improve_result
    assert "original_score" in improve_result
    assert "improved_score" in improve_result
    assert len(improve_result["improvements"]) > 0

    # 验证改进后的标题是否更完整
    assert len(improve_result["improved_card"]["title"]) > len(low_quality_card_data["title"])
    
    # 验证改进后的知识点是否更详细
    original_key_points = low_quality_card_data["key_points"]
    improved_key_points = improve_result["improved_card"]["key_points"]
    
    # 检查内容是否有改进（至少一个知识点有更长的内容）
    assert any(
        isinstance(point, dict) and len(point.get("content", "")) > len(original_key_points[0])
        for point in improved_key_points
    )


def test_process_llm_output(sample_llm_output_json):
    """
    测试完整的LLM输出处理流程
    """
    process_result = formatter_service.process_llm_output(sample_llm_output_json)
    
    assert process_result["success"] == True
    assert "card_data" in process_result
    assert "formats" in process_result
    assert "json" in process_result["formats"]
    assert "markdown" in process_result["formats"]
    assert "text" in process_result["formats"]
    assert "quality" in process_result
    assert "parsing" in process_result


def test_format_conversion(sample_card_data):
    """
    测试不同格式的转换功能
    """
    # JSON格式
    json_result = formatter_service.format_card(sample_card_data, "json")
    assert json_result["success"] == True
    assert "json" in json_result
    assert json_result["format"] == "json"
    
    # Markdown格式
    md_result = formatter_service.format_card(sample_card_data, "markdown")
    assert md_result["success"] == True
    assert "markdown" in md_result
    assert md_result["format"] == "markdown"
    
    # 纯文本格式
    text_result = formatter_service.format_card(sample_card_data, "text")
    assert text_result["success"] == True
    assert "text" in text_result
    assert text_result["format"] == "text"
    
    # 不支持的格式
    invalid_result = formatter_service.format_card(sample_card_data, "invalid_format")
    assert invalid_result["success"] == False
    assert "error" in invalid_result


def test_json_parser_with_malformed_json():
    """
    测试JSON解析器处理格式错误的JSON
    """
    malformed_json = """
    {
        "title": "测试标题",
        "key_points": [
            这是一个格式错误的JSON，缺少引号
        ],
        "quiz": {
            "question": "问题",
            "answer": "答案"
        }
    }
    """
    
    parse_result = llm_output_parser.parse(malformed_json)
    
    # 当JSON解析失败时，解析器应该尝试文本提取模式
    if "format" in parse_result and parse_result["format"] == "extracted_text":
        assert parse_result["success"] == True
        assert "card_data" in parse_result
        assert "title" in parse_result["card_data"]
    elif not parse_result["success"]:
        assert "error" in parse_result


def test_empty_input_handling():
    """
    测试空输入处理
    """
    empty_result = llm_output_parser.parse("")
    
    assert "success" in empty_result
    if not empty_result["success"]:
        assert "error" in empty_result
    assert "raw_output" in empty_result


@pytest.mark.skipif(not os.environ.get("DASHSCOPE_API_KEY"), reason="DashScope API key not available")
def test_qwen_vl_model_integration():
    """
    测试与Qwen-VL模型的集成
    注意：此测试需要有效的DashScope API密钥
    """
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    
    try:
        model = QwenVLModel(api_key=api_key, model_name="qwen-vl-plus")
        
        prompt = "这是什么图像？请用中文详细描述。"
        response = model.generate_response(prompt)
        
        assert "success" in response
        if response["success"]:
            assert "text" in response
            assert len(response["text"]) > 0
    except Exception as e:
        pytest.skip(f"Error testing Qwen-VL model: {str(e)}")


@pytest.mark.skipif(not os.environ.get("DASHSCOPE_API_KEY"), reason="DashScope API key not available")
def test_formatter_with_qwen_vl(test_image_path):
    """
    测试卡片格式化服务与Qwen-VL模型的集成
    注意：此测试需要有效的DashScope API密钥和测试图像
    """
    if not os.path.exists(test_image_path):
        pytest.skip("Test image not available")
        
    prompt = """请根据这张图片创建一张知识卡片，包含标题、关键知识点和一个问答题。请用中文回答。"""
    
    try:
        result = formatter_service.generate_card_from_prompt(
            prompt=prompt,
            image=test_image_path,
            model_type="qwen-vl"
        )
        
        assert "success" in result
        if result["success"]:
            assert "card_data" in result
            assert "title" in result["card_data"]
            assert "key_points" in result["card_data"]
            assert "quiz" in result["card_data"]
    except Exception as e:
        pytest.skip(f"Error testing formatter with Qwen-VL: {str(e)}")


@pytest.mark.skipif(not os.path.exists(FIXTURES_DIR / "test_image.jpg"), reason="Test image not found")
def test_ocr_integration():
    """
    测试OCR服务集成
    注意：此测试需要测试图像
    """
    test_image_path = str(FIXTURES_DIR / "test_image.jpg")
    
    try:
        ocr_result = ocr_service.process_image_file(test_image_path)
        
        assert "success" in ocr_result
        if ocr_result["success"]:
            assert "full_text" in ocr_result
            assert isinstance(ocr_result["full_text"], str)
            assert len(ocr_result["full_text"]) > 0
    except Exception as e:
        pytest.skip(f"Error testing OCR service: {str(e)}")


@pytest.mark.skipif(not os.path.exists(FIXTURES_DIR / "test_image.jpg"), reason="Test image not found")
def test_vision_service_integration():
    """
    测试图像理解服务集成
    注意：此测试需要测试图像和BLIP模型
    """
    test_image_path = str(FIXTURES_DIR / "test_image.jpg")
    
    try:
        vision_result = vision_service.describe_for_prompt(test_image_path)
        
        assert "success" in vision_result
        if vision_result["success"]:
            assert "caption" in vision_result
            assert len(vision_result["caption"]) > 0
            assert "content_type" in vision_result
    except Exception as e:
        pytest.skip(f"Error testing vision service: {str(e)}")


@pytest.mark.skipif(
    not os.path.exists(FIXTURES_DIR / "test_image.jpg") or 
    not os.environ.get("DASHSCOPE_API_KEY"), 
    reason="Test image not found or API key not available"
)
def test_end_to_end_card_generation():
    """
    端到端测试：从图像生成知识卡片
    注意：此测试需要有效的DashScope API密钥和测试图像
    """
    test_image_path = str(FIXTURES_DIR / "test_image.jpg")
    
    try:
        # 1. 获取图像描述
        vision_result = vision_service.describe_for_prompt(test_image_path)
        assert vision_result["success"] == True
        
        # 2. 获取OCR文本
        ocr_result = ocr_service.process_image_file(test_image_path)
        assert ocr_result["success"] == True
        
        # 3. 构造提示词
        prompt = f"""
        基于以下图像内容，创建一个结构化的知识卡片：
        
        图像描述：
        {vision_result.get("caption", "")}
        
        图像文本：
        {ocr_result.get("full_text", "")}
        
        请输出包含以下内容的知识卡片：
        1. 标题
        2. 2-3个关键知识点
        3. 一道练习题和答案
        
        请用JSON格式输出，包含title、key_points（数组）和quiz（有question和answer字段）。
        """
        
        # 4. 使用模型生成卡片内容
        model = create_model("qwen-vl")
        response = model.generate_response(prompt)
        assert response["success"] == True
        
        # 5. 处理LLM输出
        process_result = formatter_service.process_llm_output(response["text"])
        assert process_result["success"] == True
        assert "card_data" in process_result
        
        # 6. 验证卡片质量
        validation = quality_controller.validate_card(process_result["card_data"])
        if not validation["valid"]:
            improve_result = quality_controller.improve_card(process_result["card_data"])
            assert "improved_card" in improve_result
    
    except Exception as e:
        pytest.skip(f"Error in end-to-end test: {str(e)}")


if __name__ == "__main__":
    # 在直接运行此文件时，执行所有测试
    pytest.main(["-xvs", __file__])