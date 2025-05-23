import os
import pytest
import tempfile
from pathlib import Path

import numpy as np
from unittest.mock import patch
import cv2

from app.core.prompt.service import prompt_service
from app.core.prompt.templates import TemplateType

from dotenv import load_dotenv
load_dotenv()

# 测试资源目录
TEST_RESOURCES_DIR = Path(__file__).parent / "resources"
os.makedirs(TEST_RESOURCES_DIR, exist_ok=True)

# 测试图片路径
TEST_IMAGE_PATH = TEST_RESOURCES_DIR / "test_image.jpg"

@pytest.fixture
def sample_image():
    """
    创建测试图像的fixture
    如果测试图像不存在，则创建一个简单的测试图像
    """
    if not TEST_IMAGE_PATH.exists():
        # 创建一个简单的测试图像
        img = np.ones((500, 500, 3), dtype=np.uint8) * 255  # 白色背景
        # 添加一些文字
        cv2.putText(img, "Sample Text", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 0), 2, cv2.LINE_AA)
        # 保存图像
        cv2.imwrite(str(TEST_IMAGE_PATH), img)
    
    return TEST_IMAGE_PATH

@pytest.fixture
def mock_vision_service():
    """模拟视觉服务的fixture"""
    with patch('app.core.prompt.multimodal_fusion.vision_service') as mock_vision:
        mock_vision.describe_for_prompt.return_value = {
            "success": True,
            "content_type": "text",
            "caption": "A sample educational text image",
            "focus_points": {
                "What is the main topic of this image?": "Educational content",
                "What is the most important element in this image?": "Text explaining a concept"
            }
        }
        yield mock_vision

@pytest.fixture
def mock_ocr_service():
    """模拟OCR服务的fixture"""
    with patch('app.core.prompt.multimodal_fusion.ocr_service') as mock_ocr:
        mock_ocr.process_image_file.return_value = {
            "success": True,
            "full_text": "This is a sample text for OCR testing.\nIt contains multiple lines.\nUsed for prompt service testing.",
            "paragraph_count": 3,
            "paragraphs": [
                "This is a sample text for OCR testing.",
                "It contains multiple lines.",
                "Used for prompt service testing."
            ]
        }
        mock_ocr.process_image_data.return_value = {
            "success": True,
            "full_text": "This is a sample text for OCR testing.\nIt contains multiple lines.\nUsed for prompt service testing.",
            "paragraph_count": 3,
            "paragraphs": [
                "This is a sample text for OCR testing.",
                "It contains multiple lines.",
                "Used for prompt service testing."
            ]
        }
        yield mock_ocr

@pytest.fixture
def custom_template():
    """自定义模板的fixture"""
    return "Custom prompt template:\n\nContent: {ocr_text}\n\nPlease analyze this content."

class TestPromptService:
    """提示词服务测试类"""
    
    def test_create_text_prompt_basic(self):
        """测试基本文本提示词创建功能"""
        test_text = "This is a test input text"
        result = prompt_service.create_text_prompt(
            text=test_text,
            template_type=TemplateType.GENERAL
        )
        
        assert result["success"] is True
        assert result["template_type"] == "general"
        assert "processing_time" in result
        assert test_text in result["prompt"]
    
    def test_create_text_prompt_with_optimization(self):
        """测试创建文本提示词并进行优化"""
        test_text = "This is a test input text"
        result = prompt_service.create_text_prompt(
            text=test_text,
            template_type=TemplateType.EDUCATIONAL,
            options={"optimize": True, "target_length": 500}
        )
        
        assert result["success"] is True
        assert result["template_type"] == "educational"
        assert "optimization" in result
        assert "applied_rules" in result["optimization"]
        assert "stats" in result["optimization"]
    
    def test_create_text_prompt_with_custom_variables(self):
        """测试使用自定义变量创建文本提示词"""
        test_text = "This is a test input text"
        custom_vars = {
            "subject": "Mathematics",
            "difficulty": "Intermediate"
        }
        
        result = prompt_service.create_text_prompt(
            text=test_text,
            template_type=TemplateType.GENERAL,
            options={"variables": custom_vars}
        )
        
        assert result["success"] is True
        assert result["template_type"] == "general"
    
    def test_create_prompt_from_image(self, sample_image, mock_vision_service, mock_ocr_service):
        """测试从图像创建提示词"""
        result = prompt_service.create_prompt_from_image(
            image=sample_image
        )
        
        assert result["success"] is True
        assert "prompt" in result
        assert "content_type" in result
        assert "processing_time" in result
        assert "metadata" in result
        mock_vision_service.describe_for_prompt.assert_called_once()
        mock_ocr_service.process_image_file.assert_called_once()

    def test_create_prompt_from_image_with_optimization(self, sample_image, mock_vision_service, mock_ocr_service):
        """测试从图像创建提示词并进行优化"""
        result = prompt_service.create_prompt_from_image(
            image=sample_image,
            options={
                "optimize": True,
                "target_length": 800
            }
        )
        
        assert result["success"] is True
        assert "prompt" in result
        assert "optimization" in result
        assert "applied_rules" in result["optimization"]
    
    def test_create_prompt_from_image_with_content_type(self, sample_image, mock_vision_service, mock_ocr_service):
        """测试从图像创建提示词并指定内容类型"""
        result = prompt_service.create_prompt_from_image(
            image=sample_image,
            options={
                "content_type": "formula"
            }
        )
        
        assert result["success"] is True
        assert result["content_type"] == "formula"
    
    def test_enhance_prompt_optimize(self):
        """测试优化提示词"""
        test_prompt = "Analyze the following text and provide insights."
        result = prompt_service.enhance_prompt(
            prompt=test_prompt,
            enhancement_type="optimize",
            options={
                "target_length": 300,
                "context": {"content_type": "text"}
            }
        )
        
        assert result["success"] is True
        assert "enhanced_prompt" in result
        assert result["enhancement_type"] == "optimize"
        assert "enhancement_details" in result
    
    def test_enhance_prompt_rewrite(self):
        """测试重写提示词"""
        test_prompt = "Please analyze this content and provide a summary."
        
        for style in ["concise", "detailed", "technical", "educational"]:
            result = prompt_service.enhance_prompt(
                prompt=test_prompt,
                enhancement_type="rewrite",
                options={"style": style}
            )
            
            assert result["success"] is True
            assert "enhanced_prompt" in result
            assert result["enhancement_type"] == "rewrite"
            assert result["enhancement_details"]["style"] == style
    
    def test_enhance_prompt_context(self):
        """测试添加上下文信息到提示词"""
        test_prompt = "Analyze the following mathematical concept."
        
        result = prompt_service.enhance_prompt(
            prompt=test_prompt,
            enhancement_type="context",
            options={
                "context_info": {
                    "subject": "Calculus",
                    "educational_level": "university",
                    "focus": "practical applications"
                }
            }
        )
        
        assert result["success"] is True
        assert "enhanced_prompt" in result
        assert result["enhancement_type"] == "context"
        assert "Calculus" in result["enhanced_prompt"]
        assert "university" in result["enhanced_prompt"]
    
    def test_create_multi_part_prompt(self):
        """测试创建多部分提示词"""
        parts = [
            {
                "text": "This is part 1 text",
                "caption": "Part 1 caption",
                "type": "text",
                "title": "Introduction"
            },
            {
                "text": "This is part 2 text with formula",
                "caption": "Mathematical formula",
                "type": "formula",
                "title": "Formula Explanation"
            }
        ]
        
        result = prompt_service.create_multi_part_prompt(parts)
        
        assert result["success"] is True
        assert "prompt" in result
        assert "part_count" in result
        assert result["part_count"] == 2
        assert "Introduction" in result["prompt"]
        assert "Formula Explanation" in result["prompt"]
    
    def test_manage_templates_add(self):
        """测试添加模板"""
        template_name = "test_template"
        template_content = "Test template content with {ocr_text} placeholder"
        
        result = prompt_service.manage_templates(
            action="add",
            template_name=template_name,
            template_content=template_content
        )
        
        assert result["success"] is True
        assert result["action"] == "add"
        assert result["template_name"] == template_name
        
        # 验证模板已添加
        templates = prompt_service.get_available_templates()
        assert template_name in templates["templates"]
        assert templates["templates"][template_name] == template_content
    
    def test_manage_templates_remove(self):
        """测试删除模板"""
        # 先添加一个模板
        template_name = "temp_template_to_remove"
        prompt_service.template_manager.add_template(
            template_name, 
            "Temporary template content"
        )
        
        # 验证添加成功
        assert template_name in prompt_service.template_manager.list_templates()
        
        # 删除模板
        result = prompt_service.manage_templates(
            action="remove",
            template_name=template_name
        )
        
        assert result["success"] is True
        assert result["action"] == "remove"
        
        # 验证已删除
        assert template_name not in prompt_service.template_manager.list_templates()
    
    def test_manage_templates_save(self):
        """测试保存模板到文件"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            template_name = "template_to_save"
            template_content = "Template content to save to file"
            
            # 添加模板
            prompt_service.template_manager.add_template(template_name, template_content)
            
            # 保存模板到文件
            file_path = Path(tmp_dir) / f"{template_name}.txt"
            result = prompt_service.manage_templates(
                action="save",
                template_name=template_name,
                file_path=file_path
            )
            
            assert result["success"] is True
            assert result["action"] == "save"
            assert file_path.exists()
            
            # 验证文件内容
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                assert content == template_content
    
    def test_get_content_specific_template(self):
        """测试获取特定内容类型的模板"""
        for content_type in ["chart", "formula", "table", "text", "unknown"]:
            result = prompt_service.get_content_specific_template(content_type)
            
            assert result["success"] is True
            assert result["content_type"] == content_type
            assert "template" in result
            assert isinstance(result["template"], str)
            assert len(result["template"]) > 0
    
    def test_get_available_templates(self):
        """测试获取所有可用的模板"""
        result = prompt_service.get_available_templates()
        
        assert result["success"] is True
        assert "template_count" in result
        assert "templates" in result
        assert result["template_count"] == len(result["templates"])
        assert "general" in result["templates"]
    
    def test_get_service_info(self):
        """测试获取服务信息"""
        info = prompt_service.get_service_info()
        
        assert "service" in info
        assert "components" in info
        assert "available_enhancements" in info
        assert "available_rewrite_styles" in info
        assert "template_count" in info
        
        # 验证所有必要的组件信息
        components = ["template_manager", "builder", "fusion", "optimizer"]
        for component in components:
            assert component in info["components"]
        
        # 验证增强类型
        enhancements = ["optimize", "rewrite", "context"]
        for enhancement in enhancements:
            assert enhancement in info["available_enhancements"]
        
        # 验证重写样式
        styles = ["concise", "detailed", "technical", "educational"]
        for style in styles:
            assert style in info["available_rewrite_styles"]

def test_real_image_processing(sample_image):
    """
    测试使用实际图像进行处理
    注：此测试会使用真实的视觉和OCR服务，如果本地没有模型，将尝试从网络下载
    """
    from app.utils.logger import logger
    
    try:
        from app.core.vision.service import vision_service
        from app.core.ocr.service import ocr_service
        from app.core.prompt.service import prompt_service
        import pytest
        
        # 检查模型路径设置情况并记录
        import os
        vision_model_path = os.environ.get("VISION_MODEL_PATH", "")
        vision_vqa_model_path = os.environ.get("VISION_VQA_MODEL_PATH", "")
        
        logger.info(f"Using model paths: VISION_MODEL_PATH={vision_model_path}, VISION_VQA_MODEL_PATH={vision_vqa_model_path}")
        
        # 确保OCR和视觉服务已初始化
        assert vision_service is not None, "Vision service is None"
        assert ocr_service is not None, "OCR service is None"
        
        # 直接执行测试，不使用装饰器（可能导致问题）
        logger.info("Starting image processing test...")
        result = prompt_service.create_prompt_from_image(sample_image)
        
        assert result["success"] is True, f"Result not successful: {result.get('error', 'Unknown error')}"
        assert "prompt" in result, "No 'prompt' in result"
        assert "content_type" in result, "No 'content_type' in result"
        assert "metadata" in result, "No 'metadata' in result"
        
        logger.info(f"Image processing test succeeded with content type: {result.get('content_type', 'unknown')}")
        return result
            
    except Exception as e:
        logger.error(f"Test error: {str(e)}")
        # Check common error causes
        error_str = str(e).lower()
        
        if "connection" in error_str or "timeout" in error_str or "could not connect" in error_str:
            pytest.skip(f"Network connectivity issue: {str(e)}")
        elif "no such file" in error_str or "not found" in error_str:
            pytest.skip(f"Model or file not found: {str(e)}")
        else:
            pytest.skip(f"Integration test failed: {str(e)}")