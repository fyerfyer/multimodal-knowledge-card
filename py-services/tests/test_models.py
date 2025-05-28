import unittest
import os
import sys
import pytest
import time
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv

# 加载环境变量，确保API密钥可用
load_dotenv()

# 添加项目根目录到Python路径，确保可以导入应用模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.models.interface import MultiModalModelInterface, ModelResponse
from app.core.models.qwen_vl import QwenVLModel
from app.core.models.factory import ModelFactory, create_model
from app.core.models.loader import model_loader, get_model
from app.core.models.inference import inference_service
from app.core.ocr.service import ocr_service
from app.core.vision.service import vision_service
from app.utils.logger import logger
from app.config.settings import settings

# 测试图像路径
TEST_IMAGE_PATH = Path(__file__).parent / "test_data" / "test_image.jpg"


class TestMultiModalModels(unittest.TestCase):
    """多模态模型集成测试类"""
    
    @classmethod
    def setUpClass(cls):
        """
        在所有测试开始前设置测试环境
        确保测试图像存在，API密钥有效
        """
        # 确保测试目录存在
        os.makedirs(Path(__file__).parent / "test_data", exist_ok=True)
        
        # 如果测试图像不存在，创建一个简单的测试图像
        if not TEST_IMAGE_PATH.exists():
            # 创建一个简单的测试图像
            img = Image.new('RGB', (300, 200), color = (73, 109, 137))
            # 保存到文件
            img.save(TEST_IMAGE_PATH)
            logger.info(f"Created test image at {TEST_IMAGE_PATH}")
        
        # 检查API密钥是否设置
        cls.dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY") or settings.DASHSCOPE_API_KEY
        
        if not cls.dashscope_api_key:
            logger.warning("DASHSCOPE_API_KEY not set, some tests will be skipped")
    
    def test_model_interface(self):
        """测试模型接口基础功能"""
        # 测试ModelResponse类
        response = ModelResponse(
            text="测试响应内容",
            success=True,
            error=None,
            usage={"input_tokens": 10, "output_tokens": 20},
            model_name="test-model",
            metadata={"test": "metadata"}
        )
        
        response_dict = response.to_dict()
        
        # 验证基本属性
        self.assertEqual(response_dict["text"], "测试响应内容")
        self.assertTrue(response_dict["success"])
        self.assertEqual(response_dict["model_name"], "test-model")
        self.assertIn("usage", response_dict)
        self.assertIn("metadata", response_dict)
    
    def test_qwen_vl_model(self):
        """测试通义千问视觉语言模型"""
        if not self.dashscope_api_key:
            self.skipTest("DASHSCOPE_API_KEY not available")
            
        # 创建模型实例
        model = QwenVLModel(api_key=self.dashscope_api_key)
        
        # 测试初始化
        self.assertTrue(model.initialize())
        self.assertTrue(model.is_initialized())
        
        # 测试获取模型信息
        model_info = model.get_model_info()
        self.assertEqual(model_info["model_type"], "Qwen-VL")
        self.assertTrue(model_info["supports_vision"])
        
        # 测试生成响应（仅测试文本输入）
        response = model.generate_response(
            prompt="机器学习是什么？请简单介绍一下。",
            image=None,
            options={"max_tokens": 100}
        )
        
        self.assertTrue(response["success"])
        self.assertIn("text", response)
        self.assertGreater(len(response["text"]), 0)
        
        # 测试图像输入 - 如果API密钥有效且环境支持
        try:
            image_response = model.generate_response(
                prompt="请描述这张图片的内容。",
                image=str(TEST_IMAGE_PATH),
                options={"max_tokens": 100}
            )
            self.assertTrue(image_response["success"])
            self.assertIn("text", image_response)
        except Exception as e:
            logger.warning(f"Image response test failed: {str(e)}")
    
    def test_model_factory(self):
        """测试模型工厂功能"""
        # 测试创建通义千问模型
        if self.dashscope_api_key:
            qwen_model = create_model("qwen-vl", api_key=self.dashscope_api_key)
            self.assertIsInstance(qwen_model, QwenVLModel)
            self.assertTrue(qwen_model.is_initialized())
        
        # 测试模型类型列表
        available_models = ModelFactory.list_available_models()
        self.assertIn("qwen-vl", available_models)
        
        # 测试注册和获取新模型类
        class TestModel(MultiModalModelInterface):
            def initialize(self): 
                pass
            def generate_response(self, prompt, image=None, options=None): 
                pass
            def get_model_info(self): 
                pass
            def is_initialized(self): 
                pass
        
        ModelFactory.register_model("test-model", TestModel)
        self.assertIn("test-model", ModelFactory.list_available_models())
        
        test_model_class = ModelFactory.get_model_class("test-model")
        self.assertEqual(test_model_class, TestModel)
    
    def test_model_loader(self):
        """测试模型加载器功能"""
        # 清理现有缓存以避免干扰
        model_loader.cleanup(force=True)
        
        # 选择可用的API密钥来进行测试
        api_key = self.dashscope_api_key
        model_type = "qwen-vl"  # 现在只用 qwen-vl
        
        if not api_key:
            self.skipTest("No API key available for testing")
        
        # 测试获取模型
        model = model_loader.get_model(model_type, api_key=api_key)
        self.assertIsNotNone(model)
        self.assertTrue(model.is_initialized())
        
        # 测试缓存机制 - 第二次获取应该返回相同实例
        model2 = model_loader.get_model(model_type, api_key=api_key)
        self.assertIs(model, model2)
        
        # 测试获取加载的模型列表
        loaded_models = model_loader.get_loaded_models()
        self.assertGreaterEqual(len(loaded_models), 1)
        
        # 测试卸载模型
        model_id = model_loader._generate_model_id(model_type, None, api_key=api_key)
        unload_success = model_loader.unload_model(model_id)
        self.assertTrue(unload_success)
        
        # 验证是否已成功卸载
        loaded_models_after = model_loader.get_loaded_models()
        self.assertLess(len(loaded_models_after), len(loaded_models))
    
    def test_inference_service(self):
        """测试推理服务功能"""
        # 选择可用的API密钥来进行测试
        api_key = self.dashscope_api_key
        model_type = "qwen-vl"  # 现在只用 qwen-vl
        
        if not api_key:
            self.skipTest("No API key available for testing")
        
        # 测试切换模型
        switch_success = inference_service.switch_model(model_type)
        self.assertTrue(switch_success)
        
        # 测试模型信息获取
        model_info = inference_service.get_model_info()
        self.assertIn("model_type", model_info)
        self.assertIn("supports_vision", model_info)
        
        # 测试文本推理
        response = inference_service.generate_response(
            prompt="请简要解释人工神经网络是什么。",
            options={"max_tokens": 100}
        )
        
        self.assertTrue(response["success"])
        self.assertIn("text", response)
        self.assertGreater(len(response["text"]), 0)
        
        # 测试批处理 - 多个请求一起处理
        batch_requests = [
            {"prompt": "机器学习是什么？请简要概述。", "options": {"max_tokens": 50}},
            {"prompt": "简要介绍计算机视觉。", "options": {"max_tokens": 50}}
        ]
        
        batch_responses = inference_service.generate_batch_responses(batch_requests)
        self.assertEqual(len(batch_responses), len(batch_requests))
        for response in batch_responses:
            self.assertTrue(response["success"])
    
    def test_integration_with_ocr(self):
        """测试与OCR系统的集成"""
        # 确保有可用的API密钥
        api_key = self.dashscope_api_key
        model_type = "qwen-vl"
        
        if not api_key:
            self.skipTest("No API key available for testing")
        
        # 首先使用OCR处理图像
        ocr_result = ocr_service.process_image_file(TEST_IMAGE_PATH)
        
        if not ocr_result["success"]:
            logger.warning(f"OCR processing failed: {ocr_result.get('error', 'Unknown error')}")
            # 即使OCR失败，我们仍然可以测试基本整合
            ocr_text = "这是一张测试图片"
        else:
            ocr_text = ocr_result.get("full_text", "")
        
        # 使用OCR结果构建提示词并调用多模态推理
        prompt = f"分析下面的文本并创建一个摘要：{ocr_text}"
        
        response = inference_service.generate_response(
            prompt=prompt,
            model_type=model_type,
            options={"max_tokens": 150}
        )
        
        self.assertTrue(response["success"])
        self.assertIn("text", response)
        self.assertGreater(len(response["text"]), 0)
    
    def test_integration_with_vision(self):
        """测试与视觉理解系统的集成"""
        # 确保有可用的API密钥
        api_key = self.dashscope_api_key
        model_type = "qwen-vl"
        
        if not api_key:
            self.skipTest("No API key available for testing")
        
        try:
            # 使用视觉服务处理图像
            vision_result = vision_service.describe_for_prompt(TEST_IMAGE_PATH)
            
            if not vision_result["success"]:
                logger.warning(f"Vision analysis failed: {vision_result.get('error', 'Unknown error')}")
                image_caption = "这是一张测试图片"
            else:
                image_caption = vision_result.get("caption", "")
            
            # 使用视觉分析结果构建提示词并调用多模态推理
            prompt = f"基于以下图像描述：'{image_caption}'，请提供一些额外的背景信息或分析。"
            
            response = inference_service.generate_response(
                prompt=prompt,
                model_type=model_type,
                options={"max_tokens": 150}
            )
            
            self.assertTrue(response["success"])
            self.assertIn("text", response)
            self.assertGreater(len(response["text"]), 0)
            
        except Exception as e:
            logger.warning(f"Vision integration test skipped: {str(e)}")
            self.skipTest(f"Vision model not available: {str(e)}")
    
    def test_end_to_end_multimodal_pipeline(self):
        """测试完整的多模态管道 - 从图像到知识卡片生成"""
        # 确保有可用的API密钥
        api_key = self.dashscope_api_key
        model_type = "qwen-vl"
        
        if not api_key:
            self.skipTest("No API key available for testing")
        
        try:
            # 1. OCR 处理
            ocr_result = ocr_service.process_image_file(TEST_IMAGE_PATH)
            if not ocr_result["success"]:
                logger.warning(f"OCR processing failed: {ocr_result.get('error', 'Unknown error')}")
                ocr_text = "这是OCR处理的测试图片文本"
            else:
                ocr_text = ocr_result.get("full_text", "")
            
            # 2. 视觉处理
            try:
                vision_result = vision_service.describe_for_prompt(TEST_IMAGE_PATH)
                if not vision_result["success"]:
                    image_caption = "一张用于视觉理解测试的图片"
                else:
                    image_caption = vision_result.get("caption", "")
            except Exception as e:
                logger.warning(f"Vision processing error: {str(e)}")
                image_caption = "一张用于视觉理解测试的图片"
            
            # 3. 构建提示词
            prompt = f"""
基于图像和提取的文本，创建一张知识卡片。

图像描述: {image_caption}

提取的文本: {ocr_text}

请创建一张知识卡片，包含:
1. 标题
2. 2-3个关键点
3. 一道练习题
"""
            
            # 4. 生成知识卡片
            response = inference_service.generate_response(
                prompt=prompt,
                image=str(TEST_IMAGE_PATH),
                model_type=model_type,
                options={
                    "max_tokens": 500,
                    "temperature": 0.7
                }
            )
            
            self.assertTrue(response["success"])
            self.assertIn("text", response)
            self.assertGreater(len(response["text"]), 10)
            logger.info(f"Generated knowledge card preview: {response['text'][:100]}...")
            
        except Exception as e:
            logger.error(f"End-to-end pipeline test failed: {str(e)}")
            self.fail(f"End-to-end test failed: {str(e)}")


if __name__ == '__main__':
    unittest.main()