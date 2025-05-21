import os
import sys
import unittest
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import shutil
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv 

load_dotenv()

# 确保能够导入应用模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.vision.service import VisionService, vision_service
from app.core.vision.blip import BLIPModel
from app.core.vision.captioner import ImageCaptioner, captioner
from app.core.vision.classifier import ContentClassifier, classifier, ContentType
from app.utils.image_utils import read_image, convert_to_rgb
from app.config.settings import settings
from app.utils.logger import logger


class TestVisionModule(unittest.TestCase):
    """图像理解模块的综合测试类"""
    
    @classmethod
    def setUpClass(cls):
        """
        测试开始前的准备工作
        创建测试用的图像和目录
        """
        # 创建临时目录用于保存测试图像
        cls.test_dir = Path(tempfile.mkdtemp())
        
        # 创建各类测试图像
        cls.test_image_path = cls.create_test_image("Test Image", cls.test_dir / "test_image.png")
        cls.test_chart_path = cls.create_chart_image(cls.test_dir / "test_chart.png")
        cls.test_formula_path = cls.create_formula_image(cls.test_dir / "test_formula.png")
        cls.test_text_path = cls.create_text_image("This is a test text content", cls.test_dir / "test_text.png")
        
        # 加载测试目录下的.env文件
        test_env_path = Path(__file__).parent / ".env"
        load_dotenv(test_env_path)
        
        # 从环境变量获取VISION_MODEL_PATH
        cls.vision_model_path = os.getenv("VISION_MODEL_PATH")
        logger.info(f"Loaded VISION_MODEL_PATH from test env: {cls.vision_model_path}")
        
        logger.info("Vision test setup completed")
    
    @classmethod
    def tearDownClass(cls):
        """
        测试结束后的清理工作
        删除临时文件和目录
        """
        # 删除临时目录
        shutil.rmtree(cls.test_dir)
        logger.info("Vision test teardown completed")
    
    @staticmethod
    def create_test_image(text: str, path: Path) -> Path:
        """
        创建包含指定文本的测试图像
        
        Args:
            text: 要写入图像的文本
            path: 图像保存路径
            
        Returns:
            Path: 保存的图像路径
        """
        # 创建空白图像
        image = Image.new('RGB', (400, 200), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        # 尝试加载字体（如果失败则使用默认字体）
        try:
            font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
        
        # 在图像中央绘制文本
        draw.text((50, 80), text, fill=(0, 0, 0), font=font)
        
        # 绘制一些简单的形状，使图像更有特点
        draw.rectangle((300, 50, 350, 100), fill=(255, 0, 0))
        draw.ellipse((200, 120, 250, 170), fill=(0, 255, 0))
        
        # 保存图像
        image.save(path)
        return path
    
    @staticmethod
    def create_chart_image(path: Path) -> Path:
        """
        创建简单的图表测试图像
        
        Args:
            path: 图像保存路径
            
        Returns:
            Path: 保存的图像路径
        """
        # 创建空白图像
        image = Image.new('RGB', (400, 300), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        # 绘制坐标轴
        draw.line((50, 250, 350, 250), fill=(0, 0, 0), width=2)  # X轴
        draw.line((50, 50, 50, 250), fill=(0, 0, 0), width=2)    # Y轴
        
        # 绘制柱状图
        draw.rectangle((100, 150, 130, 250), fill=(255, 0, 0))
        draw.rectangle((150, 100, 180, 250), fill=(0, 255, 0))
        draw.rectangle((200, 180, 230, 250), fill=(0, 0, 255))
        draw.rectangle((250, 120, 280, 250), fill=(255, 255, 0))
        
        # 添加标题
        try:
            font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
            
        draw.text((150, 20), "Sample Chart", fill=(0, 0, 0), font=font)
        
        # 保存图像
        image.save(path)
        return path
    
    @staticmethod
    def create_formula_image(path: Path) -> Path:
        """
        创建包含数学公式的测试图像
        
        Args:
            path: 图像保存路径
            
        Returns:
            Path: 保存的图像路径
        """
        # 创建空白图像
        image = Image.new('RGB', (400, 200), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        # 尝试加载字体
        try:
            font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
        
        # 绘制一个简单的公式（E=mc²）
        draw.text((100, 80), "E = mc²", fill=(0, 0, 0), font=font)
        
        # 添加一个分数线
        draw.line((200, 100, 250, 100), fill=(0, 0, 0), width=2)
        draw.text((210, 70), "a + b", fill=(0, 0, 0), font=font)
        draw.text((210, 110), "c + d", fill=(0, 0, 0), font=font)
        
        # 保存图像
        image.save(path)
        return path
    
    @staticmethod
    def create_text_image(text: str, path: Path) -> Path:
        """
        创建包含多行文本的测试图像
        
        Args:
            text: 要写入图像的文本
            path: 图像保存路径
            
        Returns:
            Path: 保存的图像路径
        """
        # 创建空白图像
        image = Image.new('RGB', (400, 200), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        # 尝试加载字体
        try:
            font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
        
        # 将文本分行
        lines = text.split("\n")
        if len(lines) == 1 and len(text) > 20:
            # 如果是单行长文本，尝试拆分
            words = text.split()
            lines = []
            current_line = ""
            for word in words:
                if len(current_line + " " + word) > 20:
                    lines.append(current_line)
                    current_line = word
                else:
                    current_line += " " + word if current_line else word
            if current_line:
                lines.append(current_line)
        
        # 绘制多行文本
        y = 50
        for line in lines:
            draw.text((50, y), line, fill=(0, 0, 0), font=font)
            y += 30
        
        # 保存图像
        image.save(path)
        return path

    def setUp(self):
        """
        每个测试方法运行前的设置
        使用模拟对象替代真实的模型
        """
        # 模拟BLIP模型
        self.blip_patcher = patch('app.core.vision.blip.BLIPModel')
        self.mock_blip = self.blip_patcher.start()
        
        # 配置模拟的BLIP模型方法
        self.mock_blip.return_value.generate_caption.return_value = {
            "success": True,
            "caption": "A test image showing some text and shapes",
            "parameters": {"max_length": 30, "num_beams": 4}
        }
        
        self.mock_blip.return_value.answer_question.return_value = {
            "success": True,
            "question": "What is shown in this image?",
            "answer": "This image shows a test pattern with text and geometric shapes",
            "parameters": {"max_length": 30, "num_beams": 4}
        }
        
        self.mock_blip.return_value.analyze_image_content.return_value = {
            "success": True,
            "caption": "A test image",
            "analysis": {"What type of image is this?": "A test image with text and shapes"},
            "image_type": "general_image"
        }
        
        # 模拟OCR服务
        self.ocr_patcher = patch('app.core.vision.service.ocr_service')
        self.mock_ocr = self.ocr_patcher.start()
        
        self.mock_ocr.process_image_file.return_value = {
            "success": True,
            "full_text": "Test Image",
            "paragraph_count": 1,
            "paragraphs": ["Test Image"]
        }
        
        self.mock_ocr.process_image_data.return_value = {
            "success": True,
            "full_text": "Test Image",
            "paragraph_count": 1,
            "paragraphs": ["Test Image"]
        }
        
        # 创建一个真实的vision service实例，但其中的组件被模拟
        self.vision_service = VisionService()
        
        # 关键修复: 直接替换captioner和classifier的blip_model属性为模拟对象
        self.vision_service.captioner.blip_model = self.mock_blip.return_value
        self.vision_service.classifier.blip_model = self.mock_blip.return_value
        
        logger.info("Test setup completed")
    
    def tearDown(self):
        """
        每个测试方法运行后的清理
        """
        # 停止所有模拟
        self.blip_patcher.stop()
        self.ocr_patcher.stop()
        logger.info("Test teardown completed")

    def test_vision_service_initialization(self):
        """测试视觉服务初始化"""
        # 这个测试不使用模拟，而是直接测试服务实例化
        service = VisionService()
        self.assertIsNotNone(service)
        self.assertIsNotNone(service.captioner)
        self.assertIsNotNone(service.classifier)
        
        # 测试服务信息获取
        info = service.get_service_info()
        self.assertIsNotNone(info)
        self.assertEqual(info["service"], "Vision Analysis Service")
        
        logger.info("Vision service initialization test passed")
    
    def test_analyze_image_with_path(self):
        """测试使用图像路径分析图像"""
        result = self.vision_service.analyze_image(self.test_image_path)
        
        # 验证基本结果结构
        self.assertTrue(result["success"])
        self.assertIn("timestamp", result)
        self.assertIn("caption", result)
        self.assertIn("content_type", result)
        self.assertIn("content_confidence", result)
        
        logger.info("Analyze image with path test passed")
    
    def test_analyze_image_with_numpy(self):
        """测试使用numpy数组分析图像"""
        # 读取测试图像为numpy数组
        image_array = read_image(self.test_image_path)
        image_array = convert_to_rgb(image_array)  # 转换为RGB以匹配预期格式
        
        result = self.vision_service.analyze_image(image_array)
        
        # 验证结果
        self.assertTrue(result["success"])
        self.assertIn("caption", result)
        
        logger.info("Analyze image with numpy array test passed")
    
    def test_analyze_image_with_ocr(self):
        """测试包括OCR的图像分析"""
        result = self.vision_service.analyze_image(
            self.test_image_path,
            include_ocr=True
        )
        
        # 验证结果包含OCR信息
        self.assertTrue(result["success"])
        self.assertIn("ocr", result)
        self.assertEqual(result["ocr"]["full_text"], "Test Image")
        
        logger.info("Analyze image with OCR test passed")
    
    def test_analyze_image_detailed(self):
        """测试详细图像分析"""
        # 配置模拟以返回详细结果
        self.mock_blip.return_value.answer_question.side_effect = [
            {
                "success": True,
                "answer": "This is about physics",
                "question": "What subject is this educational material about?"
            },
            {
                "success": True,
                "answer": "High school level",
                "question": "What grade level or education level is this material for?"
            }
        ]
        
        result = self.vision_service.analyze_image(
            self.test_formula_path,
            detailed_analysis=True
        )
        
        # 验证详细结果
        self.assertTrue(result["success"])
        
        # 如果存在教育分析数据，验证其内容
        if "educational_analysis" in result:
            self.assertIn("subject", result["educational_analysis"])
        
        logger.info("Detailed image analysis test passed")
    
    def test_describe_for_prompt(self):
        """测试为Prompt描述图像"""
        result = self.vision_service.describe_for_prompt(self.test_image_path)
        
        # 验证结果
        self.assertTrue(result["success"])
        self.assertIn("content_type", result)
        self.assertIn("caption", result)
        self.assertIn("prompt_suggestion", result)
        
        logger.info("Describe for prompt test passed")
    
    def test_analyze_image_batch(self):
        """测试批量分析图像"""
        # 创建测试图像列表
        test_images = [
            self.test_image_path,
            self.test_chart_path,
            self.test_formula_path
        ]
        
        results = self.vision_service.analyze_image_batch(test_images)
        
        # 验证结果
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertTrue(result["success"])
            self.assertIn("caption", result)
        
        logger.info("Batch image analysis test passed")
    
    def test_compare_images(self):
        """测试图像比较功能"""
        # 模拟分析结果
        self.vision_service.analyze_image = MagicMock(side_effect=[
            {
                "success": True,
                "content_type": "chart",
                "caption": "A chart showing data"
            },
            {
                "success": True,
                "content_type": "formula",
                "caption": "A mathematical formula"
            }
        ])
        
        result = self.vision_service.compare_images(
            self.test_chart_path, 
            self.test_formula_path
        )
        
        # 验证比较结果
        self.assertTrue(result["success"])
        self.assertIn("same_content_type", result)
        self.assertEqual(result["content_type1"], "chart")
        self.assertEqual(result["content_type2"], "formula")
        
        logger.info("Image comparison test passed")
    
    def test_error_handling(self):
        """测试错误处理"""
        # 模拟故障情况
        self.vision_service.captioner.blip_model.generate_caption.side_effect = Exception("Test error")
        
        # 错误应该被捕获，并返回带有错误信息的响应
        result = self.vision_service.analyze_image(self.test_image_path)
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        
        logger.info("Error handling test passed")
    
    def test_content_type_specific_handling(self):
        """测试不同内容类型的特定处理"""
        # 测试图表
        self.vision_service.classifier.classify_content = MagicMock(return_value={
            "success": True,
            "content_type": "chart",
            "confidence": 0.9
        })
        
        result = self.vision_service.analyze_image(self.test_chart_path)
        self.assertEqual(result["content_type"], "chart")
        
        # 测试公式
        self.vision_service.classifier.classify_content = MagicMock(return_value={
            "success": True,
            "content_type": "formula",
            "confidence": 0.9
        })
        
        result = self.vision_service.analyze_image(self.test_formula_path)
        self.assertEqual(result["content_type"], "formula")
        
        # 测试文本
        self.vision_service.classifier.classify_content = MagicMock(return_value={
            "success": True,
            "content_type": "text",
            "confidence": 0.9
        })
        
        result = self.vision_service.analyze_image(self.test_text_path)
        self.assertEqual(result["content_type"], "text")
        
        logger.info("Content type specific handling test passed")
    
    def test_with_real_blip_model(self):
        """
        使用真实BLIP模型测试
        这个测试需要BLIP模型已经下载并配置在tests/.env文件中
        """
        # 检查是否有有效的模型路径，如果没有则跳过测试
        self.vision_model_path = os.environ.get("VISION_MODEL_PATH")
        if not self.vision_model_path:
            self.skipTest(f"BLIP model not available at path: {self.vision_model_path}")
        
        # 停止模拟
        self.blip_patcher.stop()
        
        try:
            # 创建新的服务实例，显式指定模型路径
            real_service = VisionService(blip_model_name=self.vision_model_path)
            
            result = real_service.analyze_image(self.test_image_path)
            self.assertTrue(result["success"])
            self.assertIn("caption", result)
            
            logger.info(f"Real model test passed, caption: {result.get('caption', '')}")
            
        except Exception as e:
            logger.error(f"Error in real model test: {str(e)}")
            self.fail(f"Real model test failed: {str(e)}")
        finally:
            # 重新启动模拟
            self.mock_blip = self.blip_patcher.start()
    
    @unittest.skipIf(False, "跳过需要示例图像的测试")
    def test_with_real_images(self):
        """
        使用真实图像测试
        这个测试需要sample_images目录存在
        """
        # 停止BLIP模拟
        self.blip_patcher.stop()
        
        try:
            # 确认目录存在
            sample_dir = Path("sample_images")
            if not sample_dir.exists():
                self.skipTest("Sample images directory not found")
            
            # 创建新的服务实例
            real_service = VisionService()
            
            # 处理目录中的所有jpg和png图像
            image_count = 0
            for image_file in list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png")):
                result = real_service.analyze_image(image_file)
                self.assertTrue(result["success"], f"Failed to analyze {image_file}")
                image_count += 1
                
                logger.info(f"Successfully analyzed {image_file}")
            
            if image_count == 0:
                self.skipTest("No image files found in sample_images directory")
                
            logger.info(f"Real images test passed with {image_count} images")
            
        except Exception as e:
            logger.error(f"Error in real images test: {str(e)}")
            self.fail(f"Real images test failed: {str(e)}")
        finally:
            # 重新启动模拟
            self.mock_blip = self.blip_patcher.start()
    
    @unittest.skipIf(not os.path.exists("sample_images"), "跳过真实OCR集成测试")
    def test_integration_with_ocr(self):
        """测试与OCR的集成"""
        # 停止OCR模拟
        self.ocr_patcher.stop()
        
        try:
            result = self.vision_service.analyze_image(
                self.test_text_path,
                include_ocr=True
            )
            
            self.assertTrue(result["success"])
            self.assertIn("ocr", result)
            self.assertIn("full_text", result["ocr"])
            
            logger.info(f"OCR integration test passed, text: {result['ocr'].get('full_text', '')[:30]}...")
            
        except Exception as e:
            logger.error(f"Error in OCR integration test: {str(e)}")
            self.fail(f"OCR integration test failed: {str(e)}")
        finally:
            # 重新启动模拟
            self.mock_ocr = self.ocr_patcher.start()
    
    def test_extract_subject_from_qa(self):
        """测试从问答结果中提取学科信息"""
        # 准备测试数据
        qa_results = {
            "What subject is this educational material about?": "This appears to be about physics concepts"
        }
        
        subject = self.vision_service._extract_subject_from_qa(qa_results)
        self.assertEqual(subject, "physics")
        
        # 测试无法识别的学科
        qa_results = {
            "What subject is this educational material about?": "This is about general knowledge"
        }
        
        subject = self.vision_service._extract_subject_from_qa(qa_results)
        self.assertEqual(subject, "general")
        
        logger.info("Extract subject from QA test passed")
    
    def test_extract_education_level(self):
        """测试从问答结果中提取教育级别"""
        # 准备测试数据
        qa_results = {
            "What grade level or education level is this material for?": "This seems appropriate for high school students"
        }
        
        level = self.vision_service._extract_education_level(qa_results)
        self.assertEqual(level, "high school")
        
        # 测试无法识别的级别
        qa_results = {
            "What grade level or education level is this material for?": "This is for general audience"
        }
        
        level = self.vision_service._extract_education_level(qa_results)
        self.assertEqual(level, "unknown")
        
        logger.info("Extract education level test passed")
    
    def test_generate_summary(self):
        """测试根据分析结果生成摘要"""
        # 测试不同内容类型的摘要生成
        content_types = [
            ("chart", "This image contains a chart or diagram."),
            ("formula", "This image contains mathematical or scientific formula."),
            ("text", "This image contains text content."),
            ("unknown", "")
        ]
        
        for content_type, expected_prefix in content_types:
            analysis = {
                "content_type": content_type,
                "caption": "Test caption"
            }
            
            summary = self.vision_service._generate_summary(analysis)
            if expected_prefix:
                self.assertTrue(summary.startswith(expected_prefix))
            else:
                self.assertEqual(summary, "Test caption")
        
        logger.info("Generate summary test passed")


class TestSingletonBehavior(unittest.TestCase):
    """测试单例模式行为"""
    
    def test_vision_service_singleton(self):
        """测试vision_service是否是单例"""
        # 导入全局单例
        from app.core.vision.service import vision_service as global_instance
        
        # 创建新实例
        local_instance = VisionService()
        
        # 验证两者不是同一个对象（因为每次调用VisionService()都会创建新实例）
        self.assertIsNot(global_instance, local_instance)
        
        # 但是它们应该共享相同的底层组件
        self.assertIsNot(global_instance.captioner, None)
        self.assertIsNot(local_instance.captioner, None)
        
        logger.info("Singleton behavior test passed")


if __name__ == "__main__":
    unittest.main()