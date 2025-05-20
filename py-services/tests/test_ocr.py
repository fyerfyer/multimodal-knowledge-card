import os
import sys
import unittest
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import shutil

# 确保能够导入应用模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.ocr.interface import OCRResult
from app.core.ocr.paddle_ocr import PaddleOCREngine
from app.core.ocr.factory import OCRFactory
from app.core.ocr.postprocess import OCRPostProcessor
from app.core.ocr.service import OCRService
from app.config.settings import settings
from app.utils.logger import logger


class TestOCRModule(unittest.TestCase):
    """OCR模块的综合测试类"""
    
    @classmethod
    def setUpClass(cls):
        """
        测试开始前的准备工作
        创建测试用的图像和目录
        """
        # 创建临时目录用于保存测试图像
        cls.test_dir = Path(tempfile.mkdtemp())
        
        # 创建测试图像
        cls.test_image_path = cls.create_test_image("Hello World", cls.test_dir / "test_image.png")
        cls.test_chinese_image_path = cls.create_test_image("你好世界", cls.test_dir / "test_chinese.png")
        cls.empty_image_path = cls.create_empty_image(cls.test_dir / "empty_image.png")
        
        # 初始化OCR服务
        cls.ocr_service = OCRService(engine_type="paddleocr")
        
        logger.info("Test setup completed")
    
    @classmethod
    def tearDownClass(cls):
        """
        测试结束后的清理工作
        删除临时文件和目录
        """
        # 删除临时目录
        shutil.rmtree(cls.test_dir)
        logger.info("Test teardown completed")
    
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
            # 对于中文文本，尝试使用支持中文的字体
            if any('\u4e00' <= char <= '\u9fff' for char in text):
                # 这里需要根据系统可用字体调整
                font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # Linux路径
                if not os.path.exists(font_path):
                    font_path = "C:\\Windows\\Fonts\\simhei.ttf"  # Windows路径
                
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, 36)
                else:
                    font = ImageFont.load_default()
            else:
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
        
        # 在图像中央绘制文本
        draw.text((50, 80), text, fill=(0, 0, 0), font=font)
        
        # 保存图像
        image.save(path)
        return path
    
    @staticmethod
    def create_empty_image(path: Path, width: int = 200, height: int = 100) -> Path:
        """
        创建空白图像
        
        Args:
            path: 图像保存路径
            width: 图像宽度
            height: 图像高度
            
        Returns:
            Path: 保存的图像路径
        """
        # 创建空白图像
        image = Image.new('RGB', (width, height), color=(255, 255, 255))
        image.save(path)
        return path

    def test_paddle_ocr_engine_initialization(self):
        """测试PaddleOCR引擎初始化"""
        try:
            engine = PaddleOCREngine(lang="en")
            self.assertIsNotNone(engine.ocr_engine, "OCR engine should be initialized")
            
            # 测试引擎信息
            info = engine.get_engine_info()
            self.assertEqual(info["name"], "PaddleOCR")
            self.assertEqual(info["language"], "en")
            
            logger.info("PaddleOCR engine initialization test passed")
        except Exception as e:
            self.fail(f"PaddleOCR engine initialization failed: {str(e)}")
    
    def test_ocr_factory(self):
        """测试OCR工厂类"""
        # 测试创建默认引擎
        default_engine = OCRFactory.create_engine()
        self.assertIsInstance(default_engine, PaddleOCREngine)
        
        # 测试创建指定类型引擎
        paddle_engine = OCRFactory.create_engine("paddleocr")
        self.assertIsInstance(paddle_engine, PaddleOCREngine)
        
        # 测试列出可用引擎
        engines = OCRFactory.list_available_engines()
        self.assertIn("paddleocr", engines)
        
        # 测试不支持的引擎类型
        with self.assertRaises(ValueError):
            OCRFactory.create_engine("unsupported_engine")
        
        logger.info("OCR factory test passed")
    
    def test_process_image_file(self):
        """测试处理图像文件"""
        # 测试处理英文图像
        result_en = self.ocr_service.process_image_file(self.test_image_path)
        self.assertTrue(result_en["success"])
        self.assertIn("text_count", result_en)
        
        # TODO: 实际结果取决于OCR准确性，可能需要调整断言
        # 我们只能大致检查是否包含了一些输出
        if result_en["text_count"] > 0:
            self.assertTrue(len(result_en["full_text"]) > 0)
        
        # 测试处理中文图像
        result_zh = self.ocr_service.process_image_file(self.test_chinese_image_path)
        self.assertTrue(result_zh["success"])
        
        # 测试处理空白图像
        result_empty = self.ocr_service.process_image_file(self.empty_image_path)
        self.assertTrue(result_empty["success"])
        
        # 测试处理不存在的图像
        result_not_exist = self.ocr_service.process_image_file("not_exist.png")
        self.assertFalse(result_not_exist["success"])
        self.assertIn("error", result_not_exist)
        
        logger.info("Process image file test passed")
    
    def test_process_image_data(self):
        """测试处理图像数据（numpy数组）"""
        # 从文件加载图像并转换为numpy数组
        image = Image.open(self.test_image_path)
        image_np = np.array(image)
        
        # 处理图像数据
        result = self.ocr_service.process_image_data(image_np)
        self.assertTrue(result["success"])
        
        # 测试处理空数组
        # 确保空数组会被捕获并返回失败结果，而不需要抛出异常
        result_empty = self.ocr_service.process_image_data(np.array([]))
        self.assertFalse(result_empty["success"])
        self.assertIn("error", result_empty)
        
        logger.info("Process image data test passed")
    
    def test_save_image_for_ocr(self):
        """测试保存上传的图像"""
        # 读取测试图像
        with open(self.test_image_path, "rb") as f:
            image_bytes = f.read()
        
        # 保存图像并获取路径
        saved_path = self.ocr_service.save_image_for_ocr(image_bytes, "test_upload.png")
        
        # 验证文件是否已保存
        self.assertTrue(os.path.exists(saved_path))
        
        # 验证文件内容
        with open(saved_path, "rb") as f:
            saved_bytes = f.read()
        
        self.assertEqual(saved_bytes, image_bytes)
        
        # 测试没有文件名的情况
        no_name_path = self.ocr_service.save_image_for_ocr(image_bytes)
        self.assertTrue(os.path.exists(no_name_path))
        
        # 清理
        if os.path.exists(saved_path):
            os.remove(saved_path)
        if os.path.exists(no_name_path):
            os.remove(no_name_path)
        
        logger.info("Save image test passed")
    
    def test_ocr_postprocessor(self):
        """测试OCR结果后处理器"""
        # 创建一些模拟的OCR结果
        results = [
            OCRResult("Hello", 0.9, [[10, 10], [100, 10], [100, 30], [10, 30]]),
            OCRResult("World", 0.85, [[120, 10], [200, 10], [200, 30], [120, 30]]),
            OCRResult("Testing", 0.8, [[10, 50], [100, 50], [100, 70], [10, 70]]),
        ]
        
        # 初始化后处理器
        processor = OCRPostProcessor()
        
        # 测试排序
        sorted_results = processor.sort_by_position(results)
        self.assertEqual(len(sorted_results), 3)
        self.assertEqual(sorted_results[0].text, "Hello")
        
        # 测试分组
        lines = processor.group_by_lines(results)
        self.assertEqual(len(lines), 2)  # 应该有两行
        
        # 测试合并为段落
        paragraphs = processor.merge_into_paragraphs(lines)
        self.assertEqual(len(paragraphs), 2)  # 应该有两个段落
        
        # 测试完整处理流程
        paragraphs, full_text = processor.process(results)
        self.assertEqual(len(paragraphs), 2)
        self.assertTrue(len(full_text) > 0)
        
        # 测试空结果
        empty_paragraphs, empty_text = processor.process([])
        self.assertEqual(len(empty_paragraphs), 0)
        self.assertEqual(empty_text, "")
        
        logger.info("OCR postprocessor test passed")
    
    def test_change_engine(self):
        """测试更改OCR引擎"""
        # 保存原始引擎信息
        original_engine_info = self.ocr_service.engine.get_engine_info()
        
        # 尝试更改为相同类型的引擎
        success = self.ocr_service.change_engine("paddleocr")
        self.assertTrue(success)
        
        # 验证引擎是否已更改
        new_engine_info = self.ocr_service.engine.get_engine_info()
        self.assertEqual(new_engine_info["name"], original_engine_info["name"])
        
        # 尝试更改为不支持的引擎类型
        success = self.ocr_service.change_engine("unsupported_engine")
        self.assertFalse(success)
        
        logger.info("Change engine test passed")
    
    @unittest.skipIf(not os.path.exists("sample_images"), "跳过需要示例图像的测试")
    def test_with_real_images(self):
        """
        使用真实图像测试（如果有示例图像目录）
        这个测试需要手动准备一些真实的图像样本
        """
        sample_dir = Path("sample_images")
        if not sample_dir.exists():
            self.skipTest("No sample images directory")
        
        for image_file in sample_dir.glob("*.jpg"):
            result = self.ocr_service.process_image_file(image_file)
            self.assertTrue(result["success"], f"Failed to process {image_file}")
            
            # 这里不验证具体的文本内容，只确保处理成功
            logger.info(f"Successfully processed {image_file}: found {result['text_count']} text regions")
    
    def test_async_process_image_bytes(self):
        """测试异步处理图像字节数据"""
        # 读取测试图像
        with open(self.test_image_path, "rb") as f:
            image_bytes = f.read()
        
        # 异步测试需要在运行循环中执行
        import asyncio
        
        async def test_async():
            # 处理图像字节
            result = await self.ocr_service.process_image_bytes(image_bytes, "test_async.png")
            self.assertTrue(result["success"])
            
            # 测试无文件名
            result_no_name = await self.ocr_service.process_image_bytes(image_bytes)
            self.assertTrue(result_no_name["success"])
        
        # 运行异步测试
        try:
            asyncio.run(test_async())
            logger.info("Async process image bytes test passed")
        except (NotImplementedError, ImportError) as e:
            # 某些环境可能不支持asyncio.run
            self.skipTest(f"Async test skipped: {str(e)}")


if __name__ == "__main__":
    unittest.main()