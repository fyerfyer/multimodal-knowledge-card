import os
import base64
import requests
from typing import Dict, Any, Union, Optional
from pathlib import Path
import numpy as np
from PIL import Image
import io
from openai import OpenAI

from app.core.models.interface import MultiModalModelInterface, ModelResponse
from app.utils.logger import logger
from app.config.settings import settings

class DeepSeekVLModel(MultiModalModelInterface):
    """
    DeepSeek视觉语言模型(DeepSeek-VL)实现
    通过DeepSeek API调用视觉语言模型进行图文理解和生成
    """
    
    # DeepSeek API端点
    API_BASE_URL = "https://api.deepseek.com"
    
    # 支持的模型列表
    SUPPORTED_MODELS = [
        "deepseek-vl",         # 视觉语言模型
        "deepseek-chat",       # 对话模型
    ]
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model_name: str = "deepseek-vl",
        temperature: float = 0.7,
        max_tokens: int = 1500,
    ):
        """
        初始化DeepSeek-VL模型
        
        Args:
            api_key: DeepSeek API密钥，如果为None则从环境变量或设置中获取
            model_name: 模型名称，默认为"deepseek-vl"
            temperature: 采样温度，控制生成文本的随机性，默认为0.7
            max_tokens: 生成的最大token数，默认为1500
        """
        # 获取API密钥
        self.api_key = api_key or settings.DEEPSEEK_API_KEY or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            logger.error("DeepSeek API key not provided. Please set DEEPSEEK_API_KEY environment variable or in settings.")
            raise ValueError("DeepSeek API key is required")
        
        # 检查模型名称是否有效
        if model_name not in self.SUPPORTED_MODELS:
            logger.warning(f"Model {model_name} is not in the list of officially supported models. This might cause issues.")
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._initialized = True
        self.client = None
        
        logger.info(f"Initialized DeepSeekVLModel with model: {model_name}")
    
    def initialize(self) -> bool:
        """
        初始化模型
        
        Returns:
            bool: 是否成功初始化
        """
        if self.client is None:
            try:
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.API_BASE_URL
                )
                logger.info("DeepSeek API client initialized successfully")
                self._initialized = True
                return True
            except Exception as e:
                logger.error(f"Failed to initialize DeepSeek API client: {str(e)}")
                self._initialized = False
                return False
        
        return self._initialized
    
    def generate_response(
        self, 
        prompt: str,
        image: Optional[Union[str, Path, np.ndarray]] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        生成模型响应
        
        Args:
            prompt: 文本提示词
            image: 可选的图像输入，可以是图像路径字符串、Path对象或numpy数组
            options: 可选的生成参数
            
        Returns:
            Dict[str, Any]: 包含生成结果的字典
        """
        if not self._initialized:
            # 尝试初始化
            if not self.initialize():
                logger.error("Model not initialized. Failed to initialize DeepSeek API client.")
                return ModelResponse(
                    text="", 
                    success=False,
                    error="Model not initialized"
                ).to_dict()
        
        try:
            # 处理选项
            opts = options or {}
            temperature = opts.get("temperature", self.temperature)
            max_tokens = opts.get("max_tokens", self.max_tokens)
            
            # 准备消息列表
            messages = []
            
            # 添加系统消息（如果提供）
            if "system_prompt" in opts:
                messages.append({
                    "role": "system", 
                    "content": opts["system_prompt"]
                })
            
            # 准备用户消息
            if image is not None:
                # 如果有图像，需要准备多模态内容
                user_content = []
                
                # 添加文本内容
                if prompt:
                    user_content.append({
                        "type": "text", 
                        "text": prompt
                    })
                
                # 处理图像并添加图像内容
                image_data = self._prepare_image(image)
                if image_data:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    })
                
                messages.append({
                    "role": "user",
                    "content": user_content
                })
            else:
                # 仅文本的消息
                messages.append({
                    "role": "user",
                    "content": prompt
                })
            
            logger.info(f"Sending request to DeepSeek API for model: {self.model_name}")
            
            # 调用API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False
            )
            
            # 提取生成的文本
            generated_text = response.choices[0].message.content if response.choices else ""
            
            # 提取使用的token信息
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            # 构造响应
            return ModelResponse(
                text=generated_text,
                success=True,
                model_name=self.model_name,
                usage=usage,
                metadata={
                    "finish_reason": response.choices[0].finish_reason if response.choices else None
                }
            ).to_dict()
            
        except Exception as e:
            logger.error(f"Error generating response from DeepSeek API: {str(e)}")
            return ModelResponse(
                text="", 
                success=False,
                error=str(e)
            ).to_dict()
    
    def _prepare_image(self, image: Union[str, Path, np.ndarray]) -> str:
        """
        将图像转换为Base64编码
        
        Args:
            image: 图像输入，可以是路径或numpy数组
            
        Returns:
            str: Base64编码的图像数据
        """
        try:
            # 处理不同类型的图像输入
            if isinstance(image, (str, Path)):
                # 如果是文件路径，直接读取
                with open(image, "rb") as f:
                    image_bytes = f.read()
                    
            elif isinstance(image, np.ndarray):
                # 如果是numpy数组，转换为PIL图像然后转为字节
                pil_image = Image.fromarray(image.astype('uint8'))
                buffer = io.BytesIO()
                pil_image.save(buffer, format="JPEG")
                image_bytes = buffer.getvalue()
                
            else:
                logger.error(f"Unsupported image type: {type(image)}")
                return ""
                
            # 转换为Base64编码
            base64_data = base64.b64encode(image_bytes).decode("utf-8")
            return base64_data
            
        except Exception as e:
            logger.error(f"Error preparing image: {str(e)}")
            return ""
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 包含模型名称、参数量等信息的字典
        """
        return {
            "model_name": self.model_name,
            "model_type": "DeepSeek-VL",
            "provider": "DeepSeek AI",
            "supports_vision": True,
            "supports_audio": False,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
    
    def is_initialized(self) -> bool:
        """
        检查模型是否已初始化
        
        Returns:
            bool: 模型是否已初始化
        """
        return self._initialized