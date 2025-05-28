import os
import base64
import requests
from typing import Dict, Any, Union, Optional
from pathlib import Path
import numpy as np
from PIL import Image
import io

from app.core.models.interface import MultiModalModelInterface, ModelResponse
from app.utils.logger import logger
from app.config.settings import settings

class QwenVLModel(MultiModalModelInterface):
    """
    通义千问视觉语言模型(Qwen-VL)实现
    通过DashScope API调用通义千问VL模型进行图文理解和生成
    """
    
    # DashScope API端点
    API_ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
    
    # 支持的模型列表
    SUPPORTED_MODELS = [
        "qwen-vl-plus",        # 功能最强大的商用版本
        "qwen-vl-max",         # 大规模版本，性能更强
        "qwen-vl-chat",        # 对话版本
        "qwen-vl-plus-7b",     # 轻量级版本
        "qwen-vl-plus-32k",    # 扩展上下文长度版本
    ]
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model_name: str = "qwen-vl-plus",
        temperature: float = 0.7,
        max_tokens: int = 1500,
    ):
        """
        初始化Qwen-VL模型
        
        Args:
            api_key: DashScope API密钥，如果为None则从环境变量或设置中获取
            model_name: 模型名称，默认为"qwen-vl-plus"
            temperature: 采样温度，控制生成文本的随机性，默认为0.7
            max_tokens: 生成的最大token数，默认为1500
        """
        # 获取API密钥
        self.api_key = api_key or settings.DASHSCOPE_API_KEY or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            logger.error("DashScope API key not provided. Please set DASHSCOPE_API_KEY environment variable or in settings.")
            raise ValueError("DashScope API key is required")
        
        # 检查模型名称是否有效
        if model_name not in self.SUPPORTED_MODELS:
            logger.warning(f"Model {model_name} is not in the list of officially supported models. This might cause issues.")
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._initialized = True
        
        logger.info(f"Initialized QwenVLModel with model: {model_name}")
    
    def initialize(self) -> bool:
        """
        初始化模型
        
        Returns:
            bool: 是否成功初始化
        """
        # 模型已在__init__中初始化，这里主要验证API密钥是否有效
        if not self._initialized or not self.api_key:
            return False
        
        # TODO: 可以添加API密钥验证逻辑，在实现完HTTP客户端模块后再完善
        return True
    
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
            logger.error("Model not initialized. Call initialize() first.")
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
            
            # 构造请求体
            request_data = self._build_request_data(prompt, image, temperature, max_tokens, opts)
            
            # 设置HTTP请求头
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # 发送请求
            logger.info(f"Sending request to DashScope API for model: {self.model_name}")
            response = requests.post(
                self.API_ENDPOINT,
                headers=headers,
                json=request_data
            )
            
            # 检查响应状态
            if response.status_code != 200:
                error_msg = f"API request failed with status code {response.status_code}: {response.text}"
                logger.error(error_msg)
                return ModelResponse(
                    text="", 
                    success=False,
                    error=error_msg
                ).to_dict()
            
            # 解析响应
            response_json = response.json()
            
            # 检查是否有错误码
            if response_json.get("code", ""):
                error_msg = f"API returned error: {response_json.get('code')}: {response_json.get('message', '')}"
                logger.error(error_msg)
                return ModelResponse(
                    text="", 
                    success=False,
                    error=error_msg
                ).to_dict()
                
            # 提取生成的文本
            generated_text = self._parse_response(response_json)
            
            # 提取使用的token信息
            usage = response_json.get("usage", {})
            
            # 构造响应
            return ModelResponse(
                text=generated_text,
                success=True,
                model_name=self.model_name,
                usage={
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0)
                },
                metadata={
                    "request_id": response_json.get("request_id", ""),
                    "finish_reason": self._get_finish_reason(response_json)
                }
            ).to_dict()
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return ModelResponse(
                text="", 
                success=False,
                error=str(e)
            ).to_dict()
    
    def _build_request_data(
        self, 
        prompt: str, 
        image: Optional[Union[str, Path, np.ndarray]],
        temperature: float,
        max_tokens: int,
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        构建API请求数据
        
        Args:
            prompt: 文本提示词
            image: 图像输入
            temperature: 温度参数
            max_tokens: 最大标记数
            options: 其他选项
            
        Returns:
            Dict[str, Any]: API请求数据
        """
        # 准备消息列表
        messages = []
        
        # 添加系统消息（如果提供）
        if "system_prompt" in options:
            messages.append({
                "role": "system", 
                "content": options["system_prompt"]
            })
        
        # 添加用户消息（包含文本和可选的图像）
        if image is not None:
            # 处理并获取图像数据URI
            image_data_uri = self._prepare_image(image)
            
            if image_data_uri:
                # 当有图像时，content必须是一个列表
                content = [
                    {"text": prompt},  # 添加文本内容
                    {"image": image_data_uri}  # 添加图像内容
                ]
                
                # 将组合内容添加到用户消息
                messages.append({
                    "role": "user",
                    "content": content
                })
            else:
                # 图像处理失败，仅使用文本
                logger.warning("Image processing failed, using text only prompt")
                messages.append({
                    "role": "user",
                    "content": prompt
                })
        else:
            # 只有文本的情况
            messages.append({
                "role": "user",
                "content": prompt
            })
        
        # 构建请求参数
        parameters = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "result_format": "message"  # 使用message格式便于解析
        }
        
        # 添加其他可能的参数
        for param in ["top_p", "top_k", "repetition_penalty", "seed"]:
            if param in options:
                parameters[param] = options[param]
                
        # 处理高分辨率图像选项
        if options.get("high_resolution", False):
            parameters["vl_high_resolution_images"] = True
        
        # 构造完整请求
        request_data = {
            "model": self.model_name,
            "input": {
                "messages": messages
            },
            "parameters": parameters
        }
        
        return request_data
    
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
                # 确保图像是RGB格式
                if len(image.shape) == 3 and image.shape[2] == 3:
                    pil_image = Image.fromarray(image.astype('uint8'))
                else:
                    # 转换为RGB格式
                    if len(image.shape) == 2:  # 灰度图像
                        pil_image = Image.fromarray(image.astype('uint8'), mode='L').convert('RGB')
                    else:
                        raise ValueError(f"Unsupported image format with shape: {image.shape}")
                
                buffer = io.BytesIO()
                pil_image.save(buffer, format="PNG")
                image_bytes = buffer.getvalue()
                
            else:
                logger.error(f"Unsupported image type: {type(image)}")
                return ""
                
            # 转换为Base64编码，DashScope需要的是纯Base64字符串，不带前缀
            base64_data = base64.b64encode(image_bytes).decode("utf-8")
            return f"data:image/png;base64,{base64_data}"  # 添加data URI前缀
        
        except Exception as e:
            logger.error(f"Error preparing image: {str(e)}")
            return ""
    
    def _parse_response(self, response_json: Dict[str, Any]) -> str:
        """
        解析API响应并提取生成的文本
        
        Args:
            response_json: API响应JSON
            
        Returns:
            str: 生成的文本
        """
        try:
            # 从response_format=message的响应中提取文本
            output = response_json.get("output", {})
            choices = output.get("choices", [])
            
            if not choices:
                return ""
                
            message = choices[0].get("message", {})
            content = message.get("content", "")
            
            # 如果内容是列表（包含文本和图像），只提取文本部分
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        text_parts.append(item["text"])
                return "".join(text_parts)
                
            return content
            
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            return ""
    
    def _get_finish_reason(self, response_json: Dict[str, Any]) -> str:
        """
        获取生成结束的原因
        
        Args:
            response_json: API响应JSON
            
        Returns:
            str: 生成结束的原因
        """
        try:
            output = response_json.get("output", {})
            choices = output.get("choices", [])
            
            if choices:
                return choices[0].get("finish_reason", "")
                
            return ""
            
        except Exception:
            return ""
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 包含模型名称、参数量等信息的字典
        """
        return {
            "model_name": self.model_name,
            "model_type": "Qwen-VL",
            "provider": "Alibaba DashScope",
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