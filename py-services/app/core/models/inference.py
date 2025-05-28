from typing import Dict, Any, Union, Optional, List
from pathlib import Path
import time
import numpy as np
from enum import Enum

from app.core.models.interface import MultiModalModelInterface, ModelResponse
from app.core.models.qwen_vl import QwenVLModel
from app.utils.logger import logger
from app.config.settings import settings

class ModelType(Enum):
    """多模态模型类型枚举"""
    QWEN_VL = "qwen-vl"


class InferenceService:
    """
    多模态模型推理服务
    负责加载模型、处理输入并生成多模态推理结果
    """
    
    def __init__(self):
        """
        初始化推理服务
        """
        # 模型实例字典，按需加载
        self.models: Dict[str, MultiModalModelInterface] = {}
        
        # 默认使用的模型
        self.default_model_type = settings.LLM_MODEL or ModelType.QWEN_VL.value
        self.current_model_type = self.default_model_type
        
        logger.info(f"Inference service initialized with default model: {self.default_model_type}")
        
    def get_model(self, model_type: Optional[str] = None) -> MultiModalModelInterface:
        """
        获取指定类型的模型实例，如果不存在则创建
        
        Args:
            model_type: 模型类型，如果为None则使用当前默认模型
            
        Returns:
            MultiModalModelInterface: 模型实例
            
        Raises:
            ValueError: 如果指定的模型类型不受支持
        """
        # 使用指定的模型类型或默认模型类型
        model_type = model_type or self.current_model_type
        model_type = model_type.lower()
        
        # 如果模型已经加载，直接返回
        if model_type in self.models:
            logger.debug(f"Using existing {model_type} model instance")
            return self.models[model_type]
        
        # 按类型创建新的模型实例
        try:
            logger.info(f"Creating new model instance for: {model_type}")
            
            if model_type == ModelType.QWEN_VL.value or model_type.startswith("qwen"):
                model = QwenVLModel(
                    api_key=settings.DASHSCOPE_API_KEY,
                    model_name=settings.LLM_MODEL_PATH or "qwen-vl-plus"
                )
            else:
                logger.error(f"Unsupported model type: {model_type}")
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # 确保模型初始化
            if not model.is_initialized():
                success = model.initialize()
                if not success:
                    logger.error(f"Failed to initialize {model_type} model")
                    raise RuntimeError(f"Failed to initialize {model_type} model")
            
            # 将模型实例保存到字典中
            self.models[model_type] = model
            return model
            
        except Exception as e:
            logger.error(f"Error creating model instance for {model_type}: {str(e)}")
            raise
    
    def switch_model(self, model_type: str) -> bool:
        """
        切换当前使用的模型
        
        Args:
            model_type: 要切换到的模型类型
            
        Returns:
            bool: 切换是否成功
        """
        try:
            # 确保模型类型有效
            model_type = model_type.lower()
            
            # 检查是否支持该模型类型
            if not (model_type == ModelType.QWEN_VL.value or model_type.startswith("qwen")):
                logger.error(f"Unsupported model type for switching: {model_type}")
                return False
            
            # 尝试加载模型
            model = self.get_model(model_type)
            if not model.is_initialized():
                model.initialize()
            
            # 成功加载后设置为当前模型
            self.current_model_type = model_type
            logger.info(f"Switched to model: {model_type}")
            return True
            
        except Exception as e:
            logger.error(f"Error switching model to {model_type}: {str(e)}")
            return False
    
    def generate_response(
        self, 
        prompt: str,
        image: Optional[Union[str, Path, np.ndarray]] = None,
        model_type: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        生成模型响应
        
        Args:
            prompt: 文本提示词
            image: 可选的图像输入，可以是路径或numpy数组
            model_type: 可选的模型类型，如果不指定则使用当前默认模型
            options: 可选的生成参数
            
        Returns:
            Dict[str, Any]: 生成的响应
        """
        start_time = time.time()
        
        try:
            # 获取指定类型的模型实例或默认模型
            model = self.get_model(model_type)
            
            # 记录推理开始
            logger.info(f"Running inference with {model.get_model_info()['model_name']}")
            if image is not None:
                image_info = str(image) if isinstance(image, (str, Path)) else f"image array shape: {image.shape}"
                logger.info(f"Input: prompt with image ({image_info})")
            else:
                logger.info(f"Input: text-only prompt")
            
            # 生成响应
            response = model.generate_response(prompt, image, options)
            
            # 记录推理完成
            processing_time = time.time() - start_time
            logger.info(f"Inference completed in {processing_time:.2f} seconds")
            
            # 添加处理时间到响应
            response["processing_time"] = processing_time
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error generating response: {str(e)}")
            
            # 构造错误响应
            return {
                "success": False,
                "error": str(e),
                "processing_time": processing_time
            }
    
    async def generate_response_async(
        self, 
        prompt: str,
        image: Optional[Union[str, Path, np.ndarray]] = None,
        model_type: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        异步生成模型响应（目前仍是同步实现）
        
        Args:
            prompt: 文本提示词
            image: 可选的图像输入，可以是路径或numpy数组
            model_type: 可选的模型类型，如果不指定则使用当前默认模型
            options: 可选的生成参数
            
        Returns:
            Dict[str, Any]: 生成的响应
        """
        # TODO: 实现真正的异步处理，与非异步版本分离
        # 目前直接调用同步版本
        return self.generate_response(prompt, image, model_type, options)
    
    def generate_batch_responses(
        self, 
        requests: List[Dict[str, Any]],
        model_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        批量生成模型响应
        
        Args:
            requests: 请求列表，每个请求是一个包含prompt、image和options的字典
            model_type: 可选的模型类型，如果不指定则使用当前默认模型
            
        Returns:
            List[Dict[str, Any]]: 响应列表
        """
        results = []
        
        # 获取模型实例，避免每次请求都重新加载
        model = self.get_model(model_type)
        
        for idx, request in enumerate(requests):
            try:
                logger.info(f"Processing batch request {idx+1}/{len(requests)}")
                
                prompt = request.get("prompt", "")
                image = request.get("image")
                options = request.get("options")
                
                # 对单个请求执行推理
                response = model.generate_response(prompt, image, options)
                results.append(response)
                
            except Exception as e:
                logger.error(f"Error processing batch request {idx+1}: {str(e)}")
                results.append({
                    "success": False,
                    "error": str(e),
                    "request_index": idx
                })
        
        return results
    
    def get_model_info(self, model_type: Optional[str] = None) -> Dict[str, Any]:
        """
        获取模型信息
        
        Args:
            model_type: 要获取信息的模型类型，如果为None则使用当前默认模型
            
        Returns:
            Dict[str, Any]: 模型信息
        """
        try:
            model = self.get_model(model_type)
            return model.get_model_info()
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {"error": str(e)}
    
    def list_available_models(self) -> Dict[str, Dict[str, Any]]:
        """
        列出所有可用的模型
        
        Returns:
            Dict[str, Dict[str, Any]]: 模型类型及其信息
        """
        models_info = {}
        
        # 列出已加载的模型
        for model_type, model in self.models.items():
            try:
                models_info[model_type] = model.get_model_info()
                models_info[model_type]["loaded"] = True
            except Exception as e:
                logger.error(f"Error getting info for model {model_type}: {str(e)}")
                models_info[model_type] = {
                    "error": str(e),
                    "loaded": True  # 虽然获取信息出错，但模型实例存在
                }
        
        # 添加未加载但支持的模型
        for model_type in [ModelType.QWEN_VL.value]:  # 只包含 QWEN_VL
            if model_type not in models_info:
                models_info[model_type] = {
                    "model_name": model_type,
                    "loaded": False
                }
        
        return models_info


# 创建单例实例，方便直接导入使用
inference_service = InferenceService()