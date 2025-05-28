import os
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
import threading
import json

from app.core.models.interface import MultiModalModelInterface
from app.core.models.factory import create_model
from app.config.settings import settings
from app.utils.logger import logger

class ModelLoader:
    """
    模型加载器
    负责模型资源的加载、缓存和管理
    支持多种模型类型，管理模型生命周期
    """
    
    def __init__(self, 
                 cache_dir: Optional[Path] = None,
                 max_cache_models: int = 3,
                 cache_ttl: int = 3600,  # 1小时
                 auto_cleanup: bool = True):
        """
        初始化模型加载器
        
        Args:
            cache_dir: 模型缓存目录
            max_cache_models: 最大缓存模型数量
            cache_ttl: 缓存生存时间（秒）
            auto_cleanup: 是否自动清理过期缓存
        """
        # 缓存目录
        self.cache_dir = cache_dir or (settings.BASE_DIR / "model_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 缓存配置
        self.max_cache_models = max_cache_models
        self.cache_ttl = cache_ttl
        self.auto_cleanup = auto_cleanup
        
        # 加载的模型实例缓存
        self._models_cache: Dict[str, Dict[str, Any]] = {}  # {model_id: {'instance': model, 'last_used': timestamp}}
        
        # 模型元数据记录
        self._models_metadata: Dict[str, Dict[str, Any]] = {}  # {model_id: {'type': type, 'version': version, ...}}
        
        # 线程锁，用于线程安全的模型管理
        self._lock = threading.RLock()
        
        # 加载模型目录
        self._load_model_registry()
        
        logger.info(f"Model loader initialized with cache directory: {self.cache_dir}")
        
        # 启动后台清理任务（如果启用）
        if self.auto_cleanup:
            self._start_cleanup_thread()
    
    def get_model(self, 
                 model_type: str, 
                 version: Optional[str] = None,
                 force_reload: bool = False,
                 **kwargs) -> MultiModalModelInterface:
        """
        获取指定类型和版本的模型
        
        Args:
            model_type: 模型类型名称
            version: 模型版本，如果为None则使用最新版本
            force_reload: 是否强制重新加载
            **kwargs: 传递给模型的其他参数
            
        Returns:
            MultiModalModelInterface: 模型实例
        """
        with self._lock:
            # 生成模型ID
            model_id = self._generate_model_id(model_type, version, **kwargs)
            
            # 检查缓存中是否有可用模型
            if not force_reload and model_id in self._models_cache:
                logger.debug(f"Using cached model: {model_id}")
                # 更新最后使用时间
                self._models_cache[model_id]["last_used"] = time.time()
                return self._models_cache[model_id]["instance"]
            
            # 如果需要重新加载或缓存中没有，创建新模型
            logger.info(f"Loading model: {model_type} (version: {version or 'latest'})")
            
            # 传递版本信息到kwargs中（如果有）
            if version:
                kwargs["model_name"] = version
                
            # 使用工厂创建模型实例
            try:
                model = create_model(model_type, **kwargs)
                
                # 确保模型初始化
                if not model.is_initialized():
                    success = model.initialize()
                    if not success:
                        logger.error(f"Failed to initialize model: {model_type}")
                        raise RuntimeError(f"Failed to initialize model: {model_type}")
                
                # 缓存模型实例
                self._cache_model(model_id, model)
                
                # 更新或添加元数据
                self._update_model_metadata(model_id, {
                    "type": model_type,
                    "version": version or "latest",
                    "last_loaded": time.time()
                })
                
                return model
                
            except Exception as e:
                logger.error(f"Error loading model {model_type}: {str(e)}")
                raise
    
    def preload_models(self, model_types: List[str]) -> Dict[str, bool]:
        """
        预加载指定类型的模型
        
        Args:
            model_types: 要预加载的模型类型列表
            
        Returns:
            Dict[str, bool]: 每种模型类型的加载结果
        """
        results = {}
        
        for model_type in model_types:
            try:
                self.get_model(model_type)
                results[model_type] = True
                logger.info(f"Preloaded model: {model_type}")
            except Exception as e:
                results[model_type] = False
                logger.error(f"Failed to preload model {model_type}: {str(e)}")
        
        return results
    
    def unload_model(self, model_id: str) -> bool:
        """
        卸载指定的模型
        
        Args:
            model_id: 模型ID
            
        Returns:
            bool: 是否成功卸载
        """
        with self._lock:
            if model_id in self._models_cache:
                try:
                    # 删除模型实例（触发Python垃圾回收）
                    del self._models_cache[model_id]
                    logger.info(f"Unloaded model: {model_id}")
                    return True
                except Exception as e:
                    logger.error(f"Error unloading model {model_id}: {str(e)}")
            
            return False
    
    def get_loaded_models(self) -> List[Dict[str, Any]]:
        """
        获取当前已加载的所有模型信息
        
        Returns:
            List[Dict[str, Any]]: 模型信息列表
        """
        with self._lock:
            loaded_models = []
            
            for model_id, cache_info in self._models_cache.items():
                metadata = self._models_metadata.get(model_id, {})
                model = cache_info["instance"]
                
                try:
                    model_info = model.get_model_info()
                    loaded_models.append({
                        "model_id": model_id,
                        "type": metadata.get("type", "unknown"),
                        "version": metadata.get("version", "unknown"),
                        "last_used": cache_info["last_used"],
                        "model_info": model_info
                    })
                except Exception as e:
                    logger.error(f"Error getting info for model {model_id}: {str(e)}")
                    loaded_models.append({
                        "model_id": model_id,
                        "type": metadata.get("type", "unknown"),
                        "version": metadata.get("version", "unknown"),
                        "last_used": cache_info["last_used"],
                        "error": str(e)
                    })
            
            return loaded_models
    
    def cleanup(self, force: bool = False) -> int:
        """
        清理未使用的模型缓存
        
        Args:
            force: 是否强制清理所有缓存
            
        Returns:
            int: 清理的模型数量
        """
        with self._lock:
            if len(self._models_cache) <= self.max_cache_models and not force:
                return 0
                
            now = time.time()
            models_to_remove = []
            
            # 如果是强制清理，清理所有模型
            if force:
                models_to_remove = list(self._models_cache.keys())
            else:
                # 根据最后使用时间和最大缓存数量选择要清理的模型
                # 先按最后使用时间排序
                sorted_models = sorted(
                    self._models_cache.items(),
                    key=lambda x: x[1]["last_used"]
                )
                
                # 清理过期模型
                expired_models = [
                    model_id for model_id, info in sorted_models
                    if now - info["last_used"] > self.cache_ttl
                ]
                models_to_remove.extend(expired_models)
                
                # 如果还是超过最大缓存数，清理最旧的几个
                if len(self._models_cache) - len(expired_models) > self.max_cache_models:
                    oldest_first = [model_id for model_id, _ in sorted_models if model_id not in expired_models]
                    excess_count = len(self._models_cache) - len(expired_models) - self.max_cache_models
                    models_to_remove.extend(oldest_first[:excess_count])
            
            # 执行卸载
            unloaded_count = 0
            for model_id in models_to_remove:
                if self.unload_model(model_id):
                    unloaded_count += 1
            
            logger.info(f"Cleaned up {unloaded_count} models from cache")
            return unloaded_count
    
    def _cache_model(self, model_id: str, model: MultiModalModelInterface) -> None:
        """
        将模型添加到缓存
        
        Args:
            model_id: 模型ID
            model: 模型实例
        """
        # 添加到缓存前检查是否需要清理
        if len(self._models_cache) >= self.max_cache_models:
            self._cleanup_least_used()
            
        self._models_cache[model_id] = {
            "instance": model,
            "last_used": time.time()
        }
    
    def _cleanup_least_used(self) -> None:
        """
        清理最少使用的模型
        """
        if not self._models_cache:
            return
            
        # 找出最少使用的模型
        oldest_model_id = min(
            self._models_cache.items(),
            key=lambda x: x[1]["last_used"]
        )[0]
        
        # 卸载该模型
        self.unload_model(oldest_model_id)
    
    def _generate_model_id(self, model_type: str, version: Optional[str], **kwargs) -> str:
        """
        生成唯一的模型ID
        
        Args:
            model_type: 模型类型
            version: 模型版本
            **kwargs: 其他参数
            
        Returns:
            str: 模型ID
        """
        # 基础ID：类型+版本
        base_id = f"{model_type}_{version or 'latest'}"
        
        # 添加关键参数到ID（如果存在）
        key_params = []
        for param_name in ["model_name", "api_key", "device"]:
            if param_name in kwargs:
                # 对于api_key，只使用前几位字符
                if param_name == "api_key" and kwargs[param_name]:
                    key_params.append(f"{param_name}={kwargs[param_name][:8]}...")
                else:
                    key_params.append(f"{param_name}={kwargs[param_name]}")
        
        # 如果有关键参数，添加到ID
        if key_params:
            return f"{base_id}_{'.'.join(key_params)}"
        
        return base_id
    
    def _update_model_metadata(self, model_id: str, metadata: Dict[str, Any]) -> None:
        """
        更新模型元数据
        
        Args:
            model_id: 模型ID
            metadata: 元数据字典
        """
        if model_id in self._models_metadata:
            self._models_metadata[model_id].update(metadata)
        else:
            self._models_metadata[model_id] = metadata
            
        # 保存到磁盘
        self._save_model_registry()
    
    def _save_model_registry(self) -> None:
        """
        保存模型注册表到磁盘
        """
        try:
            registry_path = self.cache_dir / "model_registry.json"
            
            with open(registry_path, "w", encoding="utf-8") as f:
                # 只保存元数据，不保存模型实例
                json.dump(self._models_metadata, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving model registry: {str(e)}")
    
    def _load_model_registry(self) -> None:
        """
        从磁盘加载模型注册表
        """
        try:
            registry_path = self.cache_dir / "model_registry.json"
            
            if registry_path.exists():
                with open(registry_path, "r", encoding="utf-8") as f:
                    self._models_metadata = json.load(f)
                    
                logger.info(f"Loaded model registry with {len(self._models_metadata)} entries")
        except Exception as e:
            logger.error(f"Error loading model registry: {str(e)}")
            # 如果加载失败，使用空注册表
            self._models_metadata = {}
    
    def _start_cleanup_thread(self) -> None:
        """
        启动后台清理线程
        """
        def cleanup_task():
            while self.auto_cleanup:
                # 每隔一段时间执行一次清理
                time.sleep(self.cache_ttl / 2)  # 清理周期为TTL的一半
                try:
                    self.cleanup()
                except Exception as e:
                    logger.error(f"Error in cleanup task: {str(e)}")
        
        # 创建并启动守护线程
        cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
        cleanup_thread.start()
        logger.info("Started automatic model cleanup thread")


# 创建单例实例，方便直接导入使用
model_loader = ModelLoader()


def get_model(model_type: str, version: Optional[str] = None, **kwargs) -> MultiModalModelInterface:
    """
    获取模型的便捷函数
    
    Args:
        model_type: 模型类型
        version: 模型版本
        **kwargs: 其他参数
        
    Returns:
        MultiModalModelInterface: 模型实例
    """
    return model_loader.get_model(model_type, version, **kwargs)


def preload_default_models() -> None:
    """
    预加载默认模型
    """
    default_models = []
    
    # 从设置中获取默认模型
    if settings.LLM_MODEL:
        default_models.append(settings.LLM_MODEL)
    
    # 添加常用模型
    for model_type in ["qwen-vl"]:  
        if model_type not in default_models:
            default_models.append(model_type)
    
    # 执行预加载
    if default_models:
        logger.info(f"Preloading default models: {', '.join(default_models)}")
        model_loader.preload_models(default_models)