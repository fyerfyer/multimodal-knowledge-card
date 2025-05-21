import os
import cv2
import numpy as np
import io
from typing import Union, Tuple, Dict, Any, List
from pathlib import Path
import base64
from PIL import Image

from app.utils.logger import logger

def read_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    读取图像文件到numpy数组
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        np.ndarray: 图像数组，格式为BGR
    
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 图像读取失败
    """
    # 确保路径是字符串类型
    if isinstance(image_path, Path):
        image_path = str(image_path)
        
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # 使用OpenCV读取图像
    image = cv2.imread(image_path)
    
    if image is None:
        logger.error(f"Failed to read image: {image_path}")
        raise ValueError(f"Failed to read image: {image_path}")
    
    logger.debug(f"Image read successfully: {image_path}, shape: {image.shape}")
    return image

def convert_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    将BGR格式图像转换为RGB格式
    
    Args:
        image: BGR格式的numpy数组图像
        
    Returns:
        np.ndarray: RGB格式的图像
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def resize_image(image: np.ndarray, 
                 target_size: Tuple[int, int] = None, 
                 max_size: int = 1024, 
                 keep_ratio: bool = True) -> np.ndarray:
    """
    调整图像大小
    
    Args:
        image: 输入图像
        target_size: 目标尺寸 (宽, 高)，如果为None则根据max_size自动计算
        max_size: 图像的最大边长
        keep_ratio: 是否保持宽高比
        
    Returns:
        np.ndarray: 调整大小后的图像
    """
    h, w = image.shape[:2]
    
    # 如果没有指定目标尺寸，则根据max_size自动计算
    if target_size is None:
        # 找到较大的边
        if max(h, w) > max_size:
            if w > h:
                new_w = max_size
                new_h = int(h * max_size / w)
            else:
                new_h = max_size
                new_w = int(w * max_size / h)
            target_size = (new_w, new_h)
        else:
            return image  # 如果图像已经小于max_size，则不调整
    
    # 调整图像大小
    if keep_ratio:
        # 计算缩放比例
        scale = min(target_size[0] / w, target_size[1] / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    logger.debug(f"Image resized from {(h, w)} to {resized.shape[:2]}")
    return resized

def normalize_image(image: np.ndarray, mean: List[float] = [0.485, 0.456, 0.406], 
                   std: List[float] = [0.229, 0.224, 0.225]) -> np.ndarray:
    """
    标准化图像（减均值、除以标准差）
    
    Args:
        image: 输入图像 (RGB, 0-255)
        mean: RGB通道均值
        std: RGB通道标准差
        
    Returns:
        np.ndarray: 标准化后的图像
    """
    # 将图像转换为浮点型并缩放到0-1
    image = image.astype(np.float32) / 255.0
    
    # 标准化
    for i in range(3):
        image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
    
    return image

def preprocess_image_for_blip(image_path: Union[str, Path, np.ndarray]) -> np.ndarray:
    """
    预处理图像以适配BLIP模型
    
    Args:
        image_path: 图像路径或numpy数组
        
    Returns:
        np.ndarray: 预处理后的图像，RGB格式，已调整大小和标准化
    """
    # 如果输入是路径，先读取图像
    if isinstance(image_path, (str, Path)):
        image = read_image(image_path)
    else:
        image = image_path.copy()
    
    # 转换为RGB
    image = convert_to_rgb(image)
    
    # 调整大小（根据BLIP模型的要求）
    image = resize_image(image, max_size=384)  # BLIP通常使用的大小
    
    logger.info(f"Image preprocessed for BLIP model, final shape: {image.shape}")
    return image

def crop_image(image: np.ndarray, 
              bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    裁剪图像
    
    Args:
        image: 输入图像
        bbox: 裁剪区域 (x, y, width, height)
        
    Returns:
        np.ndarray: 裁剪后的图像
    """
    x, y, w, h = bbox
    return image[y:y+h, x:x+w]

def encode_image_to_base64(image: Union[np.ndarray, str, Path]) -> str:
    """
    将图像编码为base64字符串
    
    Args:
        image: 图像数组或图像路径
        
    Returns:
        str: base64编码的图像
    """
    if isinstance(image, (str, Path)):
        # 如果是路径，直接读取文件
        with open(image, "rb") as f:
            image_bytes = f.read()
    else:
        # 如果是numpy数组，先转换为PIL图像，再编码为JPEG
        pil_image = Image.fromarray(
            image if len(image.shape) == 2 or image.shape[2] == 3 
            else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        )
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()
    
    # 编码为base64
    base64_str = base64.b64encode(image_bytes).decode("utf-8")
    
    return base64_str

def decode_base64_to_image(base64_str: str) -> np.ndarray:
    """
    将base64字符串解码为图像
    
    Args:
        base64_str: base64编码的图像
        
    Returns:
        np.ndarray: 解码后的图像
    """
    image_bytes = base64.b64decode(base64_str)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    return image

def save_image(image: np.ndarray, output_path: Union[str, Path]) -> None:
    """
    保存图像
    
    Args:
        image: 图像数组
        output_path: 输出路径
    """
    # 确保目录存在
    if isinstance(output_path, str):
        output_path = Path(output_path)
    
    os.makedirs(output_path.parent, exist_ok=True)
    
    # 保存图像
    cv2.imwrite(str(output_path), image)
    logger.debug(f"Image saved to: {output_path}")
    
def get_image_info(image_path: Union[str, Path]) -> Dict[str, Any]:
    """
    获取图像信息（尺寸、通道等）
    
    Args:
        image_path: 图像路径
        
    Returns:
        Dict[str, Any]: 图像信息字典
    """
    image = read_image(image_path)
    h, w = image.shape[:2]
    channels = 1 if len(image.shape) == 2 else image.shape[2]
    
    file_size = os.path.getsize(image_path)
    file_name = os.path.basename(str(image_path))
    
    return {
        "width": w,
        "height": h,
        "channels": channels,
        "file_size": file_size,
        "file_name": file_name,
        "shape": image.shape
    }