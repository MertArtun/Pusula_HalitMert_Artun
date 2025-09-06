"""
Fiziksel Tıp & Rehabilitasyon Veri Analizi - Ortak Logging Modülü

Bu modül tüm script'ler için tutarlı logging konfigürasyonu sağlar.
"""

import logging
import sys
from pathlib import Path


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Ortak logger konfigürasyonu oluşturur.
    
    Args:
        name (str): Logger adı (genellikle __name__)
        level (int): Log seviyesi (varsayılan: INFO)
        
    Returns:
        logging.Logger: Konfigüre edilmiş logger
    """
    # Logger oluştur
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Eğer handler'lar zaten eklenmişse tekrar ekleme
    if logger.handlers:
        return logger
    
    # Console handler oluştur
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Formatter oluştur
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Handler'ı logger'a ekle
    logger.addHandler(console_handler)
    
    return logger


# Basic config (eski script'lerle uyumluluk için)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Default logger
logger = logging.getLogger(__name__)

# Script'ler için hazır logger'lar
eda_logger = setup_logger('eda_script')
preprocess_logger = setup_logger('preprocess_script') 
features_logger = setup_logger('features_script')
