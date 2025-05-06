import yaml
from addict import Dict
from datetime import datetime
import os
from types import SimpleNamespace

class Config:
    """配置类，支持字典方式和属性方式访问配置项"""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            elif isinstance(value, (list, tuple)):
                # 如果是列表或元组，检查其中是否包含字典
                if any(isinstance(item, dict) for item in value):
                    setattr(self, key, [Config(item) if isinstance(item, dict) else item for item in value])
                else:
                    setattr(self, key, value)
            else:
                setattr(self, key, value)
    
    def get(self, key, default=None):
        """获取配置项，支持默认值"""
        return getattr(self, key, default)
    
    def __getitem__(self, key):
        """支持字典方式访问"""
        return getattr(self, key)
    
    def __contains__(self, key):
        """支持 in 操作符"""
        return hasattr(self, key)
    
    def __len__(self):
        """支持 len() 操作"""
        if hasattr(self, '__dict__'):
            return len(self.__dict__)
        return 0
    
    def __iter__(self):
        """支持迭代操作"""
        return iter(self.__dict__)
    
    def keys(self):
        """返回所有配置键"""
        return self.__dict__.keys()
    
    def values(self):
        """返回所有配置值"""
        return self.__dict__.values()
    
    def items(self):
        """返回所有配置项"""
        return self.__dict__.items()
    
    def to_dict(self):
        """转换为字典"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                config_dict[key] = value.to_dict()
            elif isinstance(value, (list, tuple)):
                # 处理列表中的 Config 对象
                config_dict[key] = [
                    item.to_dict() if isinstance(item, Config) else item 
                    for item in value
                ]
            else:
                config_dict[key] = value
        return config_dict
    
    def __repr__(self):
        """更好的字符串表示"""
        items = [f"{k}={v!r}" for k, v in self.__dict__.items()]
        return f"Config({', '.join(items)})"

def load_config(config_path):
    """加载yaml配置文件"""
    with open(config_path, 'r', encoding='utf-8') as file:
        config_dict = yaml.safe_load(file)
    
    # 将字典转换为Config对象
    config = Config(config_dict)
    return config

def create_output_dir_and_save_config(output_dir_prefix, config, specified_dir=None):
    if specified_dir:
        output_dir = specified_dir
    else:
        # Generate the output directory name with the current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{output_dir_prefix}_{timestamp}"
    
    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define the path for the config file
    config_path = os.path.join(output_dir, "config.yaml")
    
    # Convert config to dict
    config_dict = config.to_dict() if isinstance(config, Config) else vars(config)
    
    # Save the config as a YAML file
    with open(config_path, 'w') as file:
        yaml.dump(config_dict, file)
    
    return output_dir