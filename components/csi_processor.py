import numpy as np
import os
import h5py
import pandas as pd
import json
from pathlib import Path

class CSIProcessor:
    """用于处理CSI数据的类"""
    
    def __init__(self):
        self.activity_names = [
            "jumping",      # 跳跃
            "running",      # 跑步 
            "seated-breathing",  # 静坐呼吸
            "walking",      # 走路
            "wavinghand"    # 挥手
        ]
        
        self.activity_mapping = {
            0: "jumping",
            1: "running", 
            2: "seated-breathing",
            3: "walking",
            4: "wavinghand"
        }
        
        self.chinese_names = {
            "jumping": "跳跃 🦘",
            "running": "跑步 🏃",
            "seated-breathing": "静坐呼吸 🧘",
            "walking": "走路 🚶",
            "wavinghand": "挥手 👋"
        }
    
    def load_csi_sample(self, file_path):
        """
        加载CSI样本文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            CSI数据数组，如果失败返回None
        """
        try:
            if not os.path.exists(file_path):
                print(f"❌ 文件不存在: {file_path}")
                return None
            
            # 根据文件扩展名选择加载方法
            if file_path.endswith('.npy'):
                data = np.load(file_path)
            elif file_path.endswith('.h5') or file_path.endswith('.hdf5'):
                data = self._load_h5_file(file_path)
            elif file_path.endswith('.csv'):
                data = pd.read_csv(file_path).values
            else:
                print(f"⚠️ 不支持的文件格式: {file_path}")
                return self._create_dummy_data()
            
            # 验证数据形状
            if data is None or len(data.shape) < 2:
                print(f"⚠️ 数据格式不正确: {file_path}")
                return self._create_dummy_data()
            
            # 标准化数据形状
            data = self._normalize_data_shape(data)
            
            return data
            
        except Exception as e:
            print(f"❌ 加载CSI数据失败: {e}")
            return self._create_dummy_data()
    
    def _load_h5_file(self, file_path):
        """加载HDF5文件"""
        try:
            with h5py.File(file_path, 'r') as f:
                # 尝试常见的数据键名
                possible_keys = ['csi', 'data', 'csi_data', 'X', 'features']
                for key in possible_keys:
                    if key in f:
                        return f[key][:]
                
                # 如果没有找到，取第一个数据集
                keys = list(f.keys())
                if keys:
                    return f[keys[0]][:]
                    
            return None
            
        except Exception as e:
            print(f"❌ 加载H5文件失败: {e}")
            return None
    
    def _normalize_data_shape(self, data):
        """标准化数据形状为 (time_steps, features)"""
        try:
            if len(data.shape) == 1:
                # 一维数据，reshape为二维
                data = data.reshape(-1, 1)
            elif len(data.shape) == 3:
                # 三维数据，展平最后两个维度
                data = data.reshape(data.shape[0], -1)
            
            # 确保时间步长合理
            if data.shape[0] < 100:
                # 如果时间步太少，转置数据
                data = data.T
            
            return data
            
        except Exception as e:
            print(f"❌ 标准化数据形状失败: {e}")
            return self._create_dummy_data()
    
    def _create_dummy_data(self):
        """创建虚拟CSI数据用于演示"""
        # 创建500个时间步，232个特征的虚拟数据
        time_steps = 500
        features = 232
        
        # 创建一些有模式的数据
        t = np.linspace(0, 4*np.pi, time_steps)
        data = np.zeros((time_steps, features))
        
        for i in range(features):
            # 不同频率的正弦波模拟不同子载波
            frequency = 0.1 + (i / features) * 2
            amplitude = 0.5 + 0.5 * np.sin(i * 0.1)
            noise = np.random.normal(0, 0.1, time_steps)
            data[:, i] = amplitude * np.sin(frequency * t) + noise
        
        return data
    
    def get_activity_name(self, prediction_index):
        """
        根据预测索引获取活动名称
        
        Args:
            prediction_index: 预测的类别索引
            
        Returns:
            活动名称
        """
        if prediction_index in self.activity_mapping:
            return self.activity_mapping[prediction_index]
        else:
            return self.activity_names[0]  # 默认返回第一个活动
    
    def get_chinese_name(self, activity_name):
        """
        获取活动的中文名称
        
        Args:
            activity_name: 英文活动名称
            
        Returns:
            中文活动名称
        """
        return self.chinese_names.get(activity_name, activity_name)
    
    def extract_features(self, csi_data):
        """
        从CSI数据中提取特征
        
        Args:
            csi_data: CSI数据
            
        Returns:
            特征字典
        """
        try:
            features = {}
            
            # 基本统计特征
            features['mean'] = np.mean(csi_data)
            features['std'] = np.std(csi_data)
            features['min'] = np.min(csi_data)
            features['max'] = np.max(csi_data)
            
            # 时域特征
            features['variance'] = np.var(csi_data)
            features['rms'] = np.sqrt(np.mean(csi_data**2))
            
            # 空间特征（跨子载波）
            if len(csi_data.shape) >= 2:
                features['spatial_mean'] = np.mean(csi_data, axis=0)
                features['spatial_std'] = np.std(csi_data, axis=0)
                features['correlation'] = np.corrcoef(csi_data.T) if csi_data.shape[1] > 1 else np.array([[1]])
            
            # 时间特征（跨时间步）
            if csi_data.shape[0] > 1:
                features['temporal_diff'] = np.diff(csi_data, axis=0)
                features['energy'] = np.sum(csi_data**2, axis=1)
            
            return features
            
        except Exception as e:
            print(f"❌ 特征提取失败: {e}")
            return {}
    
    def preprocess_for_model(self, csi_data):
        """
        为模型预处理CSI数据
        
        Args:
            csi_data: 原始CSI数据
            
        Returns:
            预处理后的数据
        """
        try:
            # 确保数据形状正确
            data = self._normalize_data_shape(csi_data)
            
            # Z-score标准化
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            std[std == 0] = 1  # 避免除零
            data = (data - mean) / std
            
            # 裁剪或填充到固定长度
            target_length = 500
            if data.shape[0] > target_length:
                # 裁剪
                data = data[:target_length, :]
            elif data.shape[0] < target_length:
                # 填充
                padding = np.zeros((target_length - data.shape[0], data.shape[1]))
                data = np.vstack([data, padding])
            
            return data
            
        except Exception as e:
            print(f"❌ 数据预处理失败: {e}")
            return self._create_dummy_data()
    
    def get_sample_files(self, directory='csi_samples'):
        """
        获取样本文件列表
        
        Args:
            directory: 样本文件目录
            
        Returns:
            文件路径列表
        """
        try:
            if not os.path.exists(directory):
                print(f"⚠️ 目录不存在: {directory}")
                return []
            
            # 支持的文件格式
            extensions = ['.npy', '.h5', '.hdf5', '.csv']
            files = []
            
            for ext in extensions:
                pattern = f"*{ext}"
                files.extend(Path(directory).glob(pattern))
            
            return [str(f) for f in files]
            
        except Exception as e:
            print(f"❌ 获取文件列表失败: {e}")
            return []
    
    def analyze_data_quality(self, csi_data):
        """
        分析数据质量
        
        Args:
            csi_data: CSI数据
            
        Returns:
            质量报告字典
        """
        try:
            report = {}
            
            # 基本信息
            report['shape'] = csi_data.shape
            report['size'] = csi_data.size
            report['dtype'] = str(csi_data.dtype)
            
            # 缺失值检查
            report['has_nan'] = np.isnan(csi_data).any()
            report['has_inf'] = np.isinf(csi_data).any()
            report['nan_count'] = np.isnan(csi_data).sum()
            
            # 数据范围
            report['min_value'] = np.min(csi_data)
            report['max_value'] = np.max(csi_data)
            report['value_range'] = report['max_value'] - report['min_value']
            
            # 统计信息
            report['mean'] = np.mean(csi_data)
            report['std'] = np.std(csi_data)
            report['zero_ratio'] = np.sum(csi_data == 0) / csi_data.size
            
            # 质量评分 (0-1)
            quality_score = 1.0
            if report['has_nan'] or report['has_inf']:
                quality_score -= 0.3
            if report['zero_ratio'] > 0.8:
                quality_score -= 0.2
            if report['std'] < 0.001:
                quality_score -= 0.3
            
            report['quality_score'] = max(0, quality_score)
            
            return report
            
        except Exception as e:
            print(f"❌ 数据质量分析失败: {e}")
            return {'quality_score': 0, 'error': str(e)} 