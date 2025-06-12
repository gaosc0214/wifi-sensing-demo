import json
import os
import pandas as pd
import numpy as np
from PIL import Image
import glob

class ResultsLoader:
    """专门用于加载results目录中的所有实验结果"""
    
    def __init__(self, results_path='results'):
        self.results_path = results_path
        self.single_task_models = ['mlp', 'lstm', 'resnet18', 'transformer', 'vit', 'patchtst', 'timesformer1d']
        self.multi_task_models = ['transformer', 'patchtst', 'timesformer1d']
        self.multi_tasks = ['HumanActivityRecognition', 'HumanIdentification', 'ProximityRecognition']
    
    def load_single_task_results(self):
        """加载所有单任务模型的结果"""
        results = {}
        har_path = os.path.join(self.results_path, 'HumanActivityRecognition')
        
        if not os.path.exists(har_path):
            return {}
        
        for model in self.single_task_models:
            model_path = os.path.join(har_path, model)
            if os.path.exists(model_path):
                # 加载best_performance.json
                perf_file = os.path.join(model_path, 'best_performance.json')
                if os.path.exists(perf_file):
                    with open(perf_file, 'r') as f:
                        results[model] = json.load(f)
                    
                    # 添加模型特定信息
                    results[model]['model_name'] = model
                    results[model]['type'] = 'single_task'
                    
                    # 查找实验目录
                    exp_dirs = [d for d in os.listdir(model_path) if d.startswith('params_')]
                    if exp_dirs:
                        best_exp = results[model].get('best_experiment_id', exp_dirs[0])
                        exp_path = os.path.join(model_path, best_exp)
                        
                        # 加载训练历史
                        train_history_file = os.path.join(exp_path, f'{model}_HumanActivityRecognition_train_history.csv')
                        if os.path.exists(train_history_file):
                            results[model]['train_history'] = pd.read_csv(train_history_file)
                        
                        # 加载混淆矩阵图片路径
                        results[model]['confusion_matrices'] = {}
                        for split in ['test', 'test_cross_device', 'test_cross_env', 'test_cross_user', 'val']:
                            cm_file = os.path.join(exp_path, f'confusion_matrix_{split}.png')
                            if os.path.exists(cm_file):
                                results[model]['confusion_matrices'][split] = cm_file
                        
                        # 加载分类报告
                        results[model]['classification_reports'] = {}
                        for split in ['test', 'test_cross_device', 'test_cross_env', 'test_cross_user', 'val']:
                            report_file = os.path.join(exp_path, f'classification_report_{split}.csv')
                            if os.path.exists(report_file):
                                results[model]['classification_reports'][split] = pd.read_csv(report_file, index_col=0)
        
        return results
    
    def load_multi_task_results(self):
        """加载所有多任务模型的结果"""
        results = {}
        multitask_path = os.path.join(self.results_path, 'multitask')
        
        if not os.path.exists(multitask_path):
            return {}
        
        # 为每个任务加载结果
        for task in self.multi_tasks:
            task_path = os.path.join(multitask_path, task)
            if os.path.exists(task_path):
                results[task] = {}
                
                for model in self.multi_task_models:
                    model_path = os.path.join(task_path, model)
                    if os.path.exists(model_path):
                        # 加载best_performance.json
                        perf_file = os.path.join(model_path, 'best_performance.json')
                        if os.path.exists(perf_file):
                            with open(perf_file, 'r') as f:
                                results[task][model] = json.load(f)
                            
                            results[task][model]['model_name'] = model
                            results[task][model]['task_name'] = task
                            results[task][model]['type'] = 'multi_task'
                            
                            # 查找实验目录
                            exp_dirs = [d for d in os.listdir(model_path) if d.startswith('params_')]
                            if exp_dirs:
                                best_exp = results[task][model].get('best_experiment_id', exp_dirs[0])
                                exp_path = os.path.join(model_path, best_exp)
                                
                                # 加载混淆矩阵图片路径
                                results[task][model]['confusion_matrices'] = {}
                                for split in ['test', 'test_cross_device', 'test_cross_env', 'test_cross_user']:
                                    cm_file = os.path.join(exp_path, f'confusion_matrix_{split}.png')
                                    if os.path.exists(cm_file):
                                        results[task][model]['confusion_matrices'][split] = cm_file
                                
                                # 加载分类报告
                                results[task][model]['classification_reports'] = {}
                                for split in ['test', 'test_cross_device', 'test_cross_env', 'test_cross_user']:
                                    report_file = os.path.join(exp_path, f'classification_report_{split}.csv')
                                    if os.path.exists(report_file):
                                        results[task][model]['classification_reports'][split] = pd.read_csv(report_file, index_col=0)
        
        return results
    
    def get_model_comparison_data(self):
        """获取用于模型对比的数据"""
        single_task = self.load_single_task_results()
        
        comparison_data = {}
        for model, data in single_task.items():
            comparison_data[model] = {
                'test_accuracy': data.get('best_test_accuracies', {}).get('test', 0),
                'cross_device': data.get('best_test_accuracies', {}).get('test_cross_device', 0),
                'cross_env': data.get('best_test_accuracies', {}).get('test_cross_env', 0),
                'cross_user': data.get('best_test_accuracies', {}).get('test_cross_user', 0),
                'val_accuracy': data.get('best_val_accuracy', 0),
                'test_f1': data.get('best_test_f1_scores', {}).get('test', 0)
            }
        
        return comparison_data
    
    def get_multi_task_comparison_data(self):
        """获取多任务模型对比数据"""
        multi_task = self.load_multi_task_results()
        
        comparison_data = {}
        for task, models in multi_task.items():
            comparison_data[task] = {}
            for model, data in models.items():
                comparison_data[task][model] = {
                    'test_accuracy': data.get('best_test_accuracies', {}).get('test', 0),
                    'cross_device': data.get('best_test_accuracies', {}).get('test_cross_device', 0),
                    'cross_env': data.get('best_test_accuracies', {}).get('test_cross_env', 0),
                    'cross_user': data.get('best_test_accuracies', {}).get('test_cross_user', 0),
                    'val_accuracy': data.get('best_val_accuracy', 0),
                    'test_f1': data.get('best_test_f1_scores', {}).get('test', 0)
                }
        
        return comparison_data
    
    def get_best_model_info(self):
        """获取最佳模型信息"""
        single_task = self.load_single_task_results()
        
        best_models = {}
        
        # 按不同指标找最佳模型
        best_test_acc = max(single_task.items(), 
                           key=lambda x: x[1].get('best_test_accuracies', {}).get('test', 0))
        best_cross_device = max(single_task.items(), 
                               key=lambda x: x[1].get('best_test_accuracies', {}).get('test_cross_device', 0))
        best_cross_env = max(single_task.items(), 
                            key=lambda x: x[1].get('best_test_accuracies', {}).get('test_cross_env', 0))
        best_cross_user = max(single_task.items(), 
                             key=lambda x: x[1].get('best_test_accuracies', {}).get('test_cross_user', 0))
        
        best_models['test_accuracy'] = {
            'model': best_test_acc[0],
            'accuracy': best_test_acc[1].get('best_test_accuracies', {}).get('test', 0)
        }
        best_models['cross_device'] = {
            'model': best_cross_device[0],
            'accuracy': best_cross_device[1].get('best_test_accuracies', {}).get('test_cross_device', 0)
        }
        best_models['cross_env'] = {
            'model': best_cross_env[0],
            'accuracy': best_cross_env[1].get('best_test_accuracies', {}).get('test_cross_env', 0)
        }
        best_models['cross_user'] = {
            'model': best_cross_user[0],
            'accuracy': best_cross_user[1].get('best_test_accuracies', {}).get('test_cross_user', 0)
        }
        
        return best_models
    
    def get_available_models(self):
        """获取可用的模型列表"""
        single_task = self.load_single_task_results()
        multi_task = self.load_multi_task_results()
        
        return {
            'single_task': list(single_task.keys()),
            'multi_task': {task: list(models.keys()) for task, models in multi_task.items()}
        }
    
    def load_confusion_matrix_image(self, model_name, split='test', task_type='single_task', task_name=None):
        """加载混淆矩阵图片"""
        try:
            if task_type == 'single_task':
                single_task = self.load_single_task_results()
                if model_name in single_task and 'confusion_matrices' in single_task[model_name]:
                    img_path = single_task[model_name]['confusion_matrices'].get(split)
                    if img_path and os.path.exists(img_path):
                        return Image.open(img_path)
            
            elif task_type == 'multi_task' and task_name:
                multi_task = self.load_multi_task_results()
                if task_name in multi_task and model_name in multi_task[task_name]:
                    if 'confusion_matrices' in multi_task[task_name][model_name]:
                        img_path = multi_task[task_name][model_name]['confusion_matrices'].get(split)
                        if img_path and os.path.exists(img_path):
                            return Image.open(img_path)
            
            return None
            
        except Exception as e:
            print(f"Error loading confusion matrix image: {e}")
            return None 