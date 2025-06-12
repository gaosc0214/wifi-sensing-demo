import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import random
import logging
import warnings

# 彻底过滤掉所有警告，包括matplotlib字体警告和PyTorch警告
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', module='matplotlib')
warnings.filterwarnings('ignore', module='matplotlib.font_manager')
warnings.filterwarnings('ignore', module='torch')
warnings.filterwarnings('ignore', module='torch._classes')

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端避免字体问题

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加组件路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'components'))

from model_loader import ModelLoader
from csi_processor import CSIProcessor
from visualizer import (
    plot_csi_heatmap, 
    plot_model_comparison_bar,
    plot_performance_radar
)
from results_loader import ResultsLoader

# 页面配置
st.set_page_config(
    page_title="基于深度学习的WiFi CSI多维情景感知平台",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化
@st.cache_resource
def init_loaders():
    """初始化加载器"""
    model_loader = ModelLoader()
    csi_processor = CSIProcessor()
    results_loader = ResultsLoader()
    return model_loader, csi_processor, results_loader

model_loader, csi_processor, results_loader = init_loaders()

def create_dummy_csi_data():
    """创建虚拟CSI数据用于演示"""
    # 创建500个时间步，56个特征的虚拟数据
    time_steps = 500
    features = 56
    
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

def main():
    """主函数"""
    # 侧边栏选择模式
    st.sidebar.title("选择模式")
    mode = st.sidebar.radio(
        "请选择使用模式：",
        ["🔬 研究模式", "🎮 游戏模式"],
        help="研究模式：查看我的实验结果和数据分析；游戏模式：和AI比赛猜CSI活动，很好玩！"
    )
    
    if mode == "🔬 研究模式":
        # 只在研究模式显示介绍内容
        st.title("📡 基于深度学习的WiFi CSI多维情景感知")
        st.markdown("""
        👨‍🎓 **课题内容**：基于深度学习的真实复杂环境WiFi CSI多维情景感知研究
        
        这是我的期末大作业成果展示！本作业研究了如何使用WiFi CSI(Channel State Information)数据进行人体活动识别。
        我测试了7种深度学习模型(MLP、LSTM、ResNet18、Transformer、ViT、PatchTST、TimesFormer1D)，
        并在跨设备、跨环境、跨用户等场景下进行了性能评估。还做了个小游戏用于体验CSI感知技术！🎮
        """)
        research_mode()
    else:
        # 游戏模式简洁显示
        game_mode()

def research_mode():
    """研究模式界面"""
    st.header("🔬 研究模式 - 我的实验结果")
    
    # 标签页
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 模型性能对比", 
        "🧠 单任务分析", 
        "🎯 多任务分析", 
        "🔄 跨域泛化"
    ])
    
    with tab1:
        show_model_comparison()
    
    with tab2:
        show_single_task_analysis()
    
    with tab3:
        show_multi_task_analysis()
    
    with tab4:
        show_cross_domain_analysis()

def show_model_comparison():
    """显示模型对比"""
    st.subheader("📊 模型性能全面对比")
    
    # 获取模型对比数据
    comparison_data = results_loader.get_model_comparison_data()
    
    if not comparison_data:
        st.warning("❌ 没有找到模型对比数据，请检查results目录")
        return
    
    # 创建性能对比图
    col1, col2 = st.columns(2)
    
    with col1:
        # 条形图对比
        fig = plot_model_comparison_bar(comparison_data, "模型性能对比")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 性能排名表
        st.subheader("📈 性能排名")
        
        # 转换数据为DataFrame
        df_data = []
        for model, metrics in comparison_data.items():
            df_data.append({
                '模型': model.upper(),
                '测试准确率': f"{metrics['test_accuracy']:.3f}",
                '跨设备': f"{metrics['cross_device']:.3f}",
                '跨环境': f"{metrics['cross_env']:.3f}",
                '跨用户': f"{metrics['cross_user']:.3f}",
                'F1分数': f"{metrics['test_f1']:.3f}"
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)
    
    # 最佳模型信息
    st.subheader("🏆 最佳模型")
    best_models = results_loader.get_best_model_info()
    
    if best_models:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🎯 最佳测试准确率",
                f"{best_models['test_accuracy']['model'].upper()}",
                f"{best_models['test_accuracy']['accuracy']:.3f}"
            )
        
        with col2:
            st.metric(
                "📱 最佳跨设备",
                f"{best_models['cross_device']['model'].upper()}",
                f"{best_models['cross_device']['accuracy']:.3f}"
            )
        
        with col3:
            st.metric(
                "🌍 最佳跨环境",
                f"{best_models['cross_env']['model'].upper()}",
                f"{best_models['cross_env']['accuracy']:.3f}"
            )
        
        with col4:
            st.metric(
                "👤 最佳跨用户",
                f"{best_models['cross_user']['model'].upper()}",
                f"{best_models['cross_user']['accuracy']:.3f}"
            )

def plot_training_history_comparison(train_histories, metric='accuracy', selected_datasets=None):
    """
    创建所有模型训练历史的对比图
    
    Args:
        train_histories: dict, 包含所有模型的训练历史数据
        metric: str, 'accuracy' 或 'loss'
        selected_datasets: list, 要显示的数据集列表 ['train', 'val']
    
    Returns:
        plotly figure
    """
    try:
        fig = go.Figure()
        
        # 为每个模型添加一条线
        for model_name, history in train_histories.items():
            if metric == 'accuracy':
                train_col = 'Train Accuracy'
                val_col = 'Val Accuracy'
                title = "模型训练准确率对比"
                yaxis_title = "准确率"
            else:
                train_col = 'Train Loss'
                val_col = 'Val Loss'
                title = "模型训练损失对比"
                yaxis_title = "损失"
            
            # 根据选择的数据集添加曲线
            if 'train' in selected_datasets:
                fig.add_trace(go.Scatter(
                    x=history['Epoch'],
                    y=history[train_col],
                    name=f"{model_name} (训练集)",
                    line=dict(dash='solid'),
                    legendgroup=model_name
                ))
            
            if 'val' in selected_datasets:
                fig.add_trace(go.Scatter(
                    x=history['Epoch'],
                    y=history[val_col],
                    name=f"{model_name} (验证集)",
                    line=dict(dash='dot'),
                    legendgroup=model_name
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Epoch",
            yaxis_title=yaxis_title,
            hovermode='x unified',
            legend=dict(
                groupclick="toggleitem",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    except Exception as e:
        st.warning(f"创建训练历史对比图时出错: {e}")
        return None

def show_single_task_analysis():
    """显示单任务分析"""
    st.subheader("🎯 单任务学习分析")
    
    # 获取单任务数据
    single_task_data = results_loader.load_single_task_results()
    
    if not single_task_data:
        st.warning("❌ 没有找到单任务数据")
        return
    
    # 模型选择（用于详细分析）
    model_names = list(single_task_data.keys())
    selected_model = st.selectbox("选择要详细分析的模型:", model_names)
    model_data = single_task_data[selected_model]
    
    # 训练历史
    st.subheader("📈 训练历史")
    
    # 收集所有模型的训练历史
    train_histories = {}
    for model_name, data in single_task_data.items():
        if 'train_history' in data:
            train_histories[model_name] = data['train_history']
    
    if train_histories:
        # 创建两列用于选择
        col1, col2 = st.columns(2)
        
        with col1:
            # 模型多选
            available_models = list(train_histories.keys())
            default_models = [selected_model] if selected_model in available_models else [available_models[0]]
            selected_models = st.multiselect(
                "选择要对比的模型:",
                options=available_models,
                default=default_models,
                help="可以选择多个模型进行对比"
            )
        
        with col2:
            # 数据集选择
            dataset_options = {
                'train': '训练集',
                'val': '验证集'
            }
            selected_datasets = st.multiselect(
                "选择要显示的数据集:",
                options=list(dataset_options.keys()),
                default=['train', 'val'],
                format_func=lambda x: dataset_options[x],
                help="可以选择显示训练集和/或验证集的数据"
            )
        
        if not selected_models:
            st.warning("请至少选择一个模型进行对比")
            return
        
        if not selected_datasets:
            st.warning("请至少选择一个数据集")
            return
        
        # 创建标签页来分别显示准确率和损失
        tab1, tab2 = st.tabs(["准确率对比", "损失对比"])
        
        # 过滤选中的模型数据
        filtered_histories = {
            model: history for model, history in train_histories.items() 
            if model in selected_models
        }
        
        with tab1:
            fig_acc = plot_training_history_comparison(
                filtered_histories, 
                metric='accuracy',
                selected_datasets=selected_datasets
            )
            if fig_acc:
                st.plotly_chart(fig_acc, use_container_width=True)
        
        with tab2:
            fig_loss = plot_training_history_comparison(
                filtered_histories, 
                metric='loss',
                selected_datasets=selected_datasets
            )
            if fig_loss:
                st.plotly_chart(fig_loss, use_container_width=True)
    else:
        st.info("没有找到训练历史数据")
    
    # 混淆矩阵
    st.subheader("🎯 混淆矩阵")
    if 'confusion_matrices' in model_data:
        split_options = list(model_data['confusion_matrices'].keys())
        selected_split = st.selectbox("选择数据集:", split_options)
        
        img = results_loader.load_confusion_matrix_image(
            selected_model, selected_split, 'single_task'
        )
        if img:
            st.image(img, caption=f"{selected_model.upper()} - {selected_split} 混淆矩阵")
        else:
            st.warning(f"无法加载 {selected_split} 的混淆矩阵图片")
    
    # 分类报告
    if 'classification_reports' in model_data:
        st.subheader("📋 详细分类报告")
        split_options = list(model_data['classification_reports'].keys())
        selected_split = st.selectbox("选择数据集:", split_options, key="report_split")
        
        if selected_split in model_data['classification_reports']:
            report_df = model_data['classification_reports'][selected_split]
            st.dataframe(report_df, use_container_width=True)

def show_multi_task_analysis():
    """显示多任务分析"""
    st.subheader("🎯 多任务学习分析")
    
    # 获取多任务数据
    multi_task_data = results_loader.get_multi_task_comparison_data()
    
    if not multi_task_data:
        st.warning("❌ 没有找到多任务数据")
        return
    
    # 任务选择
    task_names = list(multi_task_data.keys())
    selected_task = st.selectbox("选择任务:", task_names)
    
    task_data = multi_task_data[selected_task]
    
    # 任务性能对比
    col1, col2 = st.columns(2)
    
    with col1:
        # 创建对比图
        fig = plot_model_comparison_bar(task_data, f"{selected_task} 模型对比")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 性能表格
        st.subheader("📊 性能详情")
        df_data = []
        for model, metrics in task_data.items():
            df_data.append({
                '模型': model.upper(),
                '测试准确率': f"{metrics['test_accuracy']:.3f}",
                '跨设备': f"{metrics['cross_device']:.3f}",
                '跨环境': f"{metrics['cross_env']:.3f}",
                '跨用户': f"{metrics['cross_user']:.3f}"
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)
    
    # 任务间对比
    st.subheader("📈 任务间性能对比")
    
    # 创建所有任务的对比图
    all_task_data = {}
    for task, models in multi_task_data.items():
        for model, metrics in models.items():
            if model not in all_task_data:
                all_task_data[model] = {}
            all_task_data[model][task] = metrics['test_accuracy']
    
    if all_task_data:
        try:
            # 转换为DataFrame用于可视化
            comparison_df = pd.DataFrame(all_task_data).T
            comparison_df = comparison_df.fillna(0)
            
            # 重塑数据用于plotly
            melted_df = comparison_df.reset_index().melt(id_vars='index', var_name='任务', value_name='准确率')
            melted_df.rename(columns={'index': '模型'}, inplace=True)
            
            fig = px.bar(melted_df, x='模型', y='准确率', color='任务',
                        title="所有任务的模型性能对比",
                        labels={'准确率': '准确率', '模型': '模型'})
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"无法显示任务对比图表: {e}")
            # 显示简单的表格作为备用
            comparison_df = pd.DataFrame(all_task_data).T
            comparison_df = comparison_df.fillna(0)
            st.dataframe(comparison_df, use_container_width=True)

def plot_cross_domain_comparison(model_results, selected_models, selected_datasets):
    """
    创建跨域泛化能力的折线图对比
    
    Args:
        model_results: dict, 包含所有模型的性能数据
        selected_models: list, 选中的模型列表
        selected_datasets: list, 选中的数据集列表
    
    Returns:
        plotly figure
    """
    try:
        fig = go.Figure()
        
        # 数据集映射
        dataset_mapping = {
            'test': 'In-Domain Test',
            'test_cross_device': 'Cross-Device',
            'test_cross_env': 'Cross-Environment',
            'test_cross_user': 'Cross-User'
        }
        
        # 数据集到指标的映射
        dataset_to_metric = {
            'test': 'test_accuracy',
            'test_cross_device': 'cross_device',
            'test_cross_env': 'cross_env',
            'test_cross_user': 'cross_user'
        }
        
        # 为每个选中的模型添加一条线
        for model in selected_models:
            if model in model_results:
                # 收集该模型在所有选中数据集上的性能
                accuracies = []
                for dataset in selected_datasets:
                    metric = dataset_to_metric[dataset]
                    acc = model_results[model].get(metric, 0)
                    accuracies.append(acc)
                
                # 添加折线
                fig.add_trace(go.Scatter(
                    x=[dataset_mapping[d] for d in selected_datasets],
                    y=accuracies,
                    name=model.upper(),
                    mode='lines+markers',
                    marker=dict(size=8)
                ))
        
        fig.update_layout(
            title="模型跨域泛化能力对比",
            xaxis_title="数据集",
            yaxis_title="准确率",
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            yaxis=dict(
                range=[0, 1],  # 准确率范围从0到1
                tickformat='.2%'  # 显示为百分比
            )
        )
        
        return fig
    except Exception as e:
        st.warning(f"创建跨域泛化对比图时出错: {e}")
        return None

def plot_cross_domain_radar(model_results, selected_models):
    """
    创建跨域泛化能力的雷达图
    
    Args:
        model_results: dict, 包含所有模型的性能数据
        selected_models: list, 选中的模型列表
    
    Returns:
        plotly figure
    """
    try:
        fig = go.Figure()
        
        # 数据集映射
        dataset_mapping = {
            'test_accuracy': 'In-Domain Test',
            'cross_device': 'Cross-Device',
            'cross_env': 'Cross-Environment',
            'cross_user': 'Cross-User'
        }
        
        metrics = list(dataset_mapping.keys())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        
        for i, model in enumerate(selected_models):
            if model in model_results:
                # 收集该模型在所有指标上的性能
                values = []
                for metric in metrics:
                    values.append(model_results[model].get(metric, 0))
            values.append(values[0])  # 闭合雷达图
            
                # 添加雷达图轨迹
            fig.add_trace(go.Scatterpolar(
                r=values,
                    theta=list(dataset_mapping.values()) + [list(dataset_mapping.values())[0]],
                fill='toself',
                name=model.upper(),
                    line_color=colors[i % len(colors)],
                    opacity=0.7,  # 设置填充透明度
                    line=dict(width=2)  # 加粗线条
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickformat='.1%',
                    tickfont=dict(color='black', size=12),  # 设置刻度标签为黑色，调整字体大小
                    gridcolor='lightgray',  # 设置网格线颜色
                    showline=True,  # 显示轴线
                    linewidth=1,  # 轴线宽度
                    linecolor='black'  # 轴线颜色
                ),
                angularaxis=dict(
                    tickfont=dict(color='black', size=12),  # 设置角度轴标签为黑色，调整字体大小
                    gridcolor='lightgray',  # 设置网格线颜色
                    showline=True,  # 显示轴线
                    linewidth=1,  # 轴线宽度
                    linecolor='black'  # 轴线颜色
                ),
                bgcolor='white',  # 设置背景色为白色
                domain=dict(x=[0, 1], y=[0.15, 0.85])  # 调整雷达图的位置，向下移动
            ),
            showlegend=True,
            title=dict(
                text="模型跨域泛化能力雷达图",
                font=dict(size=16, color='black'),  # 设置标题字体和颜色
                y=0.95  # 调整标题位置
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=12),  # 设置图例字体大小
                bgcolor='rgba(255, 255, 255, 0.8)'  # 设置图例背景色
            ),
            paper_bgcolor='white',  # 设置图表背景色
            plot_bgcolor='white',  # 设置绘图区域背景色
            margin=dict(t=120, b=50, l=50, r=50)  # 增加顶部边距
        )
        
        return fig
    except Exception as e:
        st.warning(f"创建雷达图时出错: {e}")
        return None

def show_cross_domain_analysis():
    """显示跨域泛化能力分析"""
    st.subheader("🔄 跨域泛化能力分析")
    
    # 获取模型对比数据
    comparison_data = results_loader.get_model_comparison_data()
    
    if not comparison_data:
        st.warning("❌ 没有找到模型对比数据")
        return
    
    # 创建两列用于选择
    col1, col2 = st.columns(2)
    
    with col1:
        # 模型多选
        available_models = list(comparison_data.keys())
        selected_models = st.multiselect(
            "选择要对比的模型:",
            options=available_models,
            default=available_models,  # 默认选择所有模型
            help="可以选择多个模型进行对比"
        )
    
    with col2:
        # 数据集选择
        dataset_options = {
            'test': 'In-Domain Test',
            'test_cross_device': 'Cross-Device',
            'test_cross_env': 'Cross-Environment',
            'test_cross_user': 'Cross-User'
        }
        selected_datasets = st.multiselect(
            "选择要显示的数据集:",
            options=list(dataset_options.keys()),
            default=list(dataset_options.keys()),  # 默认选择所有数据集
            format_func=lambda x: dataset_options[x],
            help="可以选择显示不同域的数据集"
        )
    
    if not selected_models:
        st.warning("请至少选择一个模型进行对比")
        return
        
    if not selected_datasets:
        st.warning("请至少选择一个数据集")
        return
    
    # 创建标签页来显示不同的可视化方式
    tab1, tab2, tab3, tab4 = st.tabs(["折线图对比", "条形图对比", "雷达图对比", "📊 性能详情"])
    
    with tab1:
        # 折线图对比
        fig_line = plot_cross_domain_comparison(
            comparison_data,
            selected_models,
            selected_datasets
        )
        if fig_line:
            st.plotly_chart(fig_line, use_container_width=True)
    
    with tab2:
        # 过滤选中的模型数据
        filtered_data = {model: data for model, data in comparison_data.items() 
                        if model in selected_models}
        
        # 条形图对比
        fig_bar = plot_model_comparison_bar(filtered_data, "模型跨域泛化能力对比")
        if fig_bar:
            st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab3:
        # 雷达图对比
        fig_radar = plot_cross_domain_radar(
            comparison_data,
            selected_models
        )
        if fig_radar:
            st.plotly_chart(fig_radar, use_container_width=True)
    
    with tab4:
        # 性能详情表格
        st.subheader("📊 性能详情")
        
        # 准备表格数据
        table_data = []
        processed_models = set()  # 用于跟踪已处理的模型
        
        for model in selected_models:
            if model in comparison_data and model not in processed_models:  # 确保每个模型只处理一次
                # 获取原始数值
                test_acc = comparison_data[model].get('test_accuracy', 0)
                cross_device = comparison_data[model].get('cross_device', 0)
                cross_env = comparison_data[model].get('cross_env', 0)
                cross_user = comparison_data[model].get('cross_user', 0)
                
                # 计算平均性能
                avg_performance = (test_acc + cross_device + cross_env + cross_user) / 4
                
                row = {
                    'Model': model.upper(),
                    'In-Domain Test': test_acc,
                    'Cross-Device': cross_device,
                    'Cross-Environment': cross_env,
                    'Cross-User': cross_user,
                    'Average': avg_performance
                }
                table_data.append(row)
                processed_models.add(model)  # 标记模型已处理
        
        # 转换为DataFrame并按平均性能排序
        df = pd.DataFrame(table_data)
        df = df.sort_values('Average', ascending=False)
        
        # 设置表格样式
        def highlight_max(s):
            is_max = s == s.max()
            return ['background-color: rgba(144, 238, 144, 0.3)' if v else '' for v in is_max]
        
        def highlight_model(s):
            return ['font-weight: bold' if i == 0 else '' for i in range(len(s))]
        
        # 自定义样式
        styles = [
            # 表头样式
            {'selector': 'th',
             'props': [
                 ('background-color', '#2c3e50'),
                 ('color', 'white'),
                 ('font-weight', 'bold'),
                 ('text-align', 'center'),
                 ('padding', '12px 8px'),
                 ('border', '1px solid #34495e'),
                 ('font-size', '14px'),
                 ('text-transform', 'uppercase'),
                 ('letter-spacing', '0.5px')
             ]},
            # 单元格样式
            {'selector': 'td',
             'props': [
                 ('text-align', 'center'),
                 ('padding', '10px 8px'),
                 ('border', '1px solid #e0e0e0'),
                 ('font-size', '13px'),
                 ('color', '#2c3e50')
             ]},
            # 表格容器样式
            {'selector': '',
             'props': [
                 ('border-collapse', 'collapse'),
                 ('border', '1px solid #e0e0e0'),
                 ('box-shadow', '0 2px 4px rgba(0,0,0,0.1)'),
                 ('border-radius', '4px'),
                 ('overflow', 'hidden')
             ]},
            # 交替行样式
            {'selector': 'tr:nth-of-type(even)',
             'props': [
                 ('background-color', '#f8f9fa')
             ]},
            # 鼠标悬停样式
            {'selector': 'tr:hover',
             'props': [
                 ('background-color', '#f5f5f5')
             ]},
            # Model列样式
            {'selector': 'td:first-child',
             'props': [
                 ('font-weight', 'bold'),
                 ('background-color', '#f8f9fa'),
                 ('border-right', '2px solid #e0e0e0')
             ]},
            # Average列样式
            {'selector': 'td:last-child',
             'props': [
                 ('font-weight', 'bold'),
                 ('background-color', '#f8f9fa'),
                 ('border-left', '2px solid #e0e0e0')
             ]}
        ]
        
        # 显示表格
        st.dataframe(
            df.style
            .format({
                'In-Domain Test': '{:.2%}',
                'Cross-Device': '{:.2%}',
                'Cross-Environment': '{:.2%}',
                'Cross-User': '{:.2%}',
                'Average': '{:.2%}'
            })
            .apply(highlight_max, subset=['In-Domain Test', 'Cross-Device', 'Cross-Environment', 'Cross-User', 'Average'])
            .apply(highlight_model, subset=['Model'])
            .set_table_styles(styles)
            .set_properties(**{
                'text-align': 'center',
                'min-width': '100px'
            }),
            use_container_width=True,
            height=400
        )
        
        # 添加表格说明
        st.markdown("""
        <div style='font-size: 13px; color: #666; margin-top: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 4px; border-left: 4px solid #2c3e50;'>
        <p style='margin: 0 0 8px 0; font-weight: bold; color: #2c3e50;'>📝 表格说明：</p>
        <ul style='margin: 0; padding-left: 20px;'>
            <li>所有性能指标均以百分比形式显示</li>
            <li>浅绿色背景表示该列中的最高性能</li>
            <li>Average 列表示模型在所有域的平均性能</li>
            <li>表格按平均性能降序排列</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # 最佳模型信息
    st.subheader("🏆 最佳模型")
    best_models = results_loader.get_best_model_info()
    
    # 显示每个域的最佳模型
    cols = st.columns(4)
    with cols[0]:
        st.metric(
            "In-Domain Test最佳模型",
            f"{best_models['test_accuracy']['model'].upper()}",
            f"{best_models['test_accuracy']['accuracy']:.1%}"
        )
    with cols[1]:
        st.metric(
            "Cross-Device最佳模型",
            f"{best_models['cross_device']['model'].upper()}",
            f"{best_models['cross_device']['accuracy']:.1%}"
        )
    with cols[2]:
        st.metric(
            "Cross-Environment最佳模型",
            f"{best_models['cross_env']['model'].upper()}",
            f"{best_models['cross_env']['accuracy']:.1%}"
        )
    with cols[3]:
        st.metric(
            "Cross-User最佳模型",
            f"{best_models['cross_user']['model'].upper()}",
            f"{best_models['cross_user']['accuracy']:.1%}"
        )

def game_mode():
    """游戏模式界面"""
    
    # 游戏状态初始化
    if 'game_started' not in st.session_state:
        st.session_state.game_started = False
    if 'user_wins' not in st.session_state:
        st.session_state.user_wins = 0
    if 'ai_wins' not in st.session_state:
        st.session_state.ai_wins = 0
    if 'target_wins' not in st.session_state:
        st.session_state.target_wins = 3
    if 'game_round' not in st.session_state:
        st.session_state.game_round = 0
    if 'game_over' not in st.session_state:
        st.session_state.game_over = False
    if 'total_score' not in st.session_state:
        st.session_state.total_score = 0
    
    # 如果游戏未开始，显示游戏设置和规则
    if not st.session_state.game_started:
        show_game_setup()
    elif st.session_state.game_over:
        show_game_over()
    else:
        show_game_interface()

def show_game_setup():
    """显示游戏设置和规则"""
    # 添加一些样式 - 整体向上移动
    st.markdown("""
    <style>
    .main > div {
        padding-top: 1rem !important;
    }
    .game-title {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(45deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
        margin-top: 2rem;
        padding-top: 1rem;
    }
    .game-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .score-badge {
        background: linear-gradient(45deg, #ff6b6b, #feca57);
        color: white;
        padding: 0.8rem;
        border-radius: 15px;
        display: block;
        margin: 0.3rem;
        font-weight: bold;
        text-align: center;
        width: 100%;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        font-size: 0.9rem;
        line-height: 1.3;
    }
    
    /* 胜利特效 */
    .victory-effect {
        animation: victoryPulse 2s ease-in-out infinite;
        background: linear-gradient(45deg, #ff6b6b, #feca57, #48dbfb, #ff9ff3);
        background-size: 400% 400%;
        animation: victoryGradient 3s ease infinite, victoryPulse 1.5s ease-in-out infinite;
    }
    
    @keyframes victoryGradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes victoryPulse {
        0% { transform: scale(1); box-shadow: 0 0 20px rgba(255, 107, 107, 0.3); }
        50% { transform: scale(1.02); box-shadow: 0 0 40px rgba(255, 107, 107, 0.6); }
        100% { transform: scale(1); box-shadow: 0 0 20px rgba(255, 107, 107, 0.3); }
    }
    
    /* 失败特效 */
    .defeat-effect {
        animation: defeatShake 0.5s ease-in-out 3;
        background: linear-gradient(45deg, #74b9ff, #0984e3, #a29bfe, #6c5ce7);
        background-size: 400% 400%;
        animation: defeatGradient 2s ease infinite, defeatShake 0.8s ease-in-out 2;
    }
    
    @keyframes defeatGradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes defeatShake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-5px); }
        75% { transform: translateX(5px); }
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="game-title">🎮 WiFi CSI Activity Recognition Challenge</div>', unsafe_allow_html=True)
    
    # 游戏介绍卡片
    with st.container():
        st.markdown("""
        <div class="game-card">
        <h3>🎯 Challenge Rules</h3>
        <p>📊 观察WiFi CSI热力图，识别人体活动类型</p>
        <p>🤖 与AI同台竞技，看谁的识别能力更强</p>
        <p>🏆 先达到目标胜利轮数的一方获胜</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 活动类型、得分规则和游戏设置 - 左中右排布
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("### 🏃‍♀️ 活动类型")
        activities_info = [
            ("🦘", "跳跃", "高动态活动，信号变化剧烈"),
            ("🏃", "跑步", "持续运动，周期性特征明显"),
            ("🧘", "静坐呼吸", "微动检测，需要高精度分析"),
            ("🚶", "走路", "中等动态，步态特征清晰"),
            ("👋", "挥手", "局部运动，手部动作识别")
        ]
        
        for emoji, name, desc in activities_info:
            st.markdown(f"**{emoji} {name}**: {desc}")
    
    with col2:
        st.markdown("### 🏆 得分规则")
        
        # 2x2布局的得分规则
        score_col1, score_col2 = st.columns(2)
        
        with score_col1:
            st.markdown("""
            <div class="score-badge">你对AI错<br>+30分</div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class="score-badge">双方都错<br>+10分</div>
            """, unsafe_allow_html=True)
        
        with score_col2:
            st.markdown("""
            <div class="score-badge">双方都对<br>+20分</div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class="score-badge">你错AI对<br>+5分</div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("### ⚙️ 游戏设置")
        target_wins = st.selectbox(
            "胜利条件：",
            [3, 5, 7, 10],
            index=0,
            help="先胜几轮获胜"
        )
        st.session_state.target_wins = target_wins
        
        # 增加一些空白间距
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🚀 开始挑战", type="primary", use_container_width=True):
            st.session_state.game_started = True
            st.session_state.user_wins = 0
            st.session_state.ai_wins = 0
            st.session_state.game_round = 1  # 从第1轮开始
            st.session_state.game_over = False
            st.session_state.total_score = 0
            start_new_round()
            st.rerun()

def show_game_interface():
    """显示游戏进行界面"""
    # 现代化的状态栏 - 页面向上移动
    st.markdown("""
    <style>
    .main > div {
        padding-top: 0.5rem !important;
    }
    .status-bar {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: white;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 0.5rem;
    }
    .game-progress {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 状态栏
    col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])
    
    with col1:
        st.metric("🏆 你的胜利", st.session_state.user_wins, 
                 delta=f"目标: {st.session_state.target_wins}")
    with col2:
        st.metric("🤖 AI胜利", st.session_state.ai_wins,
                 delta=f"目标: {st.session_state.target_wins}")
    with col3:
        st.metric("🎮 当前轮次", st.session_state.game_round)
    with col4:
        st.metric("💰 总分", st.session_state.total_score)
    with col5:
        # 计算胜率
        total_decided = st.session_state.user_wins + st.session_state.ai_wins
        win_rate = (st.session_state.user_wins / max(1, total_decided)) * 100
        st.metric("📊 胜率", f"{win_rate:.1f}%")
    with col6:
        if st.button("🚪 退出游戏", help="返回游戏设置", type="secondary"):
            reset_game()
            st.rerun()
    
    # 进度条
    progress_text = f"第 {st.session_state.game_round} 轮 | 你 {st.session_state.user_wins} : {st.session_state.ai_wins} AI"
    progress_percentage = max(st.session_state.user_wins, st.session_state.ai_wins) / st.session_state.target_wins
    st.progress(min(progress_percentage, 1.0), text=progress_text)
    
    # 游戏主体
    if 'current_sample' not in st.session_state:
        # 自动开始新轮次
        start_new_round()
        st.rerun()
    else:
        display_game_round()

def show_game_over():
    """显示游戏结束界面"""
    # 添加庆祝动画样式
    st.markdown("""
    <style>
    .winner-banner {
        background: linear-gradient(45deg, #ff6b6b, #feca57);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .loser-banner {
        background: linear-gradient(45deg, #74b9ff, #0984e3);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .final-stats {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 判断胜负并显示结果
    if st.session_state.user_wins >= st.session_state.target_wins:
        # 满屏幕气球特效
        for _ in range(3):  # 连续释放多次气球
            st.balloons()
        
        st.markdown("""
        <div class="winner-banner victory-effect">
        🎊 恭喜获胜！你战胜了AI！ 🏆<br>
        你的WiFi感知能力超越了人工智能！
        </div>
        """, unsafe_allow_html=True)
        
        # 添加额外的庆祝特效
        st.success("🌟 Amazing! You are a WiFi sensing expert! 🌟")
        
        # 再来一次气球效果
        st.balloons()  # 第二波气球
    else:
        st.markdown("""
        <div class="loser-banner defeat-effect">
        🤖 AI获胜！但你表现很棒！ 💪<br>
        继续练习，下次一定能战胜AI！
        </div>
        """, unsafe_allow_html=True)
        # 添加鼓励信息
        st.info("💪 Keep practicing! You're getting better! 💪")
    
    # 最终统计
    st.markdown("### 📊 比赛统计")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🏆 你的胜利", st.session_state.user_wins)
    with col2:
        st.metric("🤖 AI胜利", st.session_state.ai_wins)
    with col3:
        st.metric("🎮 总轮数", st.session_state.game_round)
    with col4:
        st.metric("💰 总得分", st.session_state.total_score)
    
    # 成就系统
    achievement = get_achievement_level(st.session_state.total_score)
    st.markdown(f"""
    ### 🏅 成就等级
    **{achievement['emoji']} {achievement['level']}**  
    {achievement['description']}
    """)
    
    # 再次挑战按钮
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 再次挑战", type="primary", use_container_width=True):
            reset_game()
            st.rerun()

def start_new_round():
    """开始新一轮游戏"""
    # 增加轮次计数（在实际开始时）
    if 'current_sample' not in st.session_state:
        # 只有在没有当前样本时才增加轮次
        pass  # 轮次在开始游戏时已经设置为1
    
    # 随机选择一个CSI样本
    csi_sample_dir = 'csi_samples'
    
    # 检查目录是否存在
    if not os.path.exists(csi_sample_dir):
        # 创建虚拟数据
        csi_data = create_dummy_csi_data()
        selected_file = f"dummy_sample_round_{st.session_state.game_round}.npy"
    else:
        csi_files = [f for f in os.listdir(csi_sample_dir) if f.endswith('.npy')]
        if not csi_files:
            csi_data = create_dummy_csi_data()
            selected_file = f"dummy_sample_round_{st.session_state.game_round}.npy"
        else:
            selected_file = random.choice(csi_files)
            csi_data = csi_processor.load_csi_sample(os.path.join(csi_sample_dir, selected_file))
    
    if csi_data is None:
        st.error("❌ 无法加载CSI数据")
        return
    
    # 真实标签（从文件名推断）
    activity = extract_activity_from_filename(selected_file)
    
    # 保存当前轮次状态
    st.session_state.current_sample = {
        'data': csi_data,
        'name': selected_file,
        'activity': activity
    }
    
    # 重置轮次状态
    st.session_state.user_answered = False
    if 'user_guess' in st.session_state:
        del st.session_state['user_guess']
    if 'ai_prediction' in st.session_state:
        del st.session_state['ai_prediction']

def display_game_round():
    """显示当前游戏轮次"""
    if 'current_sample' not in st.session_state:
        return
        
    try:
        current_sample = st.session_state.current_sample
        
        # 游戏区域样式 - 紧凑布局
        st.markdown("""
        <style>
        .game-area {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 1.5rem;
            border-radius: 15px;
            margin: 0.5rem 0;
        }
        .heatmap-container {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            margin-bottom: 0.5rem;
        }
        .answer-panel {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            border-left: 5px solid #007bff;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # 显示CSI热力图
        col1, col2 = st.columns([2.5, 1.5])
        
        with col1:
            st.markdown(f"""
            <div class="heatmap-container">
            <h4>📊 WiFi CSI Heatmap - Round {st.session_state.game_round}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            sample_data = current_sample['data']
            sample_name = current_sample['name']
            
            fig = plot_csi_heatmap(sample_data, f"WiFi CSI Signal Pattern Analysis")
            if fig:
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)  # 释放内存
            else:
                st.error("无法生成CSI热力图")
        
        with col2:
            st.markdown('<div class="answer-panel">', unsafe_allow_html=True)
            st.markdown("### 🤔 识别挑战")
            
            # 活动选择
            activities = ["jumping", "running", "seated-breathing", "walking", "wavinghand"]
            activity_names = {
                "jumping": "🦘 跳跃", 
                "running": "🏃 跑步", 
                "seated-breathing": "🧘 静坐呼吸",
                "walking": "🚶 走路", 
                "wavinghand": "👋 挥手"
            }
            
            # 检查是否已经作答
            if not st.session_state.get('user_answered', False):
                # 用户还未作答
                st.markdown("观察上方的CSI热力图，选择你认为的活动类型：")
                
                user_guess = st.selectbox(
                    "你的选择：",
                    activities,
                    format_func=lambda x: activity_names[x],
                    key=f"user_guess_{st.session_state.game_round}"
                )
                
                if st.button("✅ 提交答案", type="primary", use_container_width=True):
                    # 用户作答
                    st.session_state.user_guess = user_guess
                    st.session_state.user_answered = True
                    
                    # 生成AI预测
                    st.session_state.ai_prediction = get_ai_prediction(current_sample['data'])
                    st.rerun()
            
            else:
                # 用户已作答，显示结果对比
                show_round_results(activity_names)
            
            st.markdown('</div>', unsafe_allow_html=True)
                
    except Exception as e:
        st.error(f"显示游戏轮次时出错: {str(e)}")
        logger.error(f"Error in display_game_round: {str(e)}")

def get_ai_prediction(csi_data):
    """获取AI预测结果"""
    try:
        # 尝试使用真实模型进行预测
        model = model_loader.load_model('transformer')  # 使用最佳模型
        if model:
            prediction, confidence = model_loader.predict(model, csi_data)
            ai_prediction = csi_processor.get_activity_name(prediction)
        else:
            # 如果模型加载失败，使用随机预测
            activities = ["jumping", "running", "seated-breathing", "walking", "wavinghand"]
            ai_prediction = random.choice(activities)
    except Exception as e:
        # 如果预测失败，使用随机预测
        activities = ["jumping", "running", "seated-breathing", "walking", "wavinghand"]
        ai_prediction = random.choice(activities)
        logger.error(f"AI prediction failed: {str(e)}")
    
    return ai_prediction

def show_round_results(activity_names):
    """显示轮次结果"""
    st.markdown("### 📋 本轮结果")
    
    # 获取答案
    current_sample = st.session_state.current_sample
    true_activity = current_sample['activity']
    user_guess = st.session_state.user_guess
    ai_prediction = st.session_state.ai_prediction
    
    # 计算正确性
    user_correct = (user_guess == true_activity)
    ai_correct = (ai_prediction == true_activity)
    
    # 显示结果对比
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🎯 真实活动**")
        st.info(f"{activity_names[true_activity]}")
    
    with col2:
        st.markdown("**👤 你的答案**")
        if user_correct:
            st.success(f"✅ {activity_names[user_guess]}")
        else:
            st.error(f"❌ {activity_names[user_guess]}")
    
    with col3:
        st.markdown("**🤖 AI答案**")
        if ai_correct:
            st.success(f"✅ {activity_names[ai_prediction]}")
        else:
            st.error(f"❌ {activity_names[ai_prediction]}")
    
    # 计算并显示得分
    round_score = calculate_round_score(user_correct, ai_correct)
    st.session_state.total_score += round_score
    
    # 更新胜利计数和显示结果
    if user_correct and not ai_correct:
        st.success(f"🎉 你赢了这一轮！获得 {round_score} 分")
        st.session_state.user_wins += 1
    elif ai_correct and not user_correct:
        st.error(f"😞 AI赢了这一轮！获得 {round_score} 分")
        st.session_state.ai_wins += 1
    elif user_correct and ai_correct:
        st.info(f"🤝 这一轮平局！获得 {round_score} 分")
    else:
        st.warning(f"😅 你们都答错了！获得 {round_score} 分")
    
    # 检查游戏是否结束
    if (st.session_state.user_wins >= st.session_state.target_wins or 
        st.session_state.ai_wins >= st.session_state.target_wins):
        st.session_state.game_over = True
        st.markdown("---")
        if st.button("🎊 查看最终结果", type="primary", use_container_width=True):
            st.rerun()
    else:
        # 继续下一轮
        st.markdown("---")
        if st.button("➡️ 继续下一轮", type="primary", use_container_width=True):
            # 清理本轮状态并直接开始下一轮
            st.session_state.game_round += 1
            for key in ['current_sample', 'user_answered', 'user_guess', 'ai_prediction']:
                if key in st.session_state:
                    del st.session_state[key]
            # 直接开始新一轮，不需要额外点击
            start_new_round()
            st.rerun()

def calculate_round_score(user_correct, ai_correct):
    """计算轮次得分"""
    if user_correct and not ai_correct:
        return 30  # 你对AI错
    elif user_correct and ai_correct:
        return 20  # 双方都对
    elif not user_correct and not ai_correct:
        return 10  # 双方都错
    else:
        return 5   # 你错AI对

def get_achievement_level(total_score):
    """根据总分获取成就等级"""
    if total_score >= 500:
        return {
            'level': 'CSI大师',
            'emoji': '🏆',
            'description': '您已成为CSI感知领域的专家！',
        }
    elif total_score >= 300:
        return {
            'level': 'WiFi专家',
            'emoji': '🥇',
            'description': '对WiFi感知有很深的理解！',
        }
    elif total_score >= 200:
        return {
            'level': '信号猎手',
            'emoji': '🥈',
            'description': '能够识别大部分活动模式！',
        }
    elif total_score >= 50:
        return {
            'level': '数据探索者',
            'emoji': '🥉',
            'description': '开始理解CSI数据的奥秘！',
        }
    else:
        return {
            'level': '新手',
            'emoji': '🔰',
            'description': '继续探索WiFi的隐形世界！',
        }

def extract_activity_from_filename(filename):
    """从文件名提取活动标签"""
    # 根据文件命名规则解析活动类型
    filename_lower = filename.lower()
    
    # 检查各种活动类型
    if 'jumping' in filename_lower:
        return 'jumping'
    elif 'running' in filename_lower:
        return 'running'
    elif 'seated-breathing' in filename_lower:
        return 'seated-breathing'
    elif 'walking' in filename_lower:
        return 'walking'
    elif 'wavinghand' in filename_lower:
        return 'wavinghand'
    
    # 如果无法从文件名推断，随机选择一个
    return random.choice(csi_processor.activity_names)

def reset_game():
    """重置游戏"""
    # 重置游戏状态
    st.session_state.game_started = False
    st.session_state.user_wins = 0
    st.session_state.ai_wins = 0
    st.session_state.game_round = 0
    st.session_state.game_over = False
    st.session_state.target_wins = 3
    st.session_state.total_score = 0
    
    # 清理轮次状态
    keys_to_delete = ['current_sample', 'user_answered', 'user_guess', 'ai_prediction']
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]

if __name__ == "__main__":
    main() 