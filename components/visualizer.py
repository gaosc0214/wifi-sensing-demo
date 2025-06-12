import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 设置字体支持（避免字体警告）
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.font_manager')

# 彻底解决字体问题 - 使用系统默认设置
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
plt.rcParams.update(plt.rcParamsDefault)  # 重置为默认设置
plt.rcParams['axes.unicode_minus'] = False

def plot_csi_heatmap(csi_data, title="CSI Data Heatmap"):
    """
    Plot CSI data as a heatmap
    
    Args:
        csi_data: CSI data array (time_steps, features)
        title: plot title
        
    Returns:
        matplotlib figure
    """
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create heatmap
        im = ax.imshow(csi_data.T, 
                      aspect='auto', 
                      cmap='viridis',
                      interpolation='nearest')
        
        # Set labels and title with Chinese support
        ax.set_xlabel('Time Steps', fontsize=12)
        ax.set_ylabel('Subcarrier Index', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('CSI Amplitude', fontsize=12)
        
        # Improve layout
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        print(f"Error creating heatmap: {e}")
        # Return a simple dummy plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, f'Unable to display CSI data\n{str(e)}', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return fig

def plot_csi_time_series(csi_data, subcarrier_indices=None, title="CSI Time Series"):
    """
    Plot CSI data as time series for selected subcarriers
    
    Args:
        csi_data: CSI data array (time_steps, features)
        subcarrier_indices: list of subcarrier indices to plot
        title: plot title
        
    Returns:
        matplotlib figure
    """
    try:
        if subcarrier_indices is None:
            # Select a few representative subcarriers
            step = max(1, csi_data.shape[1] // 5)
            subcarrier_indices = list(range(0, csi_data.shape[1], step))[:5]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        time_steps = np.arange(csi_data.shape[0])
        
        for idx in subcarrier_indices:
            if idx < csi_data.shape[1]:
                ax.plot(time_steps, csi_data[:, idx], 
                       label=f'Subcarrier {idx}', alpha=0.8)
        
        ax.set_xlabel('Time Steps', fontsize=12)
        ax.set_ylabel('CSI Amplitude', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"Error creating time series plot: {e}")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, f'Unable to display time series\n{str(e)}', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return fig

def plot_activity_comparison(csi_data_dict, title="CSI Comparison for Different Activities"):
    """
    Plot comparison of CSI data for different activities
    
    Args:
        csi_data_dict: dictionary with activity names as keys and CSI data as values
        title: plot title
        
    Returns:
        matplotlib figure
    """
    try:
        n_activities = len(csi_data_dict)
        fig, axes = plt.subplots(n_activities, 1, figsize=(12, 3*n_activities))
        
        if n_activities == 1:
            axes = [axes]
        
        for idx, (activity, csi_data) in enumerate(csi_data_dict.items()):
            ax = axes[idx]
            
            # Plot heatmap for this activity
            im = ax.imshow(csi_data.T, 
                          aspect='auto', 
                          cmap='viridis',
                          interpolation='nearest')
            
            ax.set_title(f'{activity}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Subcarrier Index')
            
            # Add colorbar
            plt.colorbar(im, ax=ax)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        print(f"Error creating activity comparison plot: {e}")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, f'Unable to display activity comparison\n{str(e)}', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return fig

def plot_performance_radar(performance_data, title="Model Performance Radar Chart"):
    """
    Create a radar chart for model performance
    
    Args:
        performance_data: dictionary with metrics and values
        title: plot title
        
    Returns:
        plotly figure
    """
    try:
        metrics = list(performance_data.keys())
        values = list(performance_data.values())
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics,
            fill='toself',
            name='Performance Metrics'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title=title
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating radar chart: {e}")
        # Return simple bar chart as fallback
        fig = px.bar(x=list(performance_data.keys()), 
                    y=list(performance_data.values()),
                    title=title)
        return fig

def plot_confusion_matrix_interactive(cm, class_names, title="Interactive Confusion Matrix"):
    """
    Create an interactive confusion matrix
    
    Args:
        cm: confusion matrix array
        class_names: list of class names
        title: plot title
        
    Returns:
        plotly figure
    """
    try:
        fig = px.imshow(cm, 
                       x=class_names, 
                       y=class_names,
                       color_continuous_scale='Blues',
                       title=title)
        
        fig.update_layout(
            xaxis_title="Predicted Class",
            yaxis_title="True Class"
        )
        
        # Add text annotations
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                fig.add_annotation(
                    x=j, y=i,
                    text=str(cm[i, j]),
                    showarrow=False,
                    font=dict(color="white" if cm[i, j] > cm.max()/2 else "black")
                )
        
        return fig
        
    except Exception as e:
        print(f"Error creating interactive confusion matrix: {e}")
        # Return simple heatmap as fallback
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(title)
        return fig

def create_3d_csi_plot(csi_data, title="3D CSI Visualization"):
    """
    Create 3D visualization of CSI data
    
    Args:
        csi_data: CSI data array (time_steps, features)
        title: plot title
        
    Returns:
        plotly figure
    """
    try:
        # Sample data for 3D plot (to avoid performance issues)
        time_step = 10
        feature_step = 5
        
        time_indices = np.arange(0, csi_data.shape[0], time_step)
        feature_indices = np.arange(0, csi_data.shape[1], feature_step)
        
        # Create meshgrid
        T, F = np.meshgrid(time_indices, feature_indices)
        Z = csi_data[::time_step, ::feature_step].T
        
        fig = go.Figure(data=[go.Surface(z=Z, x=T, y=F, colorscale='Viridis')])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Time Steps',
                yaxis_title='Subcarrier Index',
                zaxis_title='CSI Amplitude'
            )
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating 3D plot: {e}")
        # Return 2D heatmap as fallback
        return plot_csi_heatmap(csi_data, title)

def plot_prediction_confidence(predictions, confidence_scores, class_names, title="Prediction Confidence Analysis"):
    """
    Plot prediction confidence distribution
    
    Args:
        predictions: list of prediction indices
        confidence_scores: list of confidence scores
        class_names: list of class names
        title: plot title
        
    Returns:
        plotly figure
    """
    try:
        # Create DataFrame for easier plotting
        df = pd.DataFrame({
            'Activity': [class_names[p] for p in predictions],
            'Confidence': confidence_scores
        })
        
        fig = px.box(df, x='Activity', y='Confidence', title=title)
        fig.update_layout(
            xaxis_title="Activity Category",
            yaxis_title="Confidence"
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating confidence plot: {e}")
        # Return simple bar chart as fallback
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(range(len(confidence_scores)), confidence_scores)
        ax.set_title(title)
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Confidence")
        return fig

def plot_model_comparison_bar(model_results, title="Model Performance Comparison"):
    """
    Create bar chart comparing different models
    
    Args:
        model_results: dict with model names as keys and performance dict as values
        title: plot title
        
    Returns:
        plotly figure
    """
    try:
        models = list(model_results.keys())
        metrics = ['test_accuracy', 'cross_device', 'cross_env', 'cross_user']
        
        data = []
        for metric in metrics:
            values = []
            for model in models:
                if metric in model_results[model]:
                    values.append(model_results[model][metric])
                else:
                    values.append(0)
            data.append(values)
        
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        metric_names = ['In-Domain Test', 'Cross-Device', 'Cross-Environment', 'Cross-User']
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            fig.add_trace(go.Bar(
                name=metric_name,
                x=models,
                y=data[i],
                marker_color=colors[i % len(colors)]
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Model",
            yaxis_title="Accuracy",
            barmode='group'
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating model comparison: {e}")
        return None 