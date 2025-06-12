# WiFi感知平台 - 部署版本

这是WiFi感知平台的部署版本，包含了运行应用所需的最小文件集。

## 目录结构

```
deploy/
├── app.py              # 主应用程序
├── requirements.txt    # Python依赖
├── render.yaml         # Render部署配置
├── models/            # 模型文件
│   └── best_model.pt  # 预训练模型
├── data/              # 演示数据
│   └── *.npy         # CSI样本数据
├── components/        # UI组件
├── utils/            # 工具函数
├── config/           # 配置文件
├── assets/           # 静态资源
└── .streamlit/       # Streamlit配置
    └── config.toml
```

## 部署说明

1. 此版本已优化，移除了所有不必要的文件
2. 总大小应控制在500MB以内，符合Render免费版限制
3. 包含了运行应用所需的所有核心组件

## 运行应用

```bash
streamlit run app.py
```

## 注意事项

- 此版本仅包含演示所需的最小数据集
- 模型文件已优化，确保部署性能
- 所有配置文件已针对生产环境优化 