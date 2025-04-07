# Urban Emotion Recognition System (UERS) Brooklyn Study
# 城市情感识别系统 / Urban Emotion Recognition System

[English Version](#english-version) | [中文版本](#chinese-version)

# English Version

## 🌟 Introduction: Why UERS?

In our modern urban life, emotions play a crucial role in our daily experiences. Imagine walking through a busy street, relaxing in a park, or navigating a crowded shopping mall - each environment affects our emotional state differently. But how can we measure and understand these emotional changes scientifically?

This is where the Urban Emotion Recognition System (UERS) comes in. It's an innovative system that combines:
- Wearable device data (physiological signals)
- GPS location information
- Advanced deep learning algorithms

to help us understand how different urban environments affect our emotional states.

### 🎯 What Problems Does It Solve?

1. **Urban Planning Challenges**
   - Traditional methods rely on surveys and interviews
   - Subjective and time-consuming
   - Limited real-time data

2. **Personal Well-being**
   - Difficulty in identifying stress triggers
   - Lack of objective emotional state measurement
   - No spatial-temporal emotional patterns

3. **Research Gaps**
   - Limited integration of physiological and location data
   - Few real-world emotion recognition systems
   - Need for non-invasive emotion monitoring

## 🔬 The Science Behind UERS

### Understanding Our Emotions Through Biology

When we experience different emotions, our body responds in measurable ways:

1. **Sweating (EDA - Electrodermal Activity)**
   - What it is: Changes in skin's electrical conductance
   - Why it matters: Indicates stress and emotional arousal
   - How we measure it: Microsensors on the wrist

2. **Heart Rate (BVP - Blood Volume Pulse)**
   - What it is: Blood flow changes in blood vessels
   - Why it matters: Reflects emotional intensity
   - How we measure it: Optical sensors

3. **Body Temperature (TEMP)**
   - What it is: Skin temperature variations
   - Why it matters: Shows autonomic nervous system activity
   - How we measure it: Temperature sensors

4. **Movement (ACC - Accelerometer)**
   - What it is: Physical activity patterns
   - Why it matters: Indicates restlessness or calmness
   - How we measure it: 3-axis accelerometer

### The Deep Learning Model

Our model uses a sophisticated architecture with attention mechanisms and residual connections:

```
Input (33-dim) → Feature Expansion (4x) → Embedding (132→100) → Positional Encoding
       ↓
    BiLSTM (multiple LSTM cells in both directions)
       ↓
LSTM Projection (200→100) → Multi-Head Attention (4 heads with QKV)
       ↓
Head Weights & Attention Norm → Feature Enhanced FNN with Residual Connections
       ↓
Global Context Vector → Full Connect Layers & Classifier (100→50→25→3)
       ↓
  Output (3-dim)
```

Key Features:
- BiLSTM for capturing temporal dependencies
- Multi-head self-attention mechanism for important pattern focus
- Residual connections for gradient flow
- Positional encoding for sequence awareness
- Layer normalization for training stability

## 🛠️ System Requirements

### Hardware
- Empatica E4 wearable device
- GPS device (smartphone works)
- Computer for analysis

### Software
- Python 3.8+
- CUDA support (for GPU training)
- Required Python packages (see Installation)

## 📊 Data Types and Classes

### Emotion Categories

1. **3-Class Model**
   - Baseline (平静状态/基线)
   - Stress (压力状态)
   - Amusement/Meditation (愉悦/冥想状态)

2. **5-Class Model**
   - Not Defined/Transient (未定义/转移态)
   - Baseline (平静状态/基线)
   - Stress (压力状态)
   - Amusement/Meditation (娱乐冥想状态)
   - Corrupted/Other (损坏/其他状态)

## 🚀 Getting Started

### 1. Installation and Setup

```bash
# Clone the repository
git clone https://github.com/XueliangYang/UrbanEmotionSystem.git

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Collection
1. Wear the Empatica E4 device
2. Ensure GPS tracking is active
3. Record activities and times

### 3. Data Processing and Analysis

Follow these steps in order:

```bash
# 1. Process individual participant data
python process_participant_data.py

# 2. Combine all participant data
python merge_participant_data.py

# 3. Process and predict in one step
python process_and_predict.py

# 4. Finalize emotion labels
python post_label_process.py

# 5. Align emotions with environmental data
python match_labels_with_env_gps.py

# 6. Analyze environmental impacts
python analyze_env_features.py
```

## 📁 Project and Data Structure

### Main Project Structure
```
Usage_v6_Brooklyn/
├── scripts/                # Main scripts
│   ├── process_participant_data.py     # Process individual participant data
│   ├── merge_participant_data.py       # Combine all participant data
│   ├── process_and_predict.py          # Process and predict in one step
│   ├── post_label_process.py           # Finalize emotion labels
│   ├── match_labels_with_env_gps.py    # Align emotions with environmental data
│   └── analyze_env_features.py         # Analyze environmental impacts
├── models/                 # Trained models
├── logs/                   # Log files
└── env_feature_analysis/   # Environmental analysis results
```

### Data Structure (External)
```
E:\NYU\academic\Y2S2\CUSP-GX-7133\DataBrooklyn\
├── ParticipantData\        # Raw participant data
│   ├── Participant#0001\   # Individual participant folders
│   │   └── raw_data\       # Raw data files from Empatica device
│   ├── Participant#0002\
│   └── ...
└── All_Participant_Process\ # Processed data files
    └── Bkln_Participant_Labeled_BiLSTM_withEnvGPS.csv # Main processed dataset combined with Bio, Env and GPS data
```

## 🎨 Visualization Examples

- Red points: Stress states
- Green points: Amusement/Meditation states
- Blue points: Baseline states

## 🔍 Technical Details

### Model Architecture

Our model uses a sophisticated architecture with attention mechanisms and residual connections:

```
Input (33-dim) → Feature Expansion (4x) → Embedding (132→100) → Positional Encoding
       ↓
    BiLSTM (multiple LSTM cells in both directions)
       ↓
LSTM Projection (200→100) → Multi-Head Attention (4 heads with QKV)
       ↓
Head Weights & Attention Norm → Feature Enhanced FNN with Residual Connections
       ↓
Global Context Vector → Full Connect Layers & Classifier (100→50→25→3)
       ↓
  Output (3-dim)
```

```python
class WristLSTM(nn.Module):
    def __init__(self, input_dim=33, embedding_dim=100, hidden_dim=100, output_dim=3, n_heads=4):
        super(WristLSTM, self).__init__()
        
        # Feature expansion and embedding
        self.feature_expansion = nn.Linear(input_dim, input_dim*4)
        self.embedding = nn.Linear(input_dim*4, embedding_dim)
        self.pos_encoding = PositionalEncoding(embedding_dim)
        
        # BiLSTM layers
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.lstm_projection = nn.Linear(hidden_dim*2, hidden_dim)
        
        # Multi-head attention
        self.multihead_attn = MultiHeadAttention(hidden_dim, n_heads)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        
        # Feature Enhanced FNN with residual connections
        self.linear1 = nn.Linear(hidden_dim, hidden_dim*4)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim*4, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, output_dim)
        )
```

### Key Features
- BiLSTM for capturing temporal dependencies
- Multi-head self-attention mechanism for important pattern focus
- Residual connections for gradient flow
- Positional encoding for sequence awareness
- Layer normalization for training stability

# Chinese Version / 中文版本

## 🌟 引言：为什么需要城市情感识别系统？

在现代城市生活中，情感在我们的日常体验中扮演着重要角色。想象一下，当你走在繁忙的街道上，在公园里放松，或在拥挤的商场中穿行时，每个环境都会以不同的方式影响我们的情绪状态。但我们如何能够科学地测量和理解这些情绪变化呢？

这就是城市情感识别系统（UERS）的用武之地。它是一个创新系统，结合了：
- 可穿戴设备数据（生理信号）
- GPS位置信息
- 先进的深度学习算法

来帮助我们理解不同城市环境如何影响我们的情绪状态。

### 🎯 它解决了什么问题？

1. **城市规划挑战**
   - 传统方法依赖调查和访谈
   - 主观且耗时
   - 实时数据有限

2. **个人健康**
   - 难以识别压力触发因素
   - 缺乏客观的情绪状态测量
   - 没有时空情绪模式

3. **研究空白**
   - 生理和位置数据的集成有限
   - 实际情绪识别系统较少
   - 需要非侵入式情绪监测

## 🔬 城市情感识别系统背后的科学原理

### 通过生物学理解我们的情绪

当我们经历不同情绪时，我们的身体会产生可测量的反应：

1. **出汗（EDA - 皮肤电活动）**
   - 是什么：皮肤电导率的变化
   - 为什么重要：表明压力和情绪唤醒
   - 如何测量：手腕上的微型传感器

2. **心率（BVP - 血容量脉搏）**
   - 是什么：血管中血流变化
   - 为什么重要：反映情绪强度
   - 如何测量：光学传感器

3. **体温（TEMP）**
   - 是什么：皮肤温度变化
   - 为什么重要：显示自主神经系统活动
   - 如何测量：温度传感器

4. **运动（ACC - 加速度计）**
   - 是什么：身体活动模式
   - 为什么重要：表明躁动或平静
   - 如何测量：三轴加速度计

### 深度学习模型

我们的模型使用了复杂的架构，包含注意力机制和残差连接：

```
输入 (33维) → 特征扩展 (4倍) → 嵌入层 (132→100) → 位置编码
      ↓
   双向LSTM (双向多个LSTM单元)
      ↓
LSTM投影 (200→100) → 多头注意力 (4个头部，QKV注意力)
      ↓
头部权重与注意力规范化 → 特征增强前馈网络（含残差连接）
      ↓
全局上下文向量 → 全连接层与分类器 (100→50→25→3)
      ↓
   输出 (3维)
```

```python
class WristLSTM(nn.Module):
    def __init__(self, input_dim=33, embedding_dim=100, hidden_dim=100, output_dim=3, n_heads=4):
        super(WristLSTM, self).__init__()
        
        # 特征扩展和嵌入
        self.feature_expansion = nn.Linear(input_dim, input_dim*4)
        self.embedding = nn.Linear(input_dim*4, embedding_dim)
        self.pos_encoding = PositionalEncoding(embedding_dim)
        
        # 双向LSTM层
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.lstm_projection = nn.Linear(hidden_dim*2, hidden_dim)
        
        # 多头注意力
        self.multihead_attn = MultiHeadAttention(hidden_dim, n_heads)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        
        # 特征增强前馈网络与残差连接
        self.linear1 = nn.Linear(hidden_dim, hidden_dim*4)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim*4, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, output_dim)
        )
```

### 核心特点
- 双向LSTM捕获时序依赖关系
- 多头自注意力机制关注重要模式
- 残差连接优化梯度流
- 位置编码提供序列感知能力
- 层归一化增强训练稳定性

## 🛠️ 系统要求

### 硬件
- Empatica E4可穿戴设备
- GPS设备（智能手机即可）
- 用于分析的计算机

### 软件
- Python 3.8+
- CUDA支持（用于GPU训练）
- 必要的Python包（见安装说明）

## 📊 数据类型和类别

### 情感类别

1. **三分类模型**
   - 基线（平静状态）
   - 压力（压力状态）
   - 愉悦/冥想（愉悦/放松状态）

2. **五分类模型**
   - Not Defined/Transient (未定义/转移态)
   - Baseline (平静状态/基线)
   - Stress (压力状态)
   - Amusement/Meditation (娱乐/冥想状态)
   - Corrupted/Other (损坏/其他状态)
   
## 🚀 开始使用

### 1. 安装和设置

```bash
# 克隆仓库
git clone https://github.com/XueliangYang/UrbanEmotionSystem.git

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据收集
1. 佩戴Empatica E4设备
2. 确保GPS追踪激活
3. 记录活动和时间

### 3. 数据处理和分析

按照以下步骤顺序执行：

```bash
# 1. 处理单个参与者数据
python process_participant_data.py

# 2. 合并所有参与者数据
python merge_participant_data.py

# 3. 一步完成处理和预测
python process_and_predict.py

# 4. 完成情绪标签处理
python post_label_process.py

# 5. 将情绪与环境数据对齐
python match_labels_with_env_gps.py

# 6. 分析环境影响
python analyze_env_features.py
```

## 📁 项目和数据结构

### 主要项目结构
```
Usage_v6_Brooklyn/
├── scripts/                # 主要脚本
│   ├── process_participant_data.py     # 处理单个参与者数据
│   ├── merge_participant_data.py       # 合并所有参与者数据
│   ├── process_and_predict.py          # 一步完成处理和预测
│   ├── post_label_process.py           # 完成情绪标签处理
│   ├── match_labels_with_env_gps.py    # 将情绪与环境数据对齐
│   └── analyze_env_features.py         # 分析环境影响
├── models/                 # 训练好的模型
├── logs/                   # 日志文件
└── env_feature_analysis/   # 环境分析结果
```

### 数据结构（外部）
```
E:\NYU\academic\Y2S2\CUSP-GX-7133\DataBrooklyn\
├── ParticipantData\        # 原始参与者数据
│   ├── Participant#0001\   # 个人参与者文件夹
│   │   └── raw_data\       # 来自E4设备的原始数据文件
│   ├── Participant#0002\
│   └── ...
└── All_Participant_Process\ # 处理后的数据文件
    └── Bkln_Participant_Labeled_BiLSTM_withEnvGPS.csv # 主要处理后的合并了生理,环境和GPS的数据集
```

## 🎨 可视化示例

- 红点：压力状态
- 绿点：愉悦/冥想状态
- 蓝点：基线状态

## 🔍 技术细节

### 模型架构

我们的模型使用了复杂的架构，包含注意力机制和残差连接：

```
输入 (33维) → 特征扩展 (4倍) → 嵌入层 (132→100) → 位置编码
      ↓
   双向LSTM (双向多个LSTM单元)
      ↓
LSTM投影 (200→100) → 多头注意力 (4个头部，QKV注意力)
      ↓
头部权重与注意力规范化 → 特征增强前馈网络（含残差连接）
      ↓
全局上下文向量 → 全连接层与分类器 (100→50→25→3)
      ↓
   输出 (3维)
```

```python
class WristLSTM(nn.Module):
    def __init__(self, input_dim=33, embedding_dim=100, hidden_dim=100, output_dim=3, n_heads=4):
        super(WristLSTM, self).__init__()
        
        # 特征扩展和嵌入
        self.feature_expansion = nn.Linear(input_dim, input_dim*4)
        self.embedding = nn.Linear(input_dim*4, embedding_dim)
        self.pos_encoding = PositionalEncoding(embedding_dim)
        
        # 双向LSTM层
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.lstm_projection = nn.Linear(hidden_dim*2, hidden_dim)
        
        # 多头注意力
        self.multihead_attn = MultiHeadAttention(hidden_dim, n_heads)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        
        # 特征增强前馈网络与残差连接
        self.linear1 = nn.Linear(hidden_dim, hidden_dim*4)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim*4, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, output_dim)
        )
```

### 核心特点
- 双向LSTM捕获时序依赖关系
- 多头自注意力机制关注重要模式
- 残差连接优化梯度流
- 位置编码提供序列感知能力
- 层归一化增强训练稳定性

## 📞 支持与帮助

如果遇到问题：
1. 检查日志文件（logs目录）
2. 确认数据格式正确
3. 验证设备连接状态

## 📝 注意事项

1. **数据安全**
   - 定期备份数据
   - 保护个人隐私信息
   - 安全存储原始数据

2. **使用限制**
   - 不适用于医疗诊断
   - 结果仅供参考
   - 考虑个体差异

## 🌟 致谢

感谢所有参与本项目的研究者、志愿者和支持者。您的贡献使这个项目成为可能。

## 📚 参考资料

- Empatica E4设备文档
- 相关研究论文
- 技术支持文档

---

*This project is under continuous improvement. Your feedback and suggestions are welcome.*
*本项目仍在持续改进中，欢迎提供反馈和建议。* 

---------------------------------------------------------------------------------------------------------------------------------------------------------


# Urban Emotion Recognition System (UERS) Brooklyn Study

This directory contains code and resources for the Brooklyn study using the Urban Emotion Recognition System.

## Directory Structure

### Project Structure
- **scripts/**: Contains scripts for data processing, prediction, and analysis.
- **models/**: Contains trained models for emotion recognition.
- **logs/**: Contains log files from running predictions and analyses.
- **env_feature_analysis/**: Contains the output of environmental feature analysis including plots and summary statistics.

### Data Structure (External)
```
E:\NYU\academic\Y2S2\CUSP-GX-7133\DataBrooklyn\
├── ParticipantData\        # Raw participant data
│   ├── Participant#0001\   # Individual participant folders
│   │   └── raw_data\       # Raw data files from E4 device
│   ├── Participant#0002\
│   └── ...
└── All_Participant_Process\ # Processed data files
    └── Bkln_Participant_Labeled_BiLSTM_withEnvGPS.csv # Main processed dataset
```

## Key Scripts

- **process_participant_data.py**: Processes individual participant folders and extracts relevant data.
- **merge_participant_data.py**: Combines data from all participants into a single dataset.
- **process_and_predict.py**: Processes raw data and then runs predictions.
- **post_label_process.py**: Finalizes the labeling of the data.
- **match_labels_with_env_gps.py**: Aligns emotion labels with environmental and GPS data based on timestamps.
- **analyze_env_features.py**: Analyzes relationships between environmental features and emotional states.

## Usage

1. **Process Participant Data**: Run `process_participant_data.py` to process individual participant folders and extract relevant data.
2. **Merge Participant Data**: Execute `merge_participant_data.py` to combine data from all participants into a single dataset.
3. **Process and Predict**: Use `process_and_predict.py` to process the merged data and run predictions.
4. **Post-Label Processing**: Run `post_label_process.py` to finalize the labeling of the data.
5. **Match Labels with Environmental GPS Data**: Execute `match_labels_with_env_gps.py` to align emotion labels with environmental and GPS data based on timestamps.
6. **Analyze Environmental Features**: Finally, run `analyze_env_features.py` to analyze the relationship between environmental factors and emotional states.

## Environmental Feature Analysis

The environmental feature analysis scripts analyze the relationship between environmental factors (such as air quality, temperature, humidity) and emotional states. Results are stored in the `env_feature_analysis/` directory. 