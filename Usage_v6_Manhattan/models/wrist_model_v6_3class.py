import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1, attention_dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(attention_dropout)
        self.out = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        batch_size, seq_length, hidden_size = x.size()
        
        q = self.q_linear(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = self.k_linear(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        v = self.v_linear(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, hidden_size)
        output = self.out(context)
        
        return output, attention_weights

class EnhancedFeatureBlock(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(EnhancedFeatureBlock, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.norm1(x + residual)
        return x

class FeatureExpansion(nn.Module):
    def __init__(self, input_size, expansion_factor, dropout=0.1):
        super(FeatureExpansion, self).__init__()
        self.expanded_size = input_size * expansion_factor
        
        self.expand = nn.Sequential(
            nn.Linear(input_size, self.expanded_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.expanded_size, input_size),
            nn.LayerNorm(input_size)
        )
        
    def forward(self, x):
        return self.expand(x) + x

class WristLSTM(nn.Module):
    def __init__(self, input_size=33, hidden_size=100, num_layers=2, dropout=0.5, 
                 num_heads=4, attention_dropout=0.1, feature_expansion=4, num_classes=3):
        super(WristLSTM, self).__init__()
        
        # 特征扩展层
        self.feature_expansion = FeatureExpansion(
            input_size=input_size,
            expansion_factor=feature_expansion,
            dropout=dropout
        )
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(hidden_size)
        
        # 输入投影
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # LSTM层
        self.lstm = nn.LSTM(hidden_size,
                           hidden_size,
                           num_layers=num_layers,
                           batch_first=True,
                           dropout=dropout if num_layers > 1 else 0,
                           bidirectional=True)
        
        # 调整LSTM输出维度
        self.lstm_proj = nn.Linear(hidden_size * 2, hidden_size)
        
        # 多头注意力
        self.attention = MultiHeadAttention(
            hidden_size, 
            num_heads=num_heads, 
            dropout=dropout,
            attention_dropout=attention_dropout
        )
        
        # 特征增强
        self.feature_enhancement = EnhancedFeatureBlock(hidden_size, dropout)
        
        # 修改attention_weights的初始化和存储
        self.stored_attention_weights = None
        self.head_dim = hidden_size // num_heads
        
        # 初始化类别权重 (3类)
        self.class_weights = nn.Parameter(torch.FloatTensor([
            1.5,  # Baseline (原label 1) - 中等权重
            2.0,  # Stress (原label 2) - 较高权重，因为这是关键类别
            1.0   # Amusement/Meditation (原label 3,4) - 基准权重
        ]))
        
        # 添加注意力权重调整层
        self.head_weights = nn.Parameter(torch.ones(num_heads))
        self.attention_norm = nn.LayerNorm(hidden_size)
        
        # 分类层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.LayerNorm(hidden_size//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.LayerNorm(hidden_size//4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//4, num_classes)  # 输出改为3类
        )
    
    def forward(self, x):
        batch_size, seq_len, feat_dim = x.size()
        assert feat_dim == 33, f"Expected 33 features, got {feat_dim}"
        
        # 特征扩展
        x = self.feature_expansion(x)
        
        # 输入投影和位置编码
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        
        # LSTM处理
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_proj(lstm_out)
        
        # 多头注意力with权重
        context, attention_weights = self.attention(lstm_out)
        weighted_attention = torch.zeros_like(context)
        
        # 存储attention_weights用于后续分析
        self.stored_attention_weights = attention_weights.detach()
        
        # 应用head-level权重
        for i in range(self.attention.num_heads):
            head_context = context[:, :, i * self.head_dim:(i + 1) * self.head_dim]
            weighted_attention[:, :, i * self.head_dim:(i + 1) * self.head_dim] = \
                head_context * F.softplus(self.head_weights[i])
        
        context = self.attention_norm(weighted_attention)
        
        # 特征增强
        enhanced_features = self.feature_enhancement(context)
        
        # 全局上下文向量
        weights = F.softmax(self.stored_attention_weights.mean(1), dim=-1)
        global_context = torch.bmm(weights, enhanced_features)
        global_context = global_context.squeeze(1)
        
        # 分类
        output = self.fc(global_context)
        
        # 应用类别权重
        weighted_output = output * F.softplus(self.class_weights)
        
        return F.log_softmax(weighted_output, dim=1) 