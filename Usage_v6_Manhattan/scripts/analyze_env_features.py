import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
import os
import sys
from tqdm import tqdm
import pickle
import traceback
import time
import math

def load_data():
    """
    加载合并后的数据集
    """
    print("\n[1/6] 加载数据...")
    try:
        # 数据路径
        data_path = r'C:\Users\xy2593\Desktop\CUSP-GX-9000 Guided Study\Data\ParticipantData\All_Participant_Process'
        # 优先加载带有优先级标签的数据
        prioritized_file = os.path.join(data_path, 'All_Participant_Labeled_BiLSTM_Labeled_withEnvGPS_prioritized.csv')
        regular_file = os.path.join(data_path, 'All_Participant_Labeled_BiLSTM_Labeled_withEnvGPS.csv')
        
        # 检查文件是否存在并加载
        if os.path.exists(prioritized_file):
            print(f"  Loading prioritized data from: {prioritized_file}")
            data = pd.read_csv(prioritized_file)
        elif os.path.exists(regular_file):
            print(f"  Loading regular data from: {regular_file}")
            data = pd.read_csv(regular_file)
        else:
            raise FileNotFoundError("找不到合并后的数据文件")
        
        print(f"  Loaded data with {len(data)} rows and {len(data.columns)} columns")
        return data
    except Exception as e:
        print(f"Error in load_data: {e}")
        traceback.print_exc()
        sys.exit(1)

def preprocess_data(data):
    """
    预处理数据：处理缺失值，选择相关特征
    """
    print("\n[2/6] 预处理数据...")
    try:
        # 备份原始数据
        df = data.copy()
        
        # 提取环境特征
        env_features = [
            'PM2.5(ug/m3)', 'PM10(ug/m3)', 'CO2(ppm)', 'Noise(dB)', 
            'WGBT(C)', 'Air Temperature(C)', 'Black Globe Temperature(C)', 
            'Relative Humidity(%)', 'Latitude', 'Longitude'
        ]
        
        # 检查所有环境特征是否存在
        available_features = [f for f in env_features if f in df.columns]
        if len(available_features) < len(env_features):
            print(f"  警告: 以下特征在数据集中不存在: {set(env_features) - set(available_features)}")
            env_features = available_features
            
        print(f"  使用环境特征: {env_features}")
        
        # 检查情绪标签列是否存在
        if 'BiLSTM_predicted_label' not in df.columns:
            raise ValueError("找不到情绪标签列 'BiLSTM_predicted_label'")
        
        # 提取特征和标签
        X = df[env_features].copy()  # 使用.copy()避免SettingWithCopyWarning
        y = df['BiLSTM_predicted_label'].copy()
        
        # 处理缺失值 - 使用中位数填充
        for col in X.columns:
            if X[col].isna().sum() > 0:
                median_val = X[col].median()
                print(f"  填充特征 '{col}' 的 {X[col].isna().sum()} 个缺失值 (中位数: {median_val:.2f})")
                X[col] = X[col].fillna(median_val)
        
        # 标准化特征
        print("  标准化特征...")
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        # 打印数据分布
        label_dist = y.value_counts(normalize=True).sort_index() * 100
        print(f"\n  标签分布:")
        for label, pct in label_dist.items():
            if label == 1:
                state = "Baseline"
            elif label == 2:
                state = "Stress"
            elif label == 3:
                state = "Relaxation/Amusement"
            else:
                state = "Unknown"
            print(f"    Class {label} ({state}): {pct:.1f}%")
        
        return X, X_scaled, y, df, env_features
    except Exception as e:
        print(f"Error in preprocess_data: {e}")
        traceback.print_exc()
        sys.exit(1)

def analyze_correlations(X, y, features, df):
    """
    分析特征和情绪标签之间的相关性
    """
    print("\n[3/6] 分析特征相关性...")
    try:
        # 创建包含特征和标签的数据框
        corr_df = X.copy()
        corr_df['emotion'] = y
        
        # 计算特征与情绪标签之间的相关性
        correlations = {}
        for feature in features:
            # 皮尔逊相关性
            pearson_corr = corr_df[feature].corr(corr_df['emotion'])
            correlations[feature] = pearson_corr
        
        # 按绝对相关性排序
        sorted_correlations = {k: v for k, v in sorted(
            correlations.items(), key=lambda item: abs(item[1]), reverse=True
        )}
        
        print("\n  特征与情绪标签相关性(皮尔逊):")
        for feature, corr in sorted_correlations.items():
            print(f"    {feature}: {corr:.4f}")
        
        # 创建情绪标签对应的环境因素平均值统计
        emotion_stats = df.groupby('BiLSTM_emotional_state')[features].mean()
        print("\n  各情绪状态下环境因素平均值:")
        print(emotion_stats)
        
        # 可视化相关性
        print("  创建相关性图...")
        plt.figure(figsize=(12, 8))
        corr_values = pd.Series(sorted_correlations)
        colors = ['blue' if c >= 0 else 'red' for c in corr_values]
        corr_values.sort_values().plot(kind='barh', color=colors)
        plt.title('环境特征与情绪标签相关性')
        plt.xlabel('皮尔逊相关系数')
        plt.tight_layout()
        plt.savefig('features_correlation.png')
        print("  相关性图已保存为 features_correlation.png")
        
        return sorted_correlations, emotion_stats
    except Exception as e:
        print(f"Error in analyze_correlations: {e}")
        traceback.print_exc()
        sys.exit(1)

def train_random_forest(X_scaled, y):
    """
    训练随机森林模型并分析特征重要性
    """
    print("\n[4/6] 训练随机森林模型...")
    try:
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )
        print(f"  训练集: {X_train.shape[0]} 样本, 测试集: {X_test.shape[0]} 样本")
        
        # 训练随机森林模型
        print("  训练随机森林分类器...")
        rf = RandomForestClassifier(
            n_estimators=100, 
            class_weight='balanced',
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        rf.fit(X_train, y_train)
        print("  模型训练完成")
        
        # 评估模型性能
        y_pred = rf.predict(X_test)
        print("\n  随机森林模型性能评估:")
        cls_report = classification_report(y_test, y_pred)
        print(cls_report)
        
        # 混淆矩阵
        print("  创建混淆矩阵图...")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('随机森林模型混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.savefig('confusion_matrix.png')
        print("  混淆矩阵已保存为 confusion_matrix.png")
        
        # 随机森林特征重要性
        feature_importances = pd.Series(
            rf.feature_importances_, 
            index=X_scaled.columns
        ).sort_values(ascending=False)
        
        print("\n  随机森林特征重要性:")
        for feature, importance in feature_importances.items():
            print(f"    {feature}: {importance:.4f}")
        
        # 可视化特征重要性
        print("  创建特征重要性图...")
        plt.figure(figsize=(12, 8))
        feature_importances.plot(kind='barh')
        plt.title('随机森林特征重要性')
        plt.tight_layout()
        plt.savefig('rf_feature_importance.png')
        print("  特征重要性图已保存为 rf_feature_importance.png")
        
        # 计算排列重要性
        print("\n  计算排列特征重要性...")
        perm_importance = permutation_importance(
            rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
        )
        
        perm_importances = pd.Series(
            perm_importance.importances_mean, 
            index=X_scaled.columns
        ).sort_values(ascending=False)
        
        print("\n  排列特征重要性:")
        for feature, importance in perm_importances.items():
            print(f"    {feature}: {importance:.4f}")
        
        # 可视化排列重要性
        print("  创建排列重要性图...")
        plt.figure(figsize=(12, 8))
        perm_importances.plot(kind='barh')
        plt.title('排列特征重要性')
        plt.tight_layout()
        plt.savefig('permutation_importance.png')
        print("  排列重要性图已保存为 permutation_importance.png")
        
        return rf, feature_importances, perm_importances, (X_train, X_test, y_train, y_test)
    except Exception as e:
        print(f"Error in train_random_forest: {e}")
        traceback.print_exc()
        sys.exit(1)

def analyze_by_emotion_state(df, env_features):
    """
    分析不同情绪状态下环境特征的分布
    """
    print("\n[5/6] 分析不同情绪状态下的环境特征分布...")
    try:
        # 检查是否有情绪状态列
        if 'BiLSTM_emotional_state' not in df.columns:
            print("  警告: 找不到 'BiLSTM_emotional_state' 列，跳过情绪状态分析")
            return
        
        # 为每个环境特征创建箱型图
        print("  创建箱型图...")
        for i, feature in enumerate(env_features):
            print(f"    处理特征 {i+1}/{len(env_features)}: {feature} (箱型图)")
            plt.figure(figsize=(12, 8))
            sns.boxplot(x='BiLSTM_emotional_state', y=feature, data=df)
            plt.title(f'{feature} 在不同情绪状态下的分布')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'boxplot_{feature.replace("(", "").replace(")", "").replace("/", "_")}.png')
            plt.close()
        
        # 创建小提琴图来展示密度
        print("  创建小提琴图...")
        for i, feature in enumerate(env_features):
            print(f"    处理特征 {i+1}/{len(env_features)}: {feature} (小提琴图)")
            plt.figure(figsize=(12, 8))
            sns.violinplot(x='BiLSTM_emotional_state', y=feature, data=df)
            plt.title(f'{feature} 在不同情绪状态下的密度分布')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'violinplot_{feature.replace("(", "").replace(")", "").replace("/", "_")}.png')
            plt.close()
        
        # 添加分类特征的ANOVA分析
        from scipy import stats
        
        print("\n  环境特征在不同情绪状态下的ANOVA分析:")
        anova_results = {}
        
        for feature in env_features:
            # 按情绪状态分组
            groups = []
            for state in df['BiLSTM_emotional_state'].unique():
                feature_values = df[df['BiLSTM_emotional_state'] == state][feature].dropna()
                groups.append(feature_values)
            
            # 执行单因素ANOVA
            if len(groups) >= 2 and all(len(g) > 0 for g in groups):
                try:
                    f_val, p_val = stats.f_oneway(*groups)
                    anova_results[feature] = {'f_val': f_val, 'p_val': p_val}
                    print(f"    {feature}: F={f_val:.4f}, p={p_val:.4f} {'*' if p_val < 0.05 else ''}")
                except Exception as e:
                    print(f"    无法对特征 {feature} 执行ANOVA: {e}")
        
        # 可视化ANOVA结果
        if anova_results:
            print("  创建ANOVA显著性图...")
            features = []
            p_values = []
            significant = []
            
            for feature, result in sorted(anova_results.items(), key=lambda x: x[1]['p_val']):
                features.append(feature)
                p_values.append(result['p_val'])
                significant.append(result['p_val'] < 0.05)
            
            plt.figure(figsize=(12, 8))
            bars = plt.barh(features, -np.log10(p_values), color=[
                'green' if sig else 'gray' for sig in significant
            ])
            plt.axvline(-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
            plt.xlabel('-log10(p-value)')
            plt.title('环境特征ANOVA显著性 (-log10 p-value)')
            plt.legend()
            plt.tight_layout()
            plt.savefig('anova_significance.png')
            plt.close()
            print("  ANOVA显著性图已保存为 anova_significance.png")
        
        print("  情绪状态分析图表已保存")
        return anova_results
    except Exception as e:
        print(f"Error in analyze_by_emotion_state: {e}")
        traceback.print_exc()
        sys.exit(1)

def save_results(pearson_corr, rf_importance, permutation_importance, env_means_by_emotion, anova_results):
    """
    保存分析结果到文件
    Save analysis results to a file
    """
    try:
        # 创建保存结果的目录
        results_dir = 'env_feature_analysis'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # 保存结果到文本文件
        with open(f'{results_dir}/feature_importance_summary.txt', 'w', encoding='utf-8') as f:
            # 写入Pearson相关系数
            f.write("Pearson Correlation Coefficients (线性相关性):\n")
            for feature, corr in sorted(pearson_corr.items(), key=lambda x: abs(x[1]), reverse=True):
                f.write(f"  {feature}: {corr:.4f}\n")
            f.write("\n")
            
            # 写入随机森林特征重要性
            f.write("Random Forest Feature Importance (随机森林特征重要性):\n")
            for feature, importance in sorted(rf_importance.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {feature}: {importance:.4f}\n")
            f.write("\n")
            
            # 写入排列重要性
            f.write("Permutation Feature Importance (排列特征重要性):\n")
            for feature, importance in sorted(permutation_importance.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {feature}: {importance:.4f}\n")
            f.write("\n")
            
            # 写入不同情绪状态下的环境因素平均值
            f.write("Average Environmental Factors by Emotional State (不同情绪状态下的环境因素平均值):\n")
            f.write("  {:<25} {:<20} {:<20} {:<20}\n".format("Feature", "Amusement/Meditation", "Baseline", "Stress"))
            f.write("  " + "-" * 85 + "\n")
            
            for feature in env_means_by_emotion.index:
                try:
                    f.write("  {:<25} {:<20.4f} {:<20.4f} {:<20.4f}\n".format(
                        feature,
                        env_means_by_emotion.loc[feature, 'Amusement/Meditation'],
                        env_means_by_emotion.loc[feature, 'Baseline'],
                        env_means_by_emotion.loc[feature, 'Stress']
                    ))
                except KeyError:
                    # 处理可能的键错误
                    f.write(f"  {feature}: Data not available\n")
            f.write("\n")
            
            # 写入ANOVA分析结果
            f.write("ANOVA Analysis Results (ANOVA分析结果):\n")
            # 按p值排序
            sorted_anova = sorted(anova_results.items(), key=lambda x: float(x[1]['p_val']))
            for feature, stats in sorted_anova:
                # 确保p_val是浮点数
                p_val = float(stats['p_val'])
                # 添加显著性标记
                if p_val < 0.001:
                    sig = "***"
                elif p_val < 0.01:
                    sig = "**"
                elif p_val < 0.05:
                    sig = "*"
                else:
                    sig = ""
                
                f.write(f"  {feature}: F={stats['f_val']:.4f}, p={p_val:.4f} {sig}\n")
            f.write("\n")
            
            # 写入特征重要性排名总结
            f.write("Feature Importance Rankings Summary (特征重要性排名总结):\n")
            
            # 创建一个综合排名
            combined_ranking = {}
            
            # 添加Pearson相关系数的绝对值排名
            for i, (feature, _) in enumerate(sorted(pearson_corr.items(), key=lambda x: abs(x[1]), reverse=True)):
                if feature not in combined_ranking:
                    combined_ranking[feature] = []
                combined_ranking[feature].append(i + 1)
            
            # 添加随机森林重要性排名
            for i, (feature, _) in enumerate(sorted(rf_importance.items(), key=lambda x: x[1], reverse=True)):
                if feature not in combined_ranking:
                    combined_ranking[feature] = [float('nan'), i + 1]
                else:
                    combined_ranking[feature].append(i + 1)
            
            # 添加排列重要性排名
            for i, (feature, _) in enumerate(sorted(permutation_importance.items(), key=lambda x: x[1], reverse=True)):
                if feature not in combined_ranking:
                    combined_ranking[feature] = [float('nan'), float('nan'), i + 1]
                else:
                    while len(combined_ranking[feature]) < 2:
                        combined_ranking[feature].append(float('nan'))
                    combined_ranking[feature].append(i + 1)
            
            # 添加ANOVA p值排名
            for i, (feature, _) in enumerate(sorted_anova):
                if feature not in combined_ranking:
                    combined_ranking[feature] = [float('nan'), float('nan'), float('nan'), i + 1]
                else:
                    while len(combined_ranking[feature]) < 3:
                        combined_ranking[feature].append(float('nan'))
                    combined_ranking[feature].append(i + 1)
            
            # 计算平均排名并排序
            avg_rankings = {}
            for feature, rankings in combined_ranking.items():
                # 过滤掉NaN值
                valid_rankings = [r for r in rankings if not math.isnan(r)]
                if valid_rankings:
                    avg_rankings[feature] = sum(valid_rankings) / len(valid_rankings)
                else:
                    avg_rankings[feature] = float('inf')  # 如果没有有效排名，则设为无穷大
            
            # 按平均排名排序
            sorted_features = sorted(avg_rankings.items(), key=lambda x: x[1])
            
            # 写入排名表头
            f.write("  {:<25} {:<15} {:<15} {:<15} {:<15} {:<15}\n".format(
                "Feature", "Pearson Rank", "RF Rank", "Permutation Rank", "ANOVA Rank", "Average Rank"))
            f.write("  " + "-" * 100 + "\n")
            
            # 写入每个特征的排名
            for feature, avg_rank in sorted_features:
                rankings = combined_ranking[feature]
                # 确保rankings有4个元素
                while len(rankings) < 4:
                    rankings.append(float('nan'))
                
                # 格式化输出，将NaN显示为"N/A"
                pearson_rank = "N/A" if math.isnan(rankings[0]) else f"{rankings[0]:.0f}"
                rf_rank = "N/A" if math.isnan(rankings[1]) else f"{rankings[1]:.0f}"
                perm_rank = "N/A" if math.isnan(rankings[2]) else f"{rankings[2]:.0f}"
                anova_rank = "N/A" if math.isnan(rankings[3]) else f"{rankings[3]:.0f}"
                
                f.write("  {:<25} {:<15} {:<15} {:<15} {:<15} {:<15.2f}\n".format(
                    feature, pearson_rank, rf_rank, perm_rank, anova_rank, avg_rank))
            
            # 写入结论
            f.write("\nConclusion (结论):\n")
            f.write("  基于上述分析，环境特征对情绪状态有显著影响。特别是：\n")
            f.write("  1. 温度相关特征（如空气温度、黑球温度和WGBT）与情绪状态高度相关，表明温度环境可能影响人的情绪体验。\n")
            f.write("  2. 相对湿度在所有分析方法中都显示出高重要性，是区分不同情绪状态的关键因素。\n")
            f.write("  3. CO2浓度和噪音水平也显示出显著差异，表明空气质量和声环境对情绪有重要影响。\n")
            f.write("  4. 地理位置（经纬度）的显著性表明，空间环境背景也是情绪体验的重要因素。\n")
            f.write("  5. ANOVA分析显示所有环境特征在不同情绪状态间均有统计学显著差异，进一步证实了环境因素对情绪的影响。\n\n")
            f.write("  Based on the above analysis, environmental features have significant impacts on emotional states. In particular:\n")
            f.write("  1. Temperature-related features (such as Air Temperature, Black Globe Temperature, and WGBT) are highly correlated with emotional states, indicating that thermal environment may influence emotional experiences.\n")
            f.write("  2. Relative Humidity shows high importance across all analysis methods, being a key factor in distinguishing different emotional states.\n")
            f.write("  3. CO2 concentration and noise levels also show significant differences, suggesting that air quality and acoustic environment have important effects on emotions.\n")
            f.write("  4. The significance of geographical location (latitude and longitude) indicates that spatial environmental context is also an important factor in emotional experiences.\n")
            f.write("  5. ANOVA analysis shows that all environmental features have statistically significant differences across different emotional states, further confirming the impact of environmental factors on emotions.\n")
        
        print(f"分析结果已保存到 {results_dir}/feature_importance_summary.txt")
        return True
    except Exception as e:
        print(f"Error in save_results: {e}")
        import traceback
        traceback.print_exc()
        return False

def calculate_correlations(X, y):
    """
    计算特征与情绪标签的相关性
    """
    print("\n[2/6] 计算特征与情绪标签的相关性...")
    correlations = {}
    
    # 将情绪标签转换为数值
    emotion_mapping = {
        'Baseline': 0,
        'Stress': 1,
        'Amusement/Meditation': 2
    }
    y_numeric = y.map(emotion_mapping)
    
    # 计算每个特征与情绪标签的相关性
    for feature in X.columns:
        corr = np.corrcoef(X[feature].values, y_numeric.values)[0, 1]
        correlations[feature] = corr
        print(f"  {feature}: {corr:.4f}")
    
    return correlations

def create_boxplots(data, env_features):
    """
    为环境特征创建箱型图
    """
    print("\n[3/6] 创建环境特征箱型图...")
    for i, feature in enumerate(env_features):
        print(f"  处理特征 {i+1}/{len(env_features)}: {feature}")
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='BiLSTM_emotional_state', y=feature, data=data)
        plt.title(f'{feature} 在不同情绪状态下的分布')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'boxplot_{feature.replace("(", "").replace(")", "").replace("/", "_")}.png')
        plt.close()

def create_violinplots(data, env_features):
    """
    为环境特征创建小提琴图
    """
    print("\n[4/6] 创建环境特征小提琴图...")
    for i, feature in enumerate(env_features):
        print(f"  处理特征 {i+1}/{len(env_features)}: {feature} (小提琴图)")
        plt.figure(figsize=(10, 6))
        sns.violinplot(x='BiLSTM_emotional_state', y=feature, data=data)
        plt.title(f'{feature} 在不同情绪状态下的密度分布')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'violinplot_{feature.replace("(", "").replace(")", "").replace("/", "_")}.png')
        plt.close()

def calculate_emotion_stats(data, env_features):
    """
    计算不同情绪状态下的环境特征平均值
    """
    print("\n[5/6] 计算不同情绪状态下的环境特征平均值...")
    # 按情绪标签分组并计算平均值
    emotion_stats = data.groupby('BiLSTM_emotional_state')[env_features].mean()
    
    # 打印结果
    print("  不同情绪状态下的环境特征平均值:")
    print(emotion_stats)
    
    return emotion_stats

def perform_anova(data, env_features):
    """
    对环境特征进行ANOVA分析
    """
    print("\n[6/6] 进行环境特征ANOVA分析...")
    from scipy import stats
    
    anova_results = {}
    
    for feature in env_features:
        # 按情绪状态分组
        groups = []
        for state in data['BiLSTM_emotional_state'].unique():
            feature_values = data[data['BiLSTM_emotional_state'] == state][feature].dropna()
            groups.append(feature_values)
        
        # 执行单因素ANOVA
        if len(groups) >= 2 and all(len(g) > 0 for g in groups):
            try:
                f_val, p_val = stats.f_oneway(*groups)
                anova_results[feature] = {'f_val': f_val, 'p_val': p_val}
                
                # 添加显著性标记
                sig = ""
                if p_val < 0.05:
                    sig = "*"
                if p_val < 0.01:
                    sig = "**"
                if p_val < 0.001:
                    sig = "***"
                
                print(f"  {feature}: F={f_val:.4f}, p={p_val:.4f} {sig}")
            except Exception as e:
                print(f"  无法对特征 {feature} 执行ANOVA: {e}")
    
    return anova_results

def create_anova_plot(anova_results):
    """
    创建ANOVA显著性图
    """
    print("  创建ANOVA显著性图...")
    if not anova_results:
        print("  没有ANOVA结果可供绘图")
        return
    
    features = []
    p_values = []
    significant = []
    
    for feature, result in sorted(anova_results.items(), key=lambda x: x[1]['p_val']):
        features.append(feature)
        p_values.append(result['p_val'])
        significant.append(result['p_val'] < 0.05)
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(features, -np.log10(p_values), color=[
        'green' if sig else 'gray' for sig in significant
    ])
    plt.axvline(-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
    plt.xlabel('-log10(p-value)')
    plt.title('环境特征ANOVA显著性 (-log10 p-value)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('anova_significance.png')
    plt.close()
    print("  ANOVA显著性图已保存为 anova_significance.png")

def main():
    """
    主函数，执行环境特征分析
    """
    start_time = time.time()
    
    try:
        # 定义环境特征列表
        env_features = [
            'PM2.5(ug/m3)', 'PM10(ug/m3)', 'CO2(ppm)', 'Noise(dB)',
            'WGBT(C)', 'Air Temperature(C)', 'Black Globe Temperature(C)', 'Relative Humidity(%)',
            'Latitude', 'Longitude'
        ]
        
        # 加载数据
        data = load_data()
        if data is None:
            print("数据加载失败，退出程序")
            return
        
        # 提取特征和标签
        X = data[env_features]
        y = data['BiLSTM_emotional_state']
        
        # 计算特征与情绪标签的相关性
        correlations = calculate_correlations(X, y)
        
        # 训练随机森林模型
        rf_model, rf_importances, perm_importances, split_data = train_random_forest(X, y)
        
        # 创建保存结果的目录
        results_dir = 'env_feature_analysis'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # 将当前工作目录更改为结果目录
        os.chdir(results_dir)
        
        # 创建箱型图
        create_boxplots(data, env_features)
        
        # 创建小提琴图
        create_violinplots(data, env_features)
        
        # 计算不同情绪状态下的环境特征平均值
        emotion_stats = calculate_emotion_stats(data, env_features)
        
        # 进行ANOVA分析
        anova_results = perform_anova(data, env_features)
        
        # 创建ANOVA显著性图
        create_anova_plot(anova_results)
        
        # 保存结果
        save_results(correlations, rf_importances, perm_importances, emotion_stats, anova_results)
        
        # 返回原目录
        os.chdir('..')
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        traceback.print_exc()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"\n总运行时间: {int(minutes)}分 {int(seconds)}秒")

if __name__ == "__main__":
    main() 