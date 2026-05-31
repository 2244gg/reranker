import pandas as pd
import numpy as np
from collections import defaultdict

def load_ml1m_data(users_path, items_path, ratings_path):
    """
    加载ML-1M数据集
    """
    # 加载用户数据
    users = pd.read_csv(users_path, sep='::', engine='python', 
                       names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
    
    # 加载物品数据
    items = pd.read_csv(items_path, sep='::', engine='python', 
                       names=['MovieID', 'Title', 'Genres'], encoding='latin-1')
    
    # 加载评分数据
    ratings = pd.read_csv(ratings_path, sep='::', engine='python', 
                         names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    
    return users, items, ratings

def map_age_to_group(age):
    """
    将年龄编码映射到年龄组
    ML-1M年龄编码: 1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 
                   45: "45-49", 50: "50-55", 56: "56+"
    我们分为：青年(1, 18), 中年(25, 35, 45), 老年(50, 56)
    """
    if age in [1, 18]:
        return 'Youth'  # 青年
    elif age in [25, 35, 45]:
        return 'Middle'  # 中年
    elif age in [50, 56]:
        return 'Senior'  # 老年
    else:
        return 'Unknown'

def create_user_groups(users):
    """
    创建用户分组
    """
    # 添加年龄组
    users['AgeGroup'] = users['Age'].apply(map_age_to_group)
    
    # 创建分组字典
    groups = {
        'gender': {
            'M': set(users[users['Gender'] == 'M']['UserID']),
            'F': set(users[users['Gender'] == 'F']['UserID'])
        },
        'age': {
            'Youth': set(users[users['AgeGroup'] == 'Youth']['UserID']),
            'Middle': set(users[users['AgeGroup'] == 'Middle']['UserID']),
            'Senior': set(users[users['AgeGroup'] == 'Senior']['UserID'])
        },
        'cross': {
            'M_Youth': set(users[(users['Gender'] == 'M') & (users['AgeGroup'] == 'Youth')]['UserID']),
            'F_Youth': set(users[(users['Gender'] == 'F') & (users['AgeGroup'] == 'Youth')]['UserID']),
            'M_Middle': set(users[(users['Gender'] == 'M') & (users['AgeGroup'] == 'Middle')]['UserID']),
            'F_Middle': set(users[(users['Gender'] == 'F') & (users['AgeGroup'] == 'Middle')]['UserID']),
            'M_Senior': set(users[(users['Gender'] == 'M') & (users['AgeGroup'] == 'Senior')]['UserID']),
            'F_Senior': set(users[(users['Gender'] == 'F') & (users['AgeGroup'] == 'Senior')]['UserID'])
        },
        'neutral': {
            'All': set(users['UserID'])
        }
    }
    
    return groups, users

def calculate_popularity(ratings, user_groups):
    """
    计算每个物品在不同用户分组中的流行度
    """
    # 初始化字典来存储物品的交互次数
    popularity = defaultdict(lambda: defaultdict(int))
    
    high_ratings = ratings[ratings['Rating'] > 3].copy()

    # 获取所有物品ID
    all_items = set(ratings['MovieID'].unique())
    
    # 为每个物品初始化所有分组的计数器
    for item in all_items:
        for group_type, subgroups in user_groups.items():
            for subgroup_name in subgroups.keys():
                popularity[item][f"{group_type}_{subgroup_name}"] = 0
    
    # 计算每个物品在不同分组中的交互次数
    for _, row in high_ratings.iterrows():
        user_id = row['UserID']
        item_id = row['MovieID']
        
        # 检查每个分组
        for group_type, subgroups in user_groups.items():
            for subgroup_name, user_set in subgroups.items():
                if user_id in user_set:
                    popularity[item_id][f"{group_type}_{subgroup_name}"] += 1
    
    return popularity

def normalize_popularity(popularity):
    """
    对流行度进行归一化（0-1范围）
    """
    # 为每个分组计算最大值
    max_counts = {}
    
    # 找出每个分组的最大交互次数
    for item in popularity.keys():
        for group_key, count in popularity[item].items():
            if group_key not in max_counts:
                max_counts[group_key] = count
            else:
                max_counts[group_key] = max(max_counts[group_key], count)
    
    # 归一化
    normalized_popularity = defaultdict(dict)
    
    for item in popularity.keys():
        normalized_popularity[item] = {}
        for group_key, count in popularity[item].items():
            if max_counts[group_key] > 0:
                normalized_popularity[item][group_key] = count / max_counts[group_key]
            else:
                normalized_popularity[item][group_key] = 0.0
    
    return normalized_popularity

def add_popularity_to_items(items, normalized_popularity):
    """
    将计算出的流行度添加到物品数据中
    """
    # 定义分组顺序（便于输出）
    group_order = [
        'neutral_All',  # 中立组
        'gender_M', 'gender_F',  # 性别组
        'age_Youth', 'age_Middle', 'age_Senior',  # 年龄组
        'cross_M_Youth', 'cross_F_Youth',  # 交叉组
        'cross_M_Middle', 'cross_F_Middle',
        'cross_M_Senior', 'cross_F_Senior'
    ]
    
    # 创建新的DataFrame来存储结果
    items_with_popularity = items.copy()
    
    # 为每个分组创建列
    for group_key in group_order:
        items_with_popularity[group_key] = 0.0
    
    # 填充流行度值
    for idx, row in items_with_popularity.iterrows():
        item_id = row['MovieID']
        if item_id in normalized_popularity:
            for group_key in group_order:
                if group_key in normalized_popularity[item_id]:
                    items_with_popularity.at[idx, group_key] = normalized_popularity[item_id][group_key]
    
    return items_with_popularity

def main():
    # 文件路径 - 请根据实际情况修改
    users_path = 'datasets/ml-1m/ml-1m/users.dat'
    items_path = 'datasets/ml-1m/ml-1m/movies.dat'
    ratings_path = 'datasets/ml-1m/ml-1m/ratings.dat'
    
    print("加载数据...")
    users, items, ratings = load_ml1m_data(users_path, items_path, ratings_path)
    
    print("创建用户分组...")
    user_groups, users_with_groups = create_user_groups(users)
    
    print("计算物品流行度...")
    item_popularity = calculate_popularity(ratings, user_groups)
    
    print("归一化流行度...")
    normalized_popularity = normalize_popularity(item_popularity)
    
    print("将流行度添加到物品数据...")
    items_with_popularity = add_popularity_to_items(items, normalized_popularity)
    
    # 输出结果
    output_path = 'movies_with_popularity.dat'
    print(f"保存结果到 {output_path}...")
    
    # 保存为dat文件，使用::分隔符
    items_with_popularity.to_csv(output_path, sep=':', index=False, header=False)
    
    # 同时保存CSV格式便于查看
    csv_output_path = 'movies_with_popularity.csv'
    items_with_popularity.to_csv(csv_output_path, index=False)
    
    print("数据预处理完成！")
    print(f"原始物品数量: {len(items)}")
    print(f"处理后的物品数量: {len(items_with_popularity)}")
    
    # 显示部分结果
    print("\n前5个物品的流行度示例:")
    print(items_with_popularity[['MovieID', 'Title', 'neutral_All', 'gender_M', 'gender_F']].head())
    
    # 统计信息
    print("\n各分组流行度统计:")
    group_columns = [col for col in items_with_popularity.columns if col not in ['MovieID', 'Title', 'Genres']]
    for col in group_columns:
        print(f"{col}: 均值={items_with_popularity[col].mean():.4f}, "
              f"最大值={items_with_popularity[col].max():.4f}, "
              f"最小值={items_with_popularity[col].min():.4f}")

def alternative_implementation():
    """
    另一种实现方式：使用矩阵运算提高效率（适用于大数据集）
    """
    # 文件路径
    users_path = 'users.dat'
    items_path = 'movies.dat'
    ratings_path = 'ratings.dat'
    
    print("加载数据...")
    users, items, ratings = load_ml1m_data(users_path, items_path, ratings_path)
    
    # 添加年龄组
    users['AgeGroup'] = users['Age'].apply(map_age_to_group)
    
    # 创建用户-物品交互矩阵
    print("创建用户-物品交互矩阵...")
    
    # 方法1：使用pivot_table
    interaction_matrix = pd.pivot_table(
        ratings, 
        values='Rating',
        index='UserID',
        columns='MovieID',
        aggfunc='count',
        fill_value=0
    )
    
    # 创建分组掩码
    groups_info = {}
    
    # 性别组
    male_users = set(users[users['Gender'] == 'M']['UserID'])
    female_users = set(users[users['Gender'] == 'F']['UserID'])
    
    # 年龄组
    youth_users = set(users[users['AgeGroup'] == 'Youth']['UserID'])
    middle_users = set(users[users['AgeGroup'] == 'Middle']['UserID'])
    senior_users = set(users[users['AgeGroup'] == 'Senior']['UserID'])
    
    # 交叉组
    male_youth = male_users.intersection(youth_users)
    female_youth = female_users.intersection(youth_users)
    male_middle = male_users.intersection(middle_users)
    female_middle = female_users.intersection(middle_users)
    male_senior = male_users.intersection(senior_users)
    female_senior = female_users.intersection(senior_users)
    
    # 计算每个分组的流行度
    popularity_results = pd.DataFrame(index=items['MovieID'])
    
    # 中立组
    popularity_results['neutral_All'] = interaction_matrix.sum(axis=0)
    
    # 性别组
    popularity_results['gender_M'] = interaction_matrix.loc[
        interaction_matrix.index.isin(male_users)
    ].sum(axis=0)
    popularity_results['gender_F'] = interaction_matrix.loc[
        interaction_matrix.index.isin(female_users)
    ].sum(axis=0)
    
    # 年龄组
    popularity_results['age_Youth'] = interaction_matrix.loc[
        interaction_matrix.index.isin(youth_users)
    ].sum(axis=0)
    popularity_results['age_Middle'] = interaction_matrix.loc[
        interaction_matrix.index.isin(middle_users)
    ].sum(axis=0)
    popularity_results['age_Senior'] = interaction_matrix.loc[
        interaction_matrix.index.isin(senior_users)
    ].sum(axis=0)
    
    # 交叉组
    popularity_results['cross_M_Youth'] = interaction_matrix.loc[
        interaction_matrix.index.isin(male_youth)
    ].sum(axis=0)
    popularity_results['cross_F_Youth'] = interaction_matrix.loc[
        interaction_matrix.index.isin(female_youth)
    ].sum(axis=0)
    popularity_results['cross_M_Middle'] = interaction_matrix.loc[
        interaction_matrix.index.isin(male_middle)
    ].sum(axis=0)
    popularity_results['cross_F_Middle'] = interaction_matrix.loc[
        interaction_matrix.index.isin(female_middle)
    ].sum(axis=0)
    popularity_results['cross_M_Senior'] = interaction_matrix.loc[
        interaction_matrix.index.isin(male_senior)
    ].sum(axis=0)
    popularity_results['cross_F_Senior'] = interaction_matrix.loc[
        interaction_matrix.index.isin(female_senior)
    ].sum(axis=0)
    
    # 归一化
    print("归一化处理...")
    normalized_popularity = popularity_results.copy()
    for column in normalized_popularity.columns:
        max_val = normalized_popularity[column].max()
        if max_val > 0:
            normalized_popularity[column] = normalized_popularity[column] / max_val
    
    # 合并到原始物品数据
    items_with_popularity = items.merge(
        normalized_popularity, 
        left_on='MovieID', 
        right_index=True,
        how='left'
    )
    
    # 填充NaN值为0
    popularity_columns = [col for col in normalized_popularity.columns]
    items_with_popularity[popularity_columns] = items_with_popularity[popularity_columns].fillna(0)
    
    # 保存结果
    items_with_popularity.to_csv('movies_with_popularity_matrix.csv', index=False)
    print("处理完成！")
    
    return items_with_popularity

if __name__ == "__main__":
    print("=== ML-1M数据集预处理 ===")
    print("选择实现方式:")
    print("1. 字典循环方式（清晰易懂）")
    print("2. 矩阵运算方式（效率更高）")
    
    choice = input("请输入选择 (1 或 2): ")
    
    if choice == "1":
        main()
    elif choice == "2":
        result = alternative_implementation()
        print("\n处理完成！")
        print(f"处理了 {len(result)} 个物品")
        print("结果已保存到 movies_with_popularity_matrix.csv")
    else:
        print("无效选择，使用默认方式...")
        main()
# 我还提供了一个辅助函数，用于验证结果：

# python
# def verify_results():
#     """
#     验证处理结果的函数
#     """
#     # 加载原始数据
#     users, items, ratings = load_ml1m_data('users.dat', 'movies.dat', 'ratings.dat')
    
#     # 加载处理后的数据
#     try:
#         processed_items = pd.read_csv('movies_with_popularity.csv')
#     except:
#         processed_items = pd.read_csv('movies_with_popularity_matrix.csv')
    
#     print("验证数据完整性...")
#     print(f"原始物品数量: {len(items)}")
#     print(f"处理后物品数量: {len(processed_items)}")
    
#     # 验证特定物品的流行度计算
#     sample_item = items['MovieID'].iloc[0]
#     print(f"\n验证物品 {sample_item} 的流行度:")
    
#     # 手动计算中立组流行度
#     item_ratings = ratings[ratings['MovieID'] == sample_item]
#     total_interactions = len(item_ratings)
    
#     print(f"手动计算的中立组交互次数: {total_interactions}")
    
#     if sample_item in processed_items['MovieID'].values:
#         item_row = processed_items[processed_items['MovieID'] == sample_item].iloc[0]
#         print(f"处理后数据的中立组流行度: {item_row['neutral_All']}")
        
#         # 计算归一化值
#         max_interactions = len(ratings[ratings['MovieID'] == sample_item])
#         normalized = total_interactions / max_interactions if max_interactions > 0 else 0
#         print(f"手动计算的归一化值: {normalized}")
    
#     # 验证性别组
#     male_users = set(users[users['Gender'] == 'M']['UserID'])
#     male_interactions = len(item_ratings[item_ratings['UserID'].isin(male_users)])
#     print(f"\n男性用户交互次数: {male_interactions}")
    
#     # 显示处理后的列
#     print(f"\n处理后数据包含的流行度列:")
#     pop_columns = [col for col in processed_items.columns if col not in ['MovieID', 'Title', 'Genres']]
#     print(pop_columns)
    
#     # 统计缺失值
#     missing_values = processed_items[pop_columns].isnull().sum()
#     print(f"\n缺失值统计:")
#     print(missing_values[missing_values > 0])