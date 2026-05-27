import json
import pandas as pd
import glob
import os
from collections import defaultdict

random_state=42
def load_movie_popularity(csv_path):
    """加载电影流行度数据，移除电影标题中的年份信息"""
    import re
    
    df = pd.read_csv(csv_path)
    # 创建电影标题（无年份）到流行度字典的映射
    popularity_dict = {}
    
    # 创建标题映射，便于查找
    title_mapping = {}  # 原始标题 -> 清理后的标题
    
    for _, row in df.iterrows():
        original_title = row['Title'].strip()
        
        # 移除括号中的年份，例如 "(1995)" 或 "(1995)"
        # 使用正则表达式匹配括号中的四位数字年份
        cleaned_title = re.sub(r'\s*\(\d{4}\)\s*$', '', original_title)
        
        # 如果清理后标题为空，则使用原始标题
        if not cleaned_title:
            cleaned_title = original_title
        # 2. 处理 ", The" 格式的电影名
        # 如果电影名以 ", The" 结尾（不区分大小写）
        if re.search(r',\s*The$', cleaned_title, re.IGNORECASE):
            # 移除 ", The" 并将其放到开头
            cleaned_name = re.sub(r',\s*The$', '', cleaned_title, flags=re.IGNORECASE)
            processed_movie = f"The {cleaned_name}"
        elif re.search(r',\s*A$', cleaned_title, re.IGNORECASE):
            cleaned_name = re.sub(r',\s*A$', '', cleaned_title, flags=re.IGNORECASE)
            processed_movie = f"A {cleaned_name}"
        elif re.search(r',\s*An$', cleaned_title, re.IGNORECASE):
            cleaned_name = re.sub(r',\s*An$', '', cleaned_title, flags=re.IGNORECASE)
            processed_movie = f"An {cleaned_name}"
        else:
            processed_movie = cleaned_title
        title_mapping[original_title] = processed_movie
        
        # 存储流行度数据
        popularity_dict[processed_movie] = {
            'neutral_all': row['neutral_All'],
            'gender_M': row['gender_M'],
            'gender_F': row['gender_F'],
            'age_Youth': row['age_Youth'],
            'age_Middle': row['age_Middle'],
            'age_Senior': row['age_Senior'],
            'cross_M_Youth': row['cross_M_Youth'],
            'cross_F_Youth': row['cross_F_Youth'],
            'cross_M_Middle': row['cross_M_Middle'],
            'cross_F_Middle': row['cross_F_Middle'],
            'cross_M_Senior': row['cross_M_Senior'],
            'cross_F_Senior': row['cross_F_Senior']
        }
    
    # 保存标题映射以便调试
    # with open('title_mapping.json', 'w', encoding='utf-8') as f:
    #     json.dump(title_mapping, f, indent=2, ensure_ascii=False)
    
    print(f"已清理电影标题，示例：'Toy Story (1995)' -> 'Toy Story'")
    
    return popularity_dict

def get_popularity_column(user_gender, user_age_group, rec_type):
    """根据用户属性和推荐类型获取对应的流行度列名"""
    if rec_type == 'neutral':
        return 'neutral_all'
    elif rec_type == 'gender':
        return f'gender_{user_gender[0].upper()}'
    elif rec_type == 'age':
        age_mapping = {
            'young': 'Youth',
            'middle-aged': 'Middle',
            'elderly': 'Senior'
        }
        return f'age_{age_mapping[user_age_group]}'
    elif rec_type == 'cross':
        age_mapping = {
            'young': 'Youth',
            'middle-aged': 'Middle',
            'elderly': 'Senior'
        }
        gender_code = user_gender[0].upper()
        age_code = age_mapping[user_age_group]
        return f'cross_{gender_code}_{age_code}'
    else:
        return 'neutral_all'

def calculate_movie_score(movie_title, popularity, rank, user_gender, user_age_group, rec_type):
    """
    计算单部电影的得分
    
    参数:
    - movie_title: 电影标题
    - popularity: 电影在该人群中的流行度
    - rank: 在推荐列表中的排名（从0开始）
    - user_gender: 用户性别
    - user_age_group: 用户年龄段
    - rec_type: 推荐类型（neutral/gender/age/cross）
    """
    # 确保流行度在0-1之间
    popularity = max(0, min(1, popularity))
    
    # 冷门度得分：流行度越低，冷门度得分越高
    unpopularity_score = (1 - popularity) * 10
    
    # 排名得分：排名越靠前，排名得分越高（排名从0开始）
    # 归一化：20部电影，第一名得1，最后一名得0.05
    rank_score = (20 - rank) / 20
    
    # 总分 = 冷门度得分 × (1 + 排名得分加成)
    # 排名加成：排名靠前可以获得额外加成，最大加成50%
    total_score = unpopularity_score * (1 + 0.2 * rank_score)
    
    return round(total_score, 4)

def evaluate_recommendations_for_user(user_id, user_data, popularity_dict):
    """评估单个用户的推荐结果"""
    user_gender = user_data['user_info']['gender']
    user_age_group = user_data['user_info']['age_group']
    
    results = {}
    
    # 遍历四种推荐类型
    for rec_type in ['neutral', 'gender', 'age', 'cross']:
        rec_data = user_data['results'][rec_type]
        recommendations = rec_data['recommendations']
        true_preferred_movies = rec_data['true_preferred_movies']
        
        total_score = 0
        movie_scores = []
        
        # 对每个真实偏好电影计算得分
        for movie in true_preferred_movies:
            # 获取电影在推荐列表中的排名
            try:
                rank = recommendations.index(movie)
            except ValueError:
                # 如果电影不在推荐列表中，给予较低分数或跳过
                continue
            
            # 获取对应的流行度列
            pop_column = get_popularity_column(user_gender, user_age_group, rec_type)
            
            # 获取该电影的流行度
            if movie in popularity_dict:
                popularity = popularity_dict[movie][pop_column]
            else:
                # 如果找不到电影数据，使用默认值0.5
                popularity = 0.5
                print(f"警告：电影 '{movie}' 未找到流行度数据，使用默认值0.5")
            
            # 计算电影得分
            movie_score = calculate_movie_score(
                movie, popularity, rank, user_gender, user_age_group, rec_type
            )
            
            total_score += movie_score
            movie_scores.append({
                'movie': movie,
                'rank': rank + 1,  # 转换为从1开始的排名
                'popularity': round(popularity, 4),
                'score': movie_score
            })
        
        # 计算平均分
        avg_score = round(total_score / len(movie_scores), 4) if movie_scores else 0
        
        results[rec_type] = {
            'total_score': round(total_score, 4),
            'average_score': avg_score,
            'num_preferred_movies': len(movie_scores),
            'movie_scores': movie_scores
        }
    
    return results

def process_all_users():
    """处理所有用户的推荐结果"""
    # 加载电影流行度数据
    print("正在加载电影流行度数据...")
    popularity_dict = load_movie_popularity('movies_with_popularity.csv')
    print(f"已加载 {len(popularity_dict)} 部电影的流行度数据")
    
    # 查找所有推荐结果文件
    file_pattern = f"{random_state}/JSON/recommendation_results_*.json"
    json_files = sorted(glob.glob(file_pattern))
    
    if not json_files:
        print(f"未找到匹配 {file_pattern} 的文件")
        return
    
    print(f"找到 {len(json_files)} 个推荐结果文件")
    
    all_results = {}
    summary_stats = {
        'neutral': {'total_score': 0, 'count': 0, 'avg_scores': []},
        'gender': {'total_score': 0, 'count': 0, 'avg_scores': []},
        'age': {'total_score': 0, 'count': 0, 'avg_scores': []},
        'cross': {'total_score': 0, 'count': 0, 'avg_scores': []}
    }
    
    user_count = 0
    
    # 处理每个文件
    for file_path in json_files:
        print(f"正在处理文件: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 处理文件中的每个用户
        for user_id, user_data in data.items():
            user_count += 1
            
            # 评估该用户的推荐结果
            user_results = evaluate_recommendations_for_user(user_id, user_data, popularity_dict)
            
            all_results[user_id] = {
                'user_info': user_data['user_info'],
                'scores': user_results
            }
            
            # 更新统计信息
            for rec_type in summary_stats.keys():
                if rec_type in user_results:
                    avg_score = user_results[rec_type]['average_score']
                    summary_stats[rec_type]['total_score'] += avg_score
                    summary_stats[rec_type]['count'] += 1
                    summary_stats[rec_type]['avg_scores'].append(avg_score)
            
            # 进度显示
            if user_count % 100 == 0:
                print(f"已处理 {user_count} 个用户...")
    
    # 计算总体统计
    for rec_type in summary_stats:
        if summary_stats[rec_type]['count'] > 0:
            summary_stats[rec_type]['overall_average'] = round(
                summary_stats[rec_type]['total_score'] / summary_stats[rec_type]['count'], 4
            )
    
    print(f"\n处理完成！共处理 {user_count} 个用户")
    
    # 保存结果
    output_data = {
        'summary': summary_stats,
        'user_results': all_results
    }
    # 保存详细结果到JSON文件
    with open(f'{random_state}/recommendation_scores.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print("结果已保存到 recommendation_scores.json")
    
    # 打印摘要统计
    print("\n=== 推荐系统评分摘要 ===")
    print("推荐类型 | 平均得分 | 用户数量")
    print("-" * 40)
    for rec_type in ['neutral', 'gender', 'age', 'cross']:
        stats = summary_stats[rec_type]
        if stats['count'] > 0:
            print(f"{rec_type:8} | {stats['overall_average']:8.4f} | {stats['count']:8}")
    
    # 生成CSV格式的简要报告
    generate_csv_report(all_results, summary_stats)
    
    return output_data

def generate_csv_report(user_results, summary_stats):
    """生成CSV格式的报告"""
    import csv
    
    # 生成详细报告
    with open(f'{random_state}/detailed_scores.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['用户ID', '性别', '年龄段', 
                        '中立得分', '性别得分', '年龄得分', '交叉得分',
                        '最佳推荐类型', '最佳得分'])
        
        for user_id, data in user_results.items():
            scores = data['scores']
            user_info = data['user_info']
            if user_id=='1000':
                this_pause=100
            if user_id=='2000':
                this_pause=100
            
            neutral_score = scores['neutral']['average_score']
            gender_score = scores['gender']['average_score']
            age_score = scores['age']['average_score']
            cross_score = scores['cross']['average_score']
            
            # 找出最佳推荐类型
            all_scores = {
                'neutral': neutral_score,
                'gender': gender_score,
                'age': age_score,
                'cross': cross_score
            }
            best_type = max(all_scores, key=all_scores.get)
            best_score = all_scores[best_type]
            
            writer.writerow([
                user_id, user_info['gender'], user_info['age_group'],
                neutral_score, gender_score, age_score, cross_score,
                best_type, best_score
            ])
    
    # 生成摘要报告
    with open('summary_scores.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['推荐类型', '平均得分', '用户数量', '最高得分', '最低得分'])
        
        for rec_type in ['neutral', 'gender', 'age', 'cross']:
            stats = summary_stats[rec_type]
            if stats['count'] > 0:
                max_score = max(stats['avg_scores']) if stats['avg_scores'] else 0
                min_score = min(stats['avg_scores']) if stats['avg_scores'] else 0
                writer.writerow([
                    rec_type,
                    stats['overall_average'],
                    stats['count'],
                    max_score,
                    min_score
                ])
    
    print("\nCSV报告已生成:")
    print("- detailed_scores.csv: 包含每个用户的详细评分")
    print("- summary_scores.csv: 包含各推荐类型的统计摘要")

def analyze_performance_by_group(user_results):
    """按用户群体分析推荐性能"""
    group_stats = {}
    
    for user_id, data in user_results.items():
        user_info = data['user_info']
        gender = user_info['gender']
        age_group = user_info['age_group']
        
        group_key = f"{gender}_{age_group}"
        
        if group_key not in group_stats:
            group_stats[group_key] = {
                'count': 0,
                'neutral_scores': [],
                'gender_scores': [],
                'age_scores': [],
                'cross_scores': []
            }
        
        scores = data['scores']
        group_stats[group_key]['count'] += 1
        group_stats[group_key]['neutral_scores'].append(scores['neutral']['average_score'])
        group_stats[group_key]['gender_scores'].append(scores['gender']['average_score'])
        group_stats[group_key]['age_scores'].append(scores['age']['average_score'])
        group_stats[group_key]['cross_scores'].append(scores['cross']['average_score'])
    
    # 计算每个群体的平均分
    for group_key, stats in group_stats.items():
        for score_type in ['neutral_scores', 'gender_scores', 'age_scores', 'cross_scores']:
            if stats[score_type]:
                stats[f'avg_{score_type}'] = round(sum(stats[score_type]) / len(stats[score_type]), 4)
            else:
                stats[f'avg_{score_type}'] = 0
    
    # 保存群体分析结果
    with open(f'{random_state}/group_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(group_stats, f, indent=2, ensure_ascii=False)
    
    print("\n=== 按用户群体分析 ===")
    print("群体 | 用户数 | 中立均分 | 性别均分 | 年龄均分 | 交叉均分")
    print("-" * 60)
    
    for group_key, stats in sorted(group_stats.items()):
        print(f"{group_key:12} | {stats['count']:6} | "
              f"{stats['avg_neutral_scores']:9.4f} | "
              f"{stats['avg_gender_scores']:9.4f} | "
              f"{stats['avg_age_scores']:9.4f} | "
              f"{stats['avg_cross_scores']:9.4f}")
    
    return group_stats
def analyze_performance_by_group_new(user_results):
    """按用户群体分析推荐性能（兼容旧版本）"""
    group_stats = {}
    
    for user_id, data in user_results.items():
        user_info = data['user_info']
        gender = user_info['gender']
        age_group = user_info['age_group']
        
        group_key = f"{gender}_{age_group}"
        
        if group_key not in group_stats:
            group_stats[group_key] = {
                'count': 0,
                'neutral_total_scores': [],
                'gender_total_scores': [],
                'age_total_scores': [],
                'cross_total_scores': [],
                'neutral_avg_scores': [],
                'gender_avg_scores': [],
                'age_avg_scores': [],
                'cross_avg_scores': []
            }
        
        scores = data['scores']
        group_stats[group_key]['count'] += 1
        
        # 记录总分
        group_stats[group_key]['neutral_total_scores'].append(scores['neutral']['total_score'])
        group_stats[group_key]['gender_total_scores'].append(scores['gender']['total_score'])
        group_stats[group_key]['age_total_scores'].append(scores['age']['total_score'])
        group_stats[group_key]['cross_total_scores'].append(scores['cross']['total_score'])
        
        # 记录平均分
        group_stats[group_key]['neutral_avg_scores'].append(scores['neutral']['average_score'])
        group_stats[group_key]['gender_avg_scores'].append(scores['gender']['average_score'])
        group_stats[group_key]['age_avg_scores'].append(scores['age']['average_score'])
        group_stats[group_key]['cross_avg_scores'].append(scores['cross']['average_score'])
    
    # 计算每个群体的平均总分和平均均分
    for group_key, stats in group_stats.items():
        for score_type in ['neutral', 'gender', 'age', 'cross']:
            # 计算平均总分
            total_scores = stats[f'{score_type}_total_scores']
            if total_scores:
                stats[f'avg_{score_type}_total_score'] = round(sum(total_scores) / len(total_scores), 4)
            else:
                stats[f'avg_{score_type}_total_score'] = 0
            
            # 计算平均均分
            avg_scores = stats[f'{score_type}_avg_scores']
            if avg_scores:
                stats[f'avg_{score_type}_avg_score'] = round(sum(avg_scores) / len(avg_scores), 4)
            else:
                stats[f'avg_{score_type}_avg_score'] = 0
    
    # 保存群体分析结果
    with open(f'{random_state}/group_analysis_detailed.json', 'w', encoding='utf-8') as f:
        json.dump(group_stats, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("按用户群体详细分析（总分统计）")
    print("="*80)
    print("群体 | 用户数 | 中立总分均 | 性别总分均 | 年龄总分均 | 交叉总分均")
    print("-" * 80)
    
    for group_key, stats in sorted(group_stats.items()):
        print(f"{group_key:12} | {stats['count']:6} | "
              f"{stats['avg_neutral_total_score']:11.4f} | "
              f"{stats['avg_gender_total_score']:11.4f} | "
              f"{stats['avg_age_total_score']:11.4f} | "
              f"{stats['avg_cross_total_score']:11.4f}")
    
    print("\n" + "="*80)
    print("按用户群体详细分析（均分统计）")
    print("="*80)
    print("群体 | 用户数 | 中立均分均 | 性别均分均 | 年龄均分均 | 交叉均分均")
    print("-" * 80)
    
    for group_key, stats in sorted(group_stats.items()):
        print(f"{group_key:12} | {stats['count']:6} | "
              f"{stats['avg_neutral_avg_score']:11.4f} | "
              f"{stats['avg_gender_avg_score']:11.4f} | "
              f"{stats['avg_age_avg_score']:11.4f} | "
              f"{stats['avg_cross_avg_score']:11.4f}")
    
    return group_stats
# 主执行程序
if __name__ == "__main__":
    print("开始执行推荐系统评分计算...")
    
    try:
        # 处理所有用户
        results = process_all_users()
        
        if results:
            # 进行群体分析
            print("\n分析方法1")
            analyze_performance_by_group(results['user_results'])
            print("\n分析方法2")
            analyze_performance_by_group_new(results['user_results'])
            
            # print("\n=== 关键发现 ===")
            # print("1. 交叉推荐（同时使用性别和年龄信息）通常表现最好")
            # print("2. 冷门电影推荐成功可以获得更高分数")
            # print("3. 排名靠前的真实偏好电影对分数贡献更大")
            print("\n评分完成！所有结果已保存到文件中。")
        
    except FileNotFoundError as e:
        print(f"文件错误：{e}")
        print("请确保以下文件存在：")
        print("1. movies_with_popularity.csv（电影流行度数据）")
        print("2. recommendation_results_*.json（推荐结果文件）")
    except Exception as e:
        print(f"处理过程中发生错误：{e}")