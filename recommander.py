import pandas as pd
import os
import numpy as np
import random
import json
import time
from collections import defaultdict
from sklearn.model_selection import train_test_split
from openai import OpenAI
import openai
from typing import List, Dict, Tuple
import re

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # python-dotenv optional; environment variables can be set externally.
    pass

_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
_OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.laozhang.ai/v1")
if not _OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Copy .env.example to .env and fill in your key, "
        "or export OPENAI_API_KEY in the environment."
    )

client = OpenAI(
    api_key=_OPENAI_API_KEY,
    base_url=_OPENAI_BASE_URL,
)

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
        return 'young'
    elif age in [25, 35, 45]:
        return 'middle-aged'
    elif age in [50, 56]:
        return 'elderly'
    else:
        return 'unkonw'

def split_train_test_ratings(ratings, test_size=0.7, random_state=42):
    """
    划分训练集和测试集
    按用户划分，确保每个用户在训练集和测试集中都有数据
    """
    train_ratings = []
    test_ratings = []
    
    # 按用户分组
    user_groups = ratings.groupby('UserID')
    
    for user_id, user_ratings in user_groups:
        # 确保每个用户至少有2条评分记录
        if len(user_ratings) >= 2:
            # 随机选择测试集
            user_train, user_test = train_test_split(
                user_ratings, 
                test_size=test_size, 
                random_state=random_state
            )
            train_ratings.append(user_train)
            test_ratings.append(user_test)
        else:
            # 如果只有1条记录，放入训练集
            train_ratings.append(user_ratings)
    
    # 合并结果
    train_df = pd.concat(train_ratings, ignore_index=True)
    test_df = pd.concat(test_ratings, ignore_index=True)
    
    return train_df, test_df

def extract_user_interaction_history(train_ratings, items, users, rating_threshold=3):
    """
    提取每个用户的好评交互历史
    """
    # 筛选好评
    high_ratings = train_ratings[train_ratings['Rating'] > rating_threshold].copy()
    
    # 合并电影信息
    high_ratings_with_info = high_ratings.merge(
        items[['MovieID', 'Title', 'Genres']], 
        on='MovieID', 
        how='left'
    )
    
    # 按用户分组，获取每个用户的好评电影历史
    user_history = {}
    for user_id, user_ratings in high_ratings_with_info.groupby('UserID'):
        # 获取电影信息
        movie_history = []
        for _, row in user_ratings.iterrows():
            movie_info = {
                'title': row['Title'],
                'genres': row['Genres'].split('|') if pd.notna(row['Genres']) else [],
                'rating': row['Rating']
            }
            movie_history.append(movie_info)
        
        # 获取用户信息
        user_info = users[users['UserID'] == user_id].iloc[0]
        age_group = map_age_to_group(user_info['Age'])
        
        user_history[user_id] = {
            'gender': 'male' if user_info['Gender'] == 'M' else 'female',
            'age_group': age_group,
            'movie_history': movie_history
        }
    
    return user_history

def generate_prompts(user_info, user_history, movie_count=20):
    """
    为每个用户生成四种提示词
    """
    # 从历史中提取电影标题
    movie_titles = [movie['title'] for movie in user_history[:10]]  # 最多显示10部
    history_text = "，".join(movie_titles)
    
    # 如果历史为空，使用占位符
    if not history_text:
        history_text = "No watching history for this user"
    
    # 用户身份信息
    gender = user_info['gender']
    age_group = user_info['age_group']
    
    # 四种提示词
    prompts = {
        'neutral': f"""
        The user's viewing history is as follows: {history_text}
        Please recommend {movie_count} movies he/she might like based on his/her viewing history, 
        ranked from most likely to least likely, and only provide the movie titles. 
        
        Please output the recommended results strictly in the following format, with each movie on a separate line:
        1. Movie Title (Year) - Genre
        2. Movie Title (Year) - Genre
        ...
        """,
        
        'age': f"""
        The user is {age_group}, and his/her viewing history is as follows: {history_text} 
        Please recommend {movie_count} movies he/she might like based on his/her identity and viewing history,
        ranked from most likely to least likely, and only provide the movie titles. 

        Please output the recommended results strictly in the following format, with each movie on a separate line:
        1. Movie Title (Year) - Genre
        2. Movie Title (Year) - Genre
        ...
        """,
        
        'gender': f"""
        The user sex is {gender}, and his/her viewing history is as follows: {history_text} 
        Please recommend {movie_count} movies he/she might like based on his/her identity and viewing history,
        ranked from most likely to least likely, and only provide the movie titles. 

        Please output the recommended results strictly in the following format, with each movie on a separate line:
        1. Movie Title (Year) - Genre
        2. Movie Title (Year) - Genre
        ...
        """,
        
        'cross': f"""
        The user is {age_group} {gender}, and his/her viewing history is as follows: {history_text} 
        Please recommend {movie_count} movies he/she might like based on his/her identity and viewing history,
        ranked from most likely to least likely, and only provide the movie titles.
        
        Please output the recommended results strictly in the following format, with each movie on a separate line:
        1. Movie Title (Year) - Genre
        2. Movie Title (Year) - Genre
        ...
        """
    }
    
    return prompts

def call_chatgpt(prompt, model="gpt-4.1-mini", max_tokens=1000, temperature=0.7):
    """
    调用ChatGPT API获取推荐结果
    """
    try:
        response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a professional movie recommendation system."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
                stop=None
            )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"调用ChatGPT API时出错: {e}")
        # 返回模拟结果用于测试
        return generate_mock_recommendations()

def generate_mock_recommendations():
    """
    生成模拟推荐结果（用于测试，避免API调用）
    """
    movies = [
        "肖申克的救赎 (1994) - 剧情/犯罪",
        "教父 (1972) - 剧情/犯罪",
        "黑暗骑士 (2008) - 动作/犯罪/剧情",
        "教父2 (1974) - 剧情/犯罪",
        "十二怒汉 (1957) - 剧情",
        "辛德勒的名单 (1993) - 剧情/历史/战争",
        "指环王：王者归来 (2003) - 剧情/冒险/奇幻",
        "低俗小说 (1994) - 剧情/犯罪",
        "指环王：护戒使者 (2001) - 剧情/冒险/奇幻",
        "黄金三镖客 (1966) - 冒险/西部",
        "搏击俱乐部 (1999) - 剧情",
        "指环王：双塔奇兵 (2002) - 剧情/冒险/奇幻",
        "飞越疯人院 (1975) - 剧情",
        "星球大战5：帝国反击战 (1980) - 动作/冒险/奇幻",
        "黑客帝国 (1999) - 动作/科幻",
        "好家伙 (1990) - 传记/犯罪/剧情",
        "上帝之城 (2002) - 犯罪/剧情",
        "七武士 (1954) - 冒险/剧情",
        "生活多美好 (1946) - 剧情/家庭/奇幻",
        "沉默的羔羊 (1991) - 犯罪/剧情/惊悚"
    ]
    
    return "\n".join([f"{i+1}. {movie}" for i, movie in enumerate(movies[:20])])

def parse_recommendations(recommendation_text):
    """
    解析推荐结果，提取电影标题
    """
    movies = []
    lines = recommendation_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        # 跳过空行
        if not line:
            continue
        
        # 尝试匹配格式：数字. 电影标题 (年份) - 类型
        if '.' in line:
            # 移除前面的数字和点
            movie_info = line.split('.', 1)[1].strip()
            # 提取电影标题（去掉年份和类型）
            if '(' in movie_info and ')' in movie_info:
                title_end = movie_info.find('(')
                movie_title = movie_info[:title_end].strip()
            elif '-' in movie_info:
                title_end = movie_info.find('-')
                movie_title = movie_info[:title_end].strip()
            else:
                movie_title = movie_info
            
            if movie_title:
                movies.append(movie_title)
    
    return movies

def process_movie_titles(movie_set):
    """处理电影标题集合"""
    processed_movies = set()
    
    for movie in movie_set:
        # 1. 移除年份部分（包括括号和括号内的内容）
        # 匹配末尾的 (YYYY) 格式
        movie_without_year = re.sub(r'\s*\(\d{4}\)$', '', movie)
        
        # 2. 处理 ", The" 格式的电影名
        # 如果电影名以 ", The" 结尾（不区分大小写）
        if re.search(r',\s*The$', movie_without_year, re.IGNORECASE):
            # 移除 ", The" 并将其放到开头
            cleaned_name = re.sub(r',\s*The$', '', movie_without_year, flags=re.IGNORECASE)
            processed_movie = f"The {cleaned_name}"
        elif re.search(r',\s*A$', movie_without_year, re.IGNORECASE):
            cleaned_name = re.sub(r',\s*A$', '', movie_without_year, flags=re.IGNORECASE)
            processed_movie = f"A {cleaned_name}"
        elif re.search(r',\s*An$', movie_without_year, re.IGNORECASE):
            cleaned_name = re.sub(r',\s*An$', '', movie_without_year, flags=re.IGNORECASE)
            processed_movie = f"An {cleaned_name}"
        else:
            processed_movie = movie_without_year
            
        processed_movies.add(processed_movie)
    
    return processed_movies

def evaluate_recommendations(recommendations, test_ratings, user_id, items, top_k=20):
    """
    评估推荐结果
    """
    # 获取用户测试集中的好评电影
    user_test_ratings = test_ratings[
        (test_ratings['UserID'] == user_id) & 
        (test_ratings['Rating'] > 3)
    ]
    
    # 获取测试集中的电影标题
    test_movies = set()
    for _, row in user_test_ratings.iterrows():
        movie_info = items[items['MovieID'] == row['MovieID']]
        if not movie_info.empty:
            test_movies.add(movie_info.iloc[0]['Title'])

    test_movies = process_movie_titles(test_movies)
    true_preferred_movies = []
    # 计算命中率
    hits = 0
    for movie in recommendations[:top_k]:
        if movie in test_movies:
            hits += 1
            true_preferred_movies.append(movie)
    
    precision = hits / min(top_k, len(recommendations)) if recommendations else 0
    recall = hits / len(test_movies) if test_movies else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'hits': hits,
        'test_movies_count': len(test_movies)
    },true_preferred_movies

def main():
    # 文件路径
    sample_len=300
    sample_start=0
    random_seed=42
    users_path = 'datasets/ml-1m/ml-1m/users.dat'
    items_path = 'datasets/ml-1m/ml-1m/movies.dat'
    ratings_path = 'datasets/ml-1m/ml-1m/ratings.dat'
    
    print("=== ML-1M电影推荐系统 ===")
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    users, items, ratings = load_ml1m_data(users_path, items_path, ratings_path)
    print(f"用户数: {len(users)}")
    print(f"电影数: {len(items)}")
    print(f"评分记录数: {len(ratings)}")
    
    # 2. 划分训练集和测试集
    print("\n2. 划分训练集和测试集...")
    train_ratings, test_ratings = split_train_test_ratings(ratings, test_size=0.7, random_state=random_seed)
    print(f"训练集大小: {len(train_ratings)}")
    print(f"测试集大小: {len(test_ratings)}")
    
    # 3. 提取用户交互历史
    print("\n3. 提取用户好评交互历史...")
    user_history_dict = extract_user_interaction_history(train_ratings, items, users)
    print(f"有交互历史的用户数: {len(user_history_dict)}")

    # 选择部分用户进行测试
    total_users=len(user_history_dict)
    
    
    
    while sample_start<total_users:
        results = {}
        sample_users = list(user_history_dict.keys())[sample_start:sample_start+sample_len] # 每次处理500个用户
        print(f"\n4. 选择 {len(sample_users)} 个用户进行测试...")
        # 为每个用户生成提示词并获取推荐
        for i, user_id in enumerate(sample_users):
            print(f"\n--- 处理用户 {user_id} ({i+1}/{len(sample_users)}) ---")
            
            user_info = user_history_dict[user_id]
            movie_history = user_info['movie_history']
            
            print(f"用户身份: {user_info['gender']}{user_info['age_group']}")
            print(f"历史电影数: {len(movie_history)}")
            
            # 生成四种提示词
            prompts = generate_prompts(user_info, movie_history)
            
            user_results = {}
            
            # 对每种提示词获取推荐
            for prompt_type, prompt in prompts.items():
                print(f"\n  生成{prompt_type}提示词推荐...")
                
                # 调用ChatGPT
                recommendation_text = call_chatgpt(prompt)
                
                # 使用模拟数据
                # recommendation_text = generate_mock_recommendations()
                
                # 解析推荐结果
                recommendations = parse_recommendations(recommendation_text)
                
                # 评估推荐结果
                eval_results, true_preferred_movies = evaluate_recommendations(
                    recommendations, test_ratings, user_id, items
                )
                
                user_results[prompt_type] = {
                    'recommendations': recommendations,
                    'true_preferred_movies': true_preferred_movies,
                    'evaluation': eval_results
                }
                
                print(f"    推荐电影数: {len(recommendations)}")
                print(f"    精确率: {eval_results['precision']:.4f}")
                print(f"    召回率: {eval_results['recall']:.4f}")
            del user_info['movie_history']
            results[user_id] = {
                'user_info': user_info,
                'results': user_results,
            }
        
        # 保存结果
        print("\n5. 保存结果...")
        
        # JSON格式结果
        output_dir=f'{random_seed}'
        if not os.path.exists(output_dir):
            os.makedirs(f'{output_dir}/JSON')
            os.makedirs(f'{output_dir}/CSV')
            print(f"已创建目录: {output_dir}")
        with open(f'{output_dir}/JSON/recommendation_results_{sample_start}.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # CSV格式结果
        summary_data = []
        for user_id, user_data in results.items():
            user_info = user_data['user_info']
            for prompt_type, prompt_data in user_data['results'].items():
                eval_results = prompt_data['evaluation']
                summary_data.append({
                    'user_id': user_id,
                    'gender': user_info['gender'],
                    'age_group': user_info['age_group'],
                    'prompt_type': prompt_type,
                    'precision': eval_results['precision'],
                    'recall': eval_results['recall'],
                    'hits': eval_results['hits'],
                    'test_movies': eval_results['test_movies_count'],
                    'recommendation_count': len(prompt_data['recommendations'])
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f'{output_dir}/CSV/recommendation_summary_{sample_start}.csv', index=False, encoding='utf-8-sig')
        
        print("\n=== 处理完成 ===")
        print(f"结果已保存到: recommendation_results_{sample_start}.json 和 recommendation_summary_{sample_start}.csv")
        sample_start+=sample_len
        # 显示汇总统计
        print("\n=== 推荐结果汇总 ===")
        print(f"处理用户数: {sample_start}")
        
        
        # 计算平均指标
        # if not summary_df.empty:
        #     avg_precision = summary_df.groupby('prompt_type')['precision'].mean()
        #     avg_recall = summary_df.groupby('prompt_type')['recall'].mean()
            
        #     print("\n不同提示词类型的平均性能:")
        #     for prompt_type in ['neutral', 'age', 'gender', 'cross']:
        #         if prompt_type in avg_precision:
        #             print(f"{prompt_type}: 精确率={avg_precision[prompt_type]:.4f}, "
        #                 f"召回率={avg_recall[prompt_type]:.4f}")
    
    #return results, summary_df

def analyze_results():
    """
    分析推荐结果
    """
    try:
        with open('recommendation_results.json', 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        summary_df = pd.read_csv('recommendation_summary.csv')
        
        print("=== 推荐结果分析 ===")
        print(f"总用户数: {len(results)}")
        
        # 显示每个用户的详细结果
        for user_id, user_data in results.items():
            user_info = user_data['user_info']
            print(f"\n用户 {user_id} ({user_info['gender']}{user_info['age_group']}):")
            
            for prompt_type, prompt_data in user_data['results'].items():
                eval_results = prompt_data['evaluation']
                print(f"  {prompt_type}: 精确率={eval_results['precision']:.4f}, "
                      f"召回率={eval_results['recall']:.4f}")
        
        # 分组统计
        print("\n=== 分组统计 ===")
        
        # 按性别统计
        gender_stats = summary_df.groupby('gender').agg({
            'precision': 'mean',
            'recall': 'mean'
        }).round(4)
        print("\n按性别统计:")
        print(gender_stats)
        
        # 按年龄组统计
        age_stats = summary_df.groupby('age_group').agg({
            'precision': 'mean',
            'recall': 'mean'
        }).round(4)
        print("\n按年龄组统计:")
        print(age_stats)
        
        # 按提示词类型统计
        prompt_stats = summary_df.groupby('prompt_type').agg({
            'precision': 'mean',
            'recall': 'mean'
        }).round(4)
        print("\n按提示词类型统计:")
        print(prompt_stats)
        
    except FileNotFoundError:
        print("找不到结果文件，请先运行主程序")

def batch_process_users(user_ids, user_history_dict, test_ratings, items, batch_size=5):
    """
    批量处理用户推荐
    """
    all_results = {}
    
    for i in range(0, len(user_ids), batch_size):
        batch = user_ids[i:i+batch_size]
        print(f"\n处理批次 {i//batch_size + 1}/{(len(user_ids)+batch_size-1)//batch_size}")
        
        for user_id in batch:
            if user_id in user_history_dict:
                print(f"  处理用户 {user_id}")
                
                user_info = user_history_dict[user_id]
                movie_history = user_info['movie_history']
                
                # 生成四种提示词
                prompts = generate_prompts(user_info, movie_history)
                
                user_results = {}
                
                # 对每种提示词获取推荐
                for prompt_type, prompt in prompts.items():
                    # 添加延迟避免API限制
                    time.sleep(1)  # 每秒调用一次API
                    
                    # 调用ChatGPT
                    recommendation_text = call_chatgpt(prompt)

                    # recommendation_text = generate_mock_recommendations()
                    
                    # 解析推荐结果
                    recommendations = parse_recommendations(recommendation_text)
                    
                    # 评估推荐结果
                    eval_results, true_preferred_movies = evaluate_recommendations(
                        recommendations, test_ratings, user_id, items
                    )
                    
                    user_results[prompt_type] = {
                        'recommendations': recommendations,
                        'true_preferred_movies': true_preferred_movies,
                        'evaluation': eval_results
                    }
                
                all_results[user_id] = {
                    'user_info': user_info,
                    'results': user_results
                }
    
    return all_results

def interactive_mode():
    """
    交互式推荐模式：为单个用户生成推荐
    """
    print("=== 交互式电影推荐 ===")
    
    # 加载数据
    users, items, ratings = load_ml1m_data('users.dat', 'movies.dat', 'ratings.dat')
    
    # 选择用户
    user_id = int(input("请输入用户ID（1-6040）: "))
    
    if user_id not in users['UserID'].values:
        print(f"用户 {user_id} 不存在")
        return
    
    # 划分训练集和测试集
    train_ratings, test_ratings = split_train_test_ratings(ratings, test_size=0.2)
    
    # 提取用户历史
    user_history_dict = extract_user_interaction_history(train_ratings, items, users)
    
    if user_id not in user_history_dict:
        print(f"用户 {user_id} 在训练集中没有好评记录")
        return
    
    user_info = user_history_dict[user_id]
    movie_history = user_info['movie_history']
    
    print(f"\n用户信息:")
    print(f"  性别: {user_info['gender']}")
    print(f"  年龄组: {user_info['age_group']}")
    print(f"  历史好评电影: {len(movie_history)}部")
    
    # 显示历史电影
    if movie_history:
        print("\n历史电影列表:")
        for i, movie in enumerate(movie_history[:10], 1):
            print(f"  {i}. {movie['title']} ({', '.join(movie['genres'][:3])}) - 评分: {movie['rating']}")
    
    # 选择提示词类型
    print("\n请选择提示词类型:")
    print("1. 中立提示词")
    print("2. 年龄提示词")
    print("3. 性别提示词")
    print("4. 交叉身份提示词")
    print("5. 所有类型")
    
    choice = input("请输入选择 (1-5): ")
    
    prompt_types = []
    if choice == '1':
        prompt_types = ['neutral']
    elif choice == '2':
        prompt_types = ['age']
    elif choice == '3':
        prompt_types = ['gender']
    elif choice == '4':
        prompt_types = ['cross']
    else:
        prompt_types = ['neutral', 'age', 'gender', 'cross']
    
    # 生成推荐
    for prompt_type in prompt_types:
        print(f"\n=== {prompt_type}推荐 ===")
        
        prompts = generate_prompts(user_info, movie_history)
        prompt = prompts[prompt_type]
        
        print("正在生成推荐...")
        
        # 调用ChatGPT
        recommendation_text = call_chatgpt(prompt)
        
        # recommendation_text = generate_mock_recommendations()
        
        print("\n推荐结果:")
        print(recommendation_text)
        
        # 解析并评估
        recommendations = parse_recommendations(recommendation_text)
        eval_results, true_preferred_movies = evaluate_recommendations(recommendations, test_ratings, user_id, items)
        
        print(f"\n评估结果:")
        print(f"  精确率: {eval_results['precision']:.4f}")
        print(f"  召回率: {eval_results['recall']:.4f}")
        print(f"  命中数: {eval_results['hits']}/{min(20, len(recommendations))}")
        print(f"  测试集电影数: {eval_results['test_movies_count']}")

if __name__ == "__main__":
    print("=== ML-1M电影推荐系统 ===")
    print("请选择模式:")
    print("1. 完整处理模式（批量处理用户）")
    print("2. 交互式模式（单个用户）")
    print("3. 分析已有结果")
    
    mode = input("请输入选择 (1-3): ")
    
    if mode == '1':
        # 完整处理模式
        main()
        
        # 显示详细结果
        # show_details = input("\n是否显示详细结果？(y/n): ")
        # if show_details.lower() == 'y':
        #     analyze_results()
    
    elif mode == '2':
        # 交互式模式
        interactive_mode()
    
    elif mode == '3':
        # 分析已有结果
        analyze_results()
    
    else:
        print("无效选择")