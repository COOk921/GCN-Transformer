def hit_at_k(actual, predicted, k):
    """
    计算单个用户的 Hit@K 得分
    :param actual: 用户实际感兴趣的项目列表
    :param predicted: 推荐系统为用户生成的推荐项目列表
    :param k: 考虑的推荐项目数量
    :return: 单个用户的 Hit@K 得分（0 或 1）
    """
    top_k_predicted = predicted[:k]
    for item in top_k_predicted:
        if item in actual:
            return 1
    return 0

def calculate_hit_at_k(actual_list, predicted_list, k):
    """
    计算多个用户的平均 Hit@K 得分
    :param actual_list: 多个用户的实际感兴趣项目列表
    :param predicted_list: 多个用户的推荐项目列表
    :param k: 考虑的推荐项目数量
    :return: 多个用户的平均 Hit@K 得分
    """
    num_users = len(actual_list)
    total_hits = 0
    for i in range(num_users):
        total_hits += hit_at_k(actual_list[i], predicted_list[i], k)
    return total_hits / num_users

# 示例数据
actual_list = [[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]]
predicted_list = [[1, 2, 3], [2, 5, 8], [3, 6, 9]]
k = 3
hit_k = calculate_hit_at_k(actual_list, predicted_list, k)
print(f"Hit@{k}: {hit_k}")