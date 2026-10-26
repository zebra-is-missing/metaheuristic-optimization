def knapsack_dp(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n+1)]
    for i in range (1, n+1):
        w = weights[i-1]
        v = values[i-1]
        for j in range (capacity+1):
            dp[i][j] = dp[i-1][j]
            if w <= j:
                dp[i][j] = max(dp[i][j], dp[i-1][j-w] + v)
    return dp[n][capacity]


weights = [2, 3, 4, 5]
values = [3, 4, 5, 6]
capacity = 5
print("Giá trị lớn nhất:", knapsack_dp(weights, values, capacity))