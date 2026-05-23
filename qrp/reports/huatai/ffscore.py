"""华泰证券 FFScore 价值选股模型

参考研报：
  《华泰证券-华泰价值选股之FFScore模型：比乔斯基选股模型A股实证研究》(2017.02.09)

核心逻辑：
  FFScore (Fama-French Score) 基于 Piotroski F-Score 框架，
  从盈利能力、财务杠杆和运营效率三个维度、9个指标打分。
"""

from __future__ import annotations

import numpy as np
import polars as pl

from qrp.core.factor import Factor


class FFScoreFactor(Factor):
    """FFScore 因子：基于基本面的综合打分模型

    评分维度：
    1. 盈利能力（4项）：ROA、CFO、ΔROA、ACCRUAL
    2. 财务杠杆（3项）：ΔLEVER、ΔLIQUID、EQ_OFFER
    3. 运营效率（2项）：ΔMARGIN、ΔTURNOVER
    """

    def __init__(self):
        super().__init__(
            name="ffscore",
            description="Fama-French Score 基本面评分因子",
        )

    def calculate(self, data: pl.DataFrame) -> pl.Series:
        """计算 FFScore（需要财务数据）"""
        n = len(data)
        score = np.zeros(n)

        required = ["roa", "cfo", "lev", "liquid", "margin", "turnover"]
        has_data = all(c in data.columns for c in required)

        if not has_data:
            # 如果没有财务数据，使用简化的价格评分
            return self._calculate_price_based(data)

        roa = data["roa"].to_numpy()
        cfo = data["cfo"].to_numpy()
        lev = data["lev"].to_numpy()
        liquid = data["liquid"].to_numpy()
        margin = data["margin"].to_numpy()
        turnover = data["turnover"].to_numpy()

        for i in range(1, n):
            # F1: ROA > 0
            if roa[i] > 0:
                score[i] += 1
            # F2: CFO > 0
            if cfo[i] > 0:
                score[i] += 1
            # F3: ΔROA > 0
            if i > 0 and roa[i] > roa[i - 1]:
                score[i] += 1
            # F4: ACCRUAL = CFO - ROA > 0
            if cfo[i] - roa[i] > 0:
                score[i] += 1
            # F5: ΔLEVER < 0
            if i > 0 and lev[i] < lev[i - 1]:
                score[i] += 1
            # F6: ΔLIQUID > 0
            if i > 0 and liquid[i] > liquid[i - 1]:
                score[i] += 1
            # F7: EQ_OFFER = 0（简化：无增发）
            score[i] += 1
            # F8: ΔMARGIN > 0
            if i > 0 and margin[i] > margin[i - 1]:
                score[i] += 1
            # F9: ΔTURNOVER > 0
            if i > 0 and turnover[i] > turnover[i - 1]:
                score[i] += 1

        return pl.Series("ffscore", score / 9.0)

    def _calculate_price_based(self, data: pl.DataFrame) -> pl.Series:
        """基于价格数据的简化评分"""
        close = data["close"]
        n = len(close)
        score = np.zeros(n)

        # 使用价格动量、波动率等替代
        for i in range(20, n):
            s = 0
            # 短期动量
            if close[i] > close[i - 5]:
                s += 1
            # 中期动量
            if close[i] > close[i - 20]:
                s += 1
            # 低波动（简化）
            vol_20 = close[i - 20 : i].std()
            vol_60 = close[max(0, i - 60) : i].std() if i >= 60 else vol_20
            if vol_20 < vol_60:
                s += 1
            score[i] = s / 3.0

        return pl.Series("ffscore_price", score)


class CSCCVFramework:
    """CSCV 回测过拟合概率框架

    参考研报：
      《华泰证券-华泰金工量化系列之二十二：基于CSCV框架的回测过拟合概率》(2019.06.17)

    通过对回测结果进行组合对称交叉验证(Combined Symmetric Cross Validation)
    计算回测过拟合概率(Probability of Backtest Overfitting)。
    """

    def __init__(self, n_splits: int = 10, n_comparisons: int = 1000):
        self.n_splits = n_splits
        self.n_comparisons = n_comparisons

    def compute_pbo(self, returns_matrix: np.ndarray) -> float:
        """计算回测过拟合概率 (PBO)

        Args:
            returns_matrix: 形状为 (n_strategies, n_periods) 的收益率矩阵

        Returns:
            PBO: 过拟合概率 [0, 1]
        """
        n_strats, n_periods = returns_matrix.shape
        np.random.seed(42)

        # 生成随机样本路径
        prob_overfit = 0
        n_valid = 0

        for _ in range(self.n_comparisons):
            # 随机分配样本到训练集和测试集
            perm = np.random.permutation(n_periods)
            n_train = n_periods // 2
            train_idx = perm[:n_train]
            test_idx = perm[n_train:]

            # 训练集：选择最优策略
            train_returns = returns_matrix[:, train_idx]
            train_sharpes = np.mean(train_returns, axis=1) / (
                np.std(train_returns, axis=1) + 1e-10
            )
            best_train = np.argmax(train_sharpes)

            # 测试集：计算最优策略的表现
            test_returns = returns_matrix[:, test_idx]
            test_sharpes = np.mean(test_returns, axis=1) / (
                np.std(test_returns, axis=1) + 1e-10
            )

            # 检查最优策略在测试集中的排名
            rank_in_test = np.sum(test_sharpes > test_sharpes[best_train])

            if train_sharpes[best_train] > 0:
                is_overfit = rank_in_test > n_strats // 2
                if is_overfit:
                    prob_overfit += 1
                n_valid += 1

        return prob_overfit / max(n_valid, 1)


if __name__ == "__main__":
    import numpy as np

    # FFScore 演示
    mock_data = pl.DataFrame(
        {
            "close": 10 + np.random.randn(200).cumsum(),
            "roa": np.random.randn(200) * 0.01 + 0.05,
            "cfo": np.random.randn(200) * 0.01 + 0.03,
            "lev": np.random.rand(200) * 0.5,
            "liquid": np.random.rand(200) * 2 + 1,
            "margin": np.random.randn(200) * 0.02 + 0.15,
            "turnover": np.random.randn(200) * 0.1 + 0.8,
        }
    )

    ffscore = FFScoreFactor()
    values = ffscore.calculate(mock_data)

    print("=" * 50)
    print("华泰 FFScore 模型")
    print("=" * 50)
    print(f"FFScore (最后10个): {values.tail(10).to_numpy()}")
    print(f"均值: {values.mean():.3f}, 标准差: {values.std():.3f}")

    # CSCV 演示
    print("\n" + "=" * 50)
    print("CSCV 过拟合概率测试")
    print("=" * 50)
    cscv = CSCCVFramework(n_splits=5, n_comparisons=500)
    mock_returns = np.random.randn(20, 100) * 0.02
    pbo = cscv.compute_pbo(mock_returns)
    print(f"回测过拟合概率 (PBO): {pbo:.1%}")
