# WH4-LinearRegression

An interactive Streamlit app for exploring linear regression with synthetic data and outlier detection.

## Features

- Generate synthetic datasets with configurable sample size, noise variance, slope, and intercept
- Fit a linear regression model using scikit-learn
- Automatically identify and rank the top 10 outliers (largest residuals)
- Visualize data points, regression line, and outliers with annotated labels
- Display model performance metrics (RMSE, R², residual variance)

## Requirements

- Python 3.x
- streamlit, numpy, pandas, matplotlib, scikit-learn

## Usage

```bash
pip install streamlit numpy pandas matplotlib scikit-learn
streamlit run synthetic_linear_regression_streamlit_app.py
```

Use the sidebar controls to adjust data parameters and click **Generate New Dataset** to create a new random sample.

---

## 项目简介

WH4-LinearRegression 是一个基于 Streamlit 的交互式数据科学应用，用于探索线性回归模型与异常值检测。

## 主要功能

- **合成数据生成**：支持自定义样本数量、噪声方差、真实斜率（a）和截距（b），快速生成用于测试的线性数据集。
- **线性回归拟合**：基于 scikit-learn 对生成的数据进行线性回归建模，自动估算模型的斜率和截距。
- **异常值检测**：自动计算每个数据点的残差绝对值，并标记残差最大的前 10 个点作为异常值，按 #1 至 #10 排序。
- **可视化呈现**：使用 Matplotlib 绘制散点图、回归线，并用橙色圆圈高亮标注异常值及其排名。
- **性能指标展示**：实时显示真实方差、残差方差、均方根误差（RMSE）和决定系数（R²）。
- **异常值数据表**：以表格形式列出前 10 个异常值的详细数据，包括序号、索引、x、y、预测值和残差。

## 使用说明

在终端运行 `streamlit run synthetic_linear_regression_streamlit_app.py` 后，通过左侧边栏调整参数，点击 **Generate New Dataset** 按钮即可生成新的随机数据集并更新结果。
