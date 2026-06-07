import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set up page config
st.set_page_config(
    page_title="Synthetic Linear Regression & Outliers",
    layout="wide"
)

st.title("Synthetic Linear Regression & Outlier Detector")
st.write(
    "This application generates synthetic linear data based on user-defined parameters, "
    "fits a linear regression model, identifies the top 10 furthest outliers (largest residuals), "
    "and visualizes the results."
)

# Initialize a session state run ID to allow generating new random datasets with the same seed slider value.
if "run_id" not in st.session_state:
    st.session_state.run_id = 0

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Data Parameters")

# Number of Points (N)
N = st.sidebar.slider(
    "Number of Points (N)",
    min_value=50,
    max_value=1000,
    value=200,
    step=10
)

# Variance of the noise
variance = st.sidebar.slider(
    "Variance",
    min_value=0,
    max_value=300000,
    value=100000,
    step=1000
)

# True Slope (a)
true_a = st.sidebar.slider(
    "True Slope (a)",
    min_value=-50,
    max_value=50,
    value=-15,
    step=1
)

# True Intercept (b)
true_b = st.sidebar.slider(
    "True Intercept (b)",
    min_value=0,
    max_value=100,
    value=50,
    step=1
)

# Random Seed
base_seed = st.sidebar.slider(
    "Random Seed",
    min_value=0,
    max_value=9999,
    value=42,
    step=1
)

# Button to generate a new dataset (it increments our state modifier to change the actual random seed)
if st.sidebar.button("Generate New Dataset"):
    st.session_state.run_id += 1

# Calculate the actual seed used for generation (combines base_seed with the button click modifier)
actual_seed = (base_seed + st.session_state.run_id) % 10000

# --- DATA GENERATION ---
# Seed the random number generator
np.random.seed(actual_seed)

# Generate independent variable x ~ Uniform(-100, 100)
x = np.random.uniform(-100, 100, N)

# Noise standard deviation is the square root of the variance
std_dev = np.sqrt(variance)
noise = np.random.normal(0, std_dev, N)

# Dependent variable y = a*x + b + noise
y = true_a * x + true_b + noise

# Package data into a Pandas DataFrame for easy manipulation
df = pd.DataFrame({
    'Index': np.arange(N),
    'x': x,
    'y': y
})

# --- FIT LINEAR REGRESSION ---
X_fit = df[['x']]
y_fit = df['y']

model = LinearRegression()
model.fit(X_fit, y_fit)

# Extract estimated parameters
estimated_a = model.coef_[0]
estimated_b = model.intercept_

# Make predictions
df['Predicted_Y'] = model.predict(X_fit)

# --- CALCULATE RESIDUALS & IDENTIFY OUTLIERS ---
# Residual = |y - predicted_y|
df['Residual'] = np.abs(df['y'] - df['Predicted_Y'])

# Find Top 10 Outliers with the largest residuals
# Rank them #1 to #10 (where #1 is the largest residual)
top_outliers = df.nlargest(10, 'Residual').copy()
top_outliers['Rank'] = np.arange(1, 11)

# --- PLOTTING ---
fig, ax = plt.subplots(figsize=(12, 8))

# 1. Blue scatter points for all data
ax.scatter(
    df['x'], 
    df['y'], 
    color='#1f77b4', 
    alpha=0.6, 
    edgecolors='none', 
    label='Data Points'
)

# 2. Red regression line
x_line = np.linspace(-100, 100, 200)
y_line = estimated_a * x_line + estimated_b
ax.plot(
    x_line, 
    y_line, 
    color='red', 
    linewidth=2.5, 
    label=f'Regression Line (y = {estimated_a:.2f}x + {estimated_b:.2f})'
)

# 3. Orange circles for top 10 outliers with a black edge
ax.scatter(
    top_outliers['x'], 
    top_outliers['y'], 
    color='orange', 
    edgecolors='black', 
    s=120, 
    linewidths=1.5,
    zorder=5, 
    label='Top 10 Outliers'
)

# 4. Label each outlier with its rank (#1, #2, ..., #10)
for idx, row in top_outliers.iterrows():
    ax.annotate(
        f"#{int(row['Rank'])}", 
        (row['x'], row['y']),
        textcoords="offset points", 
        xytext=(0, 10), 
        ha='center', 
        fontsize=10, 
        fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.7, ec="black", lw=0.5)
    )

# Chart labels, titles and formatting
title_text = (
    "Synthetic Linear Regression & Top 10 Outliers\n"
    f"True a = {true_a}, True b = {true_b} | "
    f"Estimated a = {estimated_a:.4f}, Estimated b = {estimated_b:.4f} | "
    f"Variance = {variance}"
)
ax.set_title(title_text, fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("X (Independent Variable)", fontsize=11)
ax.set_ylabel("Y (Dependent Variable)", fontsize=11)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='best')

# Adjust layout and show using Streamlit
plt.tight_layout()
st.pyplot(fig)

# --- METRICS SECTION ---
st.subheader("Model Performance Metrics")

# Compute mathematical metrics
actual_residual_variance = np.var(df['y'] - df['Predicted_Y'])
rmse = np.sqrt(mean_squared_error(df['y'], df['Predicted_Y']))
r2 = r2_score(df['y'], df['Predicted_Y'])

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="True Variance", value=f"{variance:,}")
with col2:
    st.metric(label="Residual Variance", value=f"{actual_residual_variance:,.2f}")
with col3:
    st.metric(label="RMSE", value=f"{rmse:,.4f}")
with col4:
    st.metric(label="R² Score", value=f"{r2:.4f}")

# --- OUTLIER TABLE SECTION ---
st.subheader("Top 10 Outliers Table")

# Format and select columns to match instructions: Rank, Index, x, y, Predicted_Y, Residual
outliers_display = top_outliers[['Rank', 'Index', 'x', 'y', 'Predicted_Y', 'Residual']].reset_index(drop=True)

# Render the dataframe in Streamlit
st.dataframe(outliers_display, use_container_width=True)