# crop_trend_prediction_app_linear.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Page config
st.set_page_config(page_title="🌾 Crop Production Trends & Prediction (Linear Regression)", layout="wide")
st.title("🌱 Crop Production Trends & Prediction App (Linear Regression Version)")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\Admin\Downloads\crop2.csv", index_col=0)
    return df

df = load_data()

# Data Cleaning
df = df[df['flag_description'] == 'Official figure']

# Pivot Data
df_pivot = df.pivot_table(index=['area', 'item', 'year'], 
                          columns='element', 
                          values='value').reset_index()

# Drop missing rows
df_pivot = df_pivot.dropna(subset=['Area harvested', 'Yield', 'Production'])

# Sidebar filters
st.sidebar.header("Filter Options")
selected_country = st.sidebar.selectbox("Select Country", df_pivot['area'].unique())
selected_crop = st.sidebar.selectbox("Select Crop", df_pivot['item'].unique())

# Filtered Data
filtered_df = df_pivot[(df_pivot['area'] == selected_country) & (df_pivot['item'] == selected_crop)]
filtered_df_sorted = filtered_df.sort_values('year')

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Production Trend", 
    "Yield Trend", 
    "Area Harvested Trend", 
    "Yield vs. Area", 
    "YoY Growth %", 
    "Country Summary & Top Crops"
])

# Tab 1 - Production Trend
with tab1:
    st.subheader(f"📈 Production Trend: {selected_crop} in {selected_country}")
    fig1, ax1 = plt.subplots(figsize=(12,6))
    sns.lineplot(data=filtered_df_sorted, x='year', y='Production', marker='o', ax=ax1)
    plt.xlabel("Year")
    plt.ylabel("Production (tons)")
    plt.title(f"Production Trend of {selected_crop} in {selected_country}")
    st.pyplot(fig1)

# Tab 2 - Yield Trend
with tab2:
    st.subheader(f"📈 Yield Trend: {selected_crop} in {selected_country}")
    fig2, ax2 = plt.subplots(figsize=(12,6))
    sns.lineplot(data=filtered_df_sorted, x='year', y='Yield', marker='o', ax=ax2, color='green')
    plt.xlabel("Year")
    plt.ylabel("Yield (kg/ha)")
    plt.title(f"Yield Trend of {selected_crop} in {selected_country}")
    st.pyplot(fig2)

# Tab 3 - Area Harvested Trend
with tab3:
    st.subheader(f"📈 Area Harvested Trend: {selected_crop} in {selected_country}")
    fig3, ax3 = plt.subplots(figsize=(12,6))
    sns.lineplot(data=filtered_df_sorted, x='year', y='Area harvested', marker='o', ax=ax3, color='orange')
    plt.xlabel("Year")
    plt.ylabel("Area harvested (ha)")
    plt.title(f"Area Harvested Trend of {selected_crop} in {selected_country}")
    st.pyplot(fig3)

# Tab 4 - Yield vs. Area Harvested
with tab4:
    st.subheader(f"📊 Yield vs. Area harvested: {selected_crop} in {selected_country}")
    fig4, ax4 = plt.subplots(figsize=(8,6))
    sns.scatterplot(data=filtered_df_sorted, x='Area harvested', y='Yield', hue='year', palette='viridis', s=100)
    plt.title(f"Yield vs. Area harvested for {selected_crop} in {selected_country}")
    plt.xlabel("Area harvested (ha)")
    plt.ylabel("Yield (kg/ha)")
    st.pyplot(fig4)

# Tab 5 - YoY Production Growth %
with tab5:
    st.subheader(f"📈 Year-over-Year Production Growth %: {selected_crop} in {selected_country}")
    filtered_df_sorted['YoY Growth %'] = filtered_df_sorted['Production'].pct_change() * 100
    fig5, ax5 = plt.subplots(figsize=(12,6))
    sns.barplot(data=filtered_df_sorted, x='year', y='YoY Growth %', palette='coolwarm')
    plt.axhline(0, linestyle='--', color='gray')
    plt.xlabel("Year")
    plt.ylabel("Production Growth %")
    plt.title(f"Year-over-Year Production Growth % for {selected_crop} in {selected_country}")
    st.pyplot(fig5)

# Tab 6 - Country Summary & Top Crops
with tab6:
    st.subheader(f"📊 Summary Statistics: {selected_country}")

    country_df = df_pivot[df_pivot['area'] == selected_country]
    summary_stats = country_df.groupby('item').agg({
        'Production': ['mean', 'std', 'max'],
        'Yield': ['mean', 'std', 'max'],
        'Area harvested': ['mean', 'std', 'max']
    }).reset_index()

    summary_stats.columns = ['Crop', 
                             'Production Mean', 'Production Std', 'Production Max',
                             'Yield Mean', 'Yield Std', 'Yield Max',
                             'Area Mean', 'Area Std', 'Area Max']

    st.dataframe(summary_stats.sort_values('Production Mean', ascending=False).head(10))

    # Top 5 Crops with Increasing Production Trend
    st.subheader(f"🌟 Top 5 Crops with Increasing Production Trend in {selected_country}")
    trend_df = df_pivot[df_pivot['area'] == selected_country].groupby(['item', 'year'])['Production'].mean().reset_index()

    slopes = []
    for crop in trend_df['item'].unique():
        temp = trend_df[trend_df['item'] == crop]
        if len(temp['year'].unique()) >= 2:
            slope = (temp['Production'].values[-1] - temp['Production'].values[0]) / (temp['year'].values[-1] - temp['year'].values[0])
            slopes.append((crop, slope))

    slopes_sorted = sorted(slopes, key=lambda x: x[1], reverse=True)[:5]
    top5_df = pd.DataFrame(slopes_sorted, columns=['Crop', 'Production Trend (tons/year increase)'])
    st.table(top5_df)

# Prediction Section
st.sidebar.header("🔮 Predict Production")
X = df_pivot[['Area harvested', 'Yield']]
y = df_pivot['Production']

# Sample 10k rows to avoid MemoryError
sample_size = min(10000, len(X))
X_sample = X.sample(n=sample_size, random_state=42)
y_sample = y.loc[X_sample.index]

# Force use Linear Regression
model = LinearRegression()
model.fit(X_sample, y_sample)
model_name = "LinearRegression (forced)"

# User input
area_input = st.sidebar.number_input("Enter Area harvested (ha):", min_value=0.0, value=1000.0)
yield_input = st.sidebar.number_input("Enter Yield (kg/ha):", min_value=0.0, value=2000.0)

if st.sidebar.button("Predict Production"):
    prediction = model.predict([[area_input, yield_input]])
    st.sidebar.success(f"Estimated Production: {prediction[0]:,.2f} tons")

# Model Evaluation
with st.sidebar.expander("📊 Model Evaluation (R² and MAE)"):
    X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    st.sidebar.write(f"**Model used:** {model_name}")
    st.sidebar.write(f"**R² Score:** {r2:.2f}")
    st.sidebar.write(f"**Mean Absolute Error (MAE):** {mae:.2f} tons")


