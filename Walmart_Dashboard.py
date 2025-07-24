# WALMART DASHBOARD

#Libraries 
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import plotly.express as px

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("walmart.csv")

df = load_data()

# Sidebar navigation
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Go to:",
    ["EDA", "A/B Testing", "Segmentation", "Forecasting", "Predictive Modeling"]
)

st.title("E-Commerce Customer Insights Dashboard")
st.write("**Column Names:**", df.columns.tolist())

# Summary Section 

if menu == "Summary":
    st.subheader("E-Commerce Dashboard Summary")

    # KPIs
    total_customers = df['User_ID'].nunique()
    total_revenue = df['Purchase'].sum()
    avg_purchase = df['Purchase'].mean()
    top_category = df.groupby('Product_Category')['Purchase'].sum().idxmax()

    st.write("### Key Metrics")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Customers", f"{total_customers:,}")
    kpi2.metric("Total Revenue", f"${total_revenue:,.0f}")
    kpi3.metric("Avg Purchase", f"${avg_purchase:,.2f}")
    kpi4.metric("Top Category", f"{top_category}")

    # Top 5 Product Categories
    st.write("### Top 5 Product Categories by Revenue")
    top_categories = df.groupby('Product_Category')['Purchase'].sum().sort_values(ascending=False).head(5)
    st.bar_chart(top_categories)

    # Gender Distribution
    st.write("### Gender Distribution")
    gender_dist = df.groupby('Gender')['User_ID'].nunique()
    fig, ax = plt.subplots()
    ax.pie(gender_dist, labels=gender_dist.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    # Daily Revenue Trend
    st.write("### Revenue Trend (Last 30 Days)")
    if 'checkout_time' in df.columns:
        revenue_trend = df.groupby(df['checkout_time'].dt.date)['Purchase'].sum().tail(30)
        st.line_chart(revenue_trend)
    else:
        st.info("Revenue trend requires 'checkout_time' column or simulated dates.")

# EDA Section

if menu == "EDA":
    st.subheader("Exploratory Data Analysis")

    st.write("**Dataset Overview:**")
    st.write(df.head())

    st.write("**Summary Statistics:**")
    st.write(df.describe())

    st.write("**Purchase Distribution:**")
    fig, ax = plt.subplots()
    sns.histplot(df['Purchase'], bins=30, kde=True, ax=ax)
    ax.set_title("Purchase Distribution")
    st.pyplot(fig)



# A/B Testing Section 
    
if menu == "A/B Testing":
    st.subheader("A/B Testing")

    if 'group' not in df.columns:
        np.random.seed(42)
        df['group'] = np.random.choice(['control', 'test'], size=len(df))
        st.info("Control and Test groups generated for analysis.")

    control = df[df['group'] == 'control']['Purchase']
    test = df[df['group'] == 'test']['Purchase']

    st.write(f"**Control mean:** {control.mean():.2f}")
    st.write(f"**Test mean:** {test.mean():.2f}")

    fig, ax = plt.subplots()
    sns.histplot(df, x='Purchase', hue='group', bins=30, kde=True, ax=ax)
    ax.set_title("Purchase Distribution by Group")
    st.pyplot(fig)

    from scipy.stats import ttest_ind, norm
    t_stat, p_val = ttest_ind(test, control, equal_var=False)
    st.write(f"**T-statistic:** {t_stat:.3f}")
    st.write(f"**p-value:** {p_val:.4f}")

    mean_diff = test.mean() - control.mean()
    se_diff = np.sqrt(test.var() / len(test) + control.var() / len(control))
    ci = norm.interval(0.95, loc=mean_diff, scale=se_diff)
    st.write(f"**95% CI for mean difference:** [{ci[0]:.2f}, {ci[1]:.2f}]")



# K-Means Segmentation 
    
if menu == "Segmentation":
    st.subheader("Customer Segmentation")

    if not all(col in df.columns for col in ['total_purchases', 'avg_purchase', 'cart_completion_rate']):
        st.warning("Generating segmentation features from raw data...")
        user_df = df.groupby('User_ID').agg(
            total_purchases=('Purchase', 'count'),
            avg_purchase=('Purchase', 'mean')
        ).reset_index()
        user_df['cart_completion_rate'] = np.random.uniform(0.4, 0.9, size=len(user_df))
        df = df.merge(user_df, on='User_ID', how='left')

    features = ['total_purchases', 'avg_purchase', 'cart_completion_rate']
    X = df[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42)
    df['segment'] = kmeans.fit_predict(X_scaled)

    st.write("### Segment Profiles")
    st.write(df.groupby('segment')[features].mean().round(2))

    st.write("### Segment Summary")
    segment_summary = df.groupby('segment').agg(
        total_customers=('User_ID', 'nunique'),
        avg_spend=('avg_purchase', 'mean'),
        avg_total_purchases=('total_purchases', 'mean')
    ).reset_index()
    st.dataframe(segment_summary.style.format({'avg_spend': '{:.2f}', 'avg_total_purchases': '{:.0f}'}))

    st.write("**Customer Segments Visualization:**")
    fig = px.scatter(
        df, x='total_purchases', y='avg_purchase', color='segment',
        title="Customer Segments (K-Means)",
        labels={"total_purchases": "Total Purchases", "avg_purchase": "Average Purchase"}
    )
    st.plotly_chart(fig)

    
# Forecasting Section
    
if menu == "Forecasting":
    st.subheader("Revenue Forecasting")

    if 'checkout_time' not in df.columns:
        date_range = pd.date_range(start="2021-01-01", periods=365, freq='D')
        df['checkout_time'] = np.random.choice(date_range, size=len(df))
        st.info("Simulated checkout dates generated for forecasting.")

    ts_df = df.groupby(df['checkout_time'].dt.date)['Purchase'].sum().reset_index()
    ts_df.columns = ['ds', 'y']
    ts_df['ds'] = pd.to_datetime(ts_df['ds'])

    st.write("### Daily Revenue (Last 14 Days)")
    last_14_days = ts_df.tail(14).copy()
    last_14_days['ds'] = last_14_days['ds'].dt.strftime('%Y-%m-%d')
    st.dataframe(last_14_days.rename(columns={'ds': 'Date', 'y': 'Revenue'}))

    model = Prophet(daily_seasonality=True)
    model.fit(ts_df)

    future = model.make_future_dataframe(periods=14)
    forecast = model.predict(future)

    st.write("**Forecast Data (Last 10 Days):**")
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))

    st.write("### Forecast Plot")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    st.write("### Trend and Seasonality Components")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)


# User-Level Aggregation 
    
if {'total_purchases', 'avg_purchase', 'cart_completion_rate'}.isdisjoint(df.columns):
    st.info("Generating user-level metrics...")

    # Group by user and calculate metrics
    user_summary = df.groupby('User_ID').agg(
        total_purchases=('Purchase', 'count'),
        avg_purchase=('Purchase', 'mean'),
        total_spent=('Purchase', 'sum')
    ).reset_index()

    # Simulate cart_completion_rate (if not in original dataset)
    user_summary['cart_completion_rate'] = np.random.uniform(0.4, 0.8, size=len(user_summary))

    # Merge back into df
    df = df.merge(user_summary, on='User_ID', how='left')



# Predictive Modeling Section
    
if menu == "Predictive Modeling":
    st.subheader("High Spender Prediction")

    # Threshold Slider
    threshold = st.slider(
        "Set High Spender Threshold:",
        min_value=int(df['Purchase'].min()),
        max_value=int(df['Purchase'].max()),
        value=int(df['Purchase'].quantile(0.75)),
        key="high_spender_threshold"
    )
    df['high_spender'] = (df['Purchase'] > threshold).astype(int)
    st.write(f"**High Spender Threshold:** {threshold:.2f}")

    # High Spender Summary (Pie Chart + Table)
    st.write("### High Spender Summary")
    spender_summary = df['high_spender'].value_counts().rename({0: 'Low Spenders', 1: 'High Spenders'})

    # Pie Chart
    fig, ax = plt.subplots()
    ax.pie(spender_summary, labels=spender_summary.index, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff9999'])
    ax.set_title("High vs. Low Spenders")
    st.pyplot(fig)

    # Summary Table
    st.dataframe(spender_summary.reset_index().rename(columns={'index': 'Category', 'high_spender': 'Count'}))

    # Ensure required features
    features = ['total_purchases', 'avg_purchase', 'cart_completion_rate']
    if not all(col in df.columns for col in features):
        st.error("Required features not found in dataset. Please run EDA/User Aggregation first.")
    else:
        # Prepare data
        X = df[features].fillna(0)
        y = df['high_spender']

        # Train logistic regression model
        model = LogisticRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        # Classification Report
        st.write("### Classification Report:")
        st.text(classification_report(y, y_pred))

        # Confusion Matrix (Table)
        cm = confusion_matrix(y, y_pred)
        st.write("### Confusion Matrix (Table)")
        st.dataframe(pd.DataFrame(
            cm,
            index=['Actual Not High', 'Actual High'],
            columns=['Predicted Not High', 'Predicted High']
        ))

        # Confusion Matrix Heatmap
        st.write("### Confusion Matrix (Heatmap)")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                    xticklabels=['Predicted Not High', 'Predicted High'],
                    yticklabels=['Actual Not High', 'Actual High'],
                    ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # Feature Importance
        importance = pd.DataFrame({
            'Feature': features,
            'Coefficient': model.coef_[0]
        }).sort_values(by='Coefficient', ascending=False)

        st.write("### Feature Importance:")
        st.bar_chart(importance.set_index('Feature'))

        # ROC Curve
        st.write("### ROC Curve")
        y_scores = model.predict_proba(X)[:, 1]
        fpr, tpr, thresholds = roc_curve(y, y_scores)
        auc = roc_auc_score(y, y_scores)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        st.pyplot(fig)