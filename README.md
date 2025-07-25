# Walmart E-Commerce Customer Behavior Dashboard

This repository features a complete **data analytics dashboard and analysis** of Walmart’s e-commerce customer data. It combines **Exploratory Data Analysis (EDA)**, **A/B Testing**, **Customer Segmentation**, **Revenue Forecasting**, and **Predictive Modeling** into an interactive Streamlit dashboard.

---

## **Project 1: Walmart E-Commerce Dashboard**

### **Overview**
The Walmart Dashboard (`Walmart_Dashboard.py`) is a fully interactive Streamlit application designed to uncover actionable insights from Walmart's e-commerce data.  
It enables business and data teams to:
- **Explore purchase behavior** with demographic breakdowns.
- **Run A/B tests** on control vs test groups.
- **Segment customers** using **K-Means clustering**.
- **Forecast future revenue** using **Prophet**.
- **Predict high-value customers** with **Logistic Regression** and display **feature importance** interactively.

### **Key Features**
- **EDA**: Visualizes purchase distributions, customer trends, and summary statistics.  
- **A/B Testing**: Simulates control and test groups with t-tests, confidence intervals, and effect sizes.  
- **Segmentation**: Automatically clusters customers into segments using behavioral metrics (e.g., total purchases, average spend, cart completion rate).  
- **Forecasting**: Provides daily revenue forecasts with trend and seasonality components.  
- **Predictive Modeling**: Identifies high spenders with adjustable thresholds, confusion matrices, and feature importance charts.

---

## **Project 2: Beyond the Checkout – What Drives Walmart Shoppers?**

### **Introduction**
This project provides a **deep dive analysis of Walmart’s e-commerce customers** to uncover purchase drivers, behavioral segments, and revenue trends. It builds on the dashboard by applying additional statistical modeling and advanced insights.

### **Objectives**
- **Segment customers** into actionable personas for targeted marketing.
- **Evaluate interventions** using robust A/B testing.
- **Forecast revenue trends** for operational planning.
- **Predict high-value customers** with logistic regression and Bayesian modeling.

### **Highlights**
- **K-Means Segmentation**: Identified shopper personas (e.g., Engaged Power Shoppers, One-Time Big Spenders).  
- **A/B Testing**: Verified that promotional campaigns showed no significant uplift (p > 0.05).  
- **Forecasting**: Prophet and ARIMA provided reliable 2 week revenue forecasts.  
- **Predictive Modeling**: Achieved 99% accuracy in classifying high value customers, with probabilistic insights from Bayesian inference.

---

## **How to Run the Dashboard**
Clone the repository:
   ```bash
   git clone https://github.com/tathiap/Walmart-E---Commerce-Sales.git
   cd Walmart-E---Commerce-Sales
