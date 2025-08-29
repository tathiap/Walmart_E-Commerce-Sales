# Walmart E-Commerce Customer Behavior Dashboard

This repository features a complete data analytics dashboard and deep-dive analysis of Walmart’s e-commerce customer data. It combines:
* **Exploratory Data Analysis (EDA)**
*  **A/B Testing**
* **Customer Segmentation**
* **Time Series Forecasting**
* **Predictive Modeling**
*  **Streamlit dashboard** and Jupyter based analysis.
---

## **Project 1: Walmart E-Commerce Dashboard**

### **Overview**
The Walmart Dashboard (Walmart_Dashboard.py) is a fully interactive Streamlit application designed to uncover actionable insights from Walmart's e-commerce data.
It enables business and data teams to:

* Explore purchase behavior with demographic breakdowns
* Run A/B tests on control vs test groups
* Segment customers using K-Means clustering
* Forecast future revenue using Prophet & ARIMA
* Predict high-value customers with Logistic Regression

**Key Features**

* EDA: Visualizes purchase distributions, customer trends, and summary statistics
* A/B Testing: Simulates control/test groups, computes t-tests, confidence intervals, and effect sizes
* Segmentation: Clusters customers into personas using K-Means (Engaged Power Shoppers, One-Time Big Spenders, Window Shoppers, Luxury Browsers)
* Forecasting: Provides daily revenue forecasts with Prophet & ARIMA
* Predictive Modeling: Identifies high spenders with Logistic Regression and Bayesian inference
---

## **Project 2: Beyond the Checkout – What Drives Walmart Shoppers?**

**Objectives**
* Segment customers into actionable personas for targeted marketing
* Evaluate interventions using robust A/B testing
* Forecast revenue trends for operational planning
* Predict high-value customers with logistic regression & Bayesian modeling
  
**A/B Testing Findings**

* Descriptive Statistics: The control group had an average purchase of 9,204.97, while the test group averaged 9,186.29, with nearly identical medians and standard deviations. At a surface level, no major behavioral differences were observed.
* T-Test: t = -1.3993, p = 0.1617 → not statistically significant. The null hypothesis could not be rejected.
* Effect Size: Cohen’s d = -0.004 → negligible, confirming no practical impact of the experimental change.
* 95% Confidence Interval: [-44.85, 7.49] includes zero, further reinforcing that any difference in means is minimal and not meaningful.

***Interpretation***: Despite the large sample size, both statistical and practical evidence show that the treatment had no real effect on purchase behavior.


**Cohort & Retention Analysis**

Cohort analysis grouped customers by their first purchase month and tracked retention/revenue over time.

Findings

* Retention Heatmap: Strong Month-0 purchases followed by a steep drop in Month-1. Beyond Month-3, retention stabilizes at lower levels.
* Revenue Heatmap: Revenue was front loaded in early months, with later months contributing materially less. Certain cohorts (likely promo driven) sustained stronger early retention.
* Headline Metrics: Month-1 retention dropped sharply across all cohorts; Month-3 and Month-6 retention remained low, highlighting the challenge of long term engagement.

***Interpretation***: Walmart experiences front loaded customer value, emphasizing the need for better retention strategies beyond the first purchase window.

**Segmentation & Segment-Based A/B Testing**
K-Means clustering identified four personas:

* Engaged Power Shoppers
* One Time Big Spenders
* Window Shoppers
* Luxury Browsers

Segment based A/B testing compared control/test purchases within each persona:

* Overall results mirrored the global test (no significant lift).
* High spending segments (Luxury Browsers, Power Shoppers) consistently exhibited higher spend levels, suggesting opportunities for targeted future experimentation.


**Forecasting & Time Series Analysis** 

This section modeled daily revenue trends to support forward-looking business decisions.

* Prophet decomposed the series into trend, weekly, and daily components, showing clear Monday spikes and intra-day shopping cycles. It produced a reliable 14-day forecast.
* ARIMA (1,1,1) provided an alternative autoregressive forecast, capturing temporal dynamics and producing a 14-day baseline projection with confidence intervals.

***Interpretation***: Together, Prophet and ARIMA equip Walmart with interpretable and robust demand forecasts, useful for inventory management, marketing, and logistics planning.

**Predictive Modeling**
* Logistic Regression: Accurately classified high value customers. ROC AUC = 1.00, showing excellent separation, though additional validation is needed to confirm generalization and rule out overfitting.
* Bayesian Logistic Regression: Reinforced findings while quantifying uncertainty around feature importance. Priors enabled probabilistic inference, with purchase volume and cart completion rate emerging as strong drivers of customer value.

***Interpretation*** : Logistic models were highly effective at identifying high-value customers. The Bayesian approach provided richer, uncertainty-aware insights that improve interpretability and risk-aware decision-making.

**Conclusion**

This project delivered a comprehensive analysis of Walmart’s e-commerce customer behavior through a full data science workflow.

* Segmentation uncovered four distinct personas, offering opportunities for personalization.
* A/B Testing showed no statistically or practically significant uplift, though certain segments warrant future targeted testing.
* Forecasting with Prophet & ARIMA provided reliable short term revenue projections to support operational planning.
* Predictive Modeling identified high value customers with high accuracy, reinforced by Bayesian modeling’s uncertainty quantification.

***Overall takeaway***: By combining segmentation, experimentation, forecasting, and predictive analytics, Walmart can turn raw transactional data into strategic insights, enabling deeper personalization, smarter marketing investment, and stronger demand planning for sustained revenue growth.


---
## Tools & Tech
- **Languages**: Python 
- **Libraries**: pandas, numpy, matplotlib, seaborn, scikit-learn, statsmodels, Prophet, PyMC  
- **Visualization & Apps**: Streamlit, Jupyter Notebooks  
- **Techniques**: A/B Testing, Cohort Analysis, K-Means Clustering, Forecasting (Prophet/ARIMA), Predictive Modeling (Logistic & Bayesian Regression)  
- **Data**: Walmart e-commerce customer dataset (synthetic/simulated for analysis)  

---
## Key Insights
- Customer retention **drops sharply after Month 1**, stabilizing at single-digit levels by Month 6 → early lifecycle engagement is the biggest growth opportunity.  
- **Luxury Browsers** and **Power Shoppers** consistently show higher absolute spend, making them prime targets for future campaigns despite no global A/B uplift.  
- Prophet forecasts revealed **weekly patterns** (peaks on Mondays, troughs on weekends) and **intra-day cycles**, guiding inventory and marketing timing.  
- Logistic and Bayesian models identified **cart completion rate** and **purchase volume** as the strongest predictors of customer value.  

---
## **How to Run the Dashboard**
Install dependencies:
 pip install -r requirements.txt
 
Run the Streamlit dashboard:
streamlit run Walmart_Dashboard.py

Clone the repository:
   ```bash
   git clone https://github.com/tathiap/Walmart-E---Commerce-Sales.git
   cd Walmart-E---Commerce-Sales ```
