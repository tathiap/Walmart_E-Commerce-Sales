# Walmart E-Commerce Customer Behavior Dashboard

This repository features a complete data analytics dashboard and deep-dive analysis of Walmart’s e-commerce customer data. It combines:
* **Exploratory Data Analysis (EDA)**
*  **A/B Testing**
* **Customer Segmentation**
* **Time Series Forecasting**
* **Predictive Modeling**
  
into an interactive **Streamlit dashboard** and Jupyter based analysis.
---

## **Project 1: Walmart E-Commerce Dashboard**

### **Overview**
Interactive Streamlit app (`Walmart_Dashboard.py`) enabling teams to:
* Explore purchase behavior with demographic breakdowns
* Run simulated A/B tests
* Segment customers via K-Means clustering
* Forecast revenue with Prophet
* Predict high value customers with Logistic Regression
---

## **Project 2: Beyond the Checkout – What Drives Walmart Shoppers?**

**Objectives**
* Segment customers into actionable personas
* Test interventions with robust statistical A/B methods
* Forecast short term revenue trends
* Predict high value customers with logistic regression + Bayesian modeling
---

## Time Series Forecasting
We compared multiple models:
* **ARIMA** → Strong at capturing long-term autoregressive trends
* **Prophet** → Captured weekly seasonality and promotional spikes
* **Rolling Forecast Windows** → Validated stability over time

**Metrics Evaluated:** RMSE, MAE, sMAPE  
**Result:** Prophet produced more reliable short-term forecasts, while ARIMA gave interpretable long-term trend insights.
---

## Customer Segmentation
- K-Means identified 3 personas:  
  1. **Engaged Power Shoppers**  
  2. **One Time Big Spenders**  
  3. **Casual Browsers**  

Segmentation informed recommendations for targeted marketing strategies.
---

## A/B Testing
- Verified impact of promotional campaigns (t-tests, confidence intervals, effect size)  
- Results: No significant uplift detected (p > 0.05)  

---
##  Predictive Modeling
- **Logistic Regression** + Bayesian Inference for predicting high-value customers  
- Evaluation: Accuracy, Precision, Recall, ROC-AUC  
- Feature importance visualized interactively  

---
## **How to Run the Dashboard**
Clone the repository:
   ```bash
   git clone https://github.com/tathiap/Walmart-E---Commerce-Sales.git
   cd Walmart-E---Commerce-Sales ``` 

---
## Tools & Tech
* **Languages**: Python, SQL  
* **Libraries**: pandas, matplotlib, seaborn, scikit-learn, Prophet, statsmodels, pymc3  
* **Visualization**: Streamlit, Jupyter  
* **Data**: Walmart e-commerce customer dataset  

---

## Key Insights
* Repeat purchase rates decline sharply after Month 1 → retention is a major growth opportunity.  
* High value customers skew younger with higher cart completion rates.  
* Seasonal spikes in November/December drive significant revenue lift → promotions should focus there.  
