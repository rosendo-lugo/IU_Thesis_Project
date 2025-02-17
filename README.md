# Predicting Housing Market Trends Using Data Science

[ [Project Scope](#project-scope) ]  
[ [Research Plan](#research-plan) ]  
[ [Research Milestone Timeline](#research-milestone-timeline) ]  
[ [Acquisition Phase](#acquisition-phase) ]  
[ [Preparation Phase](#preparation-phase) ]  
[ [Exploration Phase](#exploration-phase) ]  
[ [Modeling Phase](#modeling-phase) ]  
[ [Conclusion](#conclusion) ]  
[ [Steps to Reproduce](#steps-to-reproduce) ]


**Student’s Name:** Rosendo Lugo Jr.

**Advisor:** Dr. Sridhar Ramachandran

**Source Link:** <a href="https://www.redfin.com/news/data-center/" target="_blank">Redfin Monthly Housing Market Data</a>   

**Potential Insights:** This dataset provides insights into housing market trends across different regions. I chose this dataset because housing market data reflects economic patterns, consumer behavior, and financial stability. By analyzing price fluctuations, sales trends, and inventory changes, I aim to identify key indicators that predict market shifts. The structured approach from Kaggle competitions will be applied to extract meaningful patterns, focusing on how market dynamics evolve over time.     

Additionally, the dataset's inclusion of multiple region types allows for comparisons between national trends and city-specific behaviors, providing an opportunity to explore how different real estate markets respond to economic factors. By leveraging machine learning techniques learned from the competitions, I will analyze potential leading indicators of price changes and market cycles.

---
## Project Scope
**Objectives:** Identify key trends and patterns in the housing market, predict market fluctuations, and provide actionable insights into real estate trend      

**Deliverables:** Charts, graphs, statistical reports, predictive models, and a final research paper detailing insights and methodologies.      

Milestones:      
- Data collection and cleaning (Week 1)    
- Exploratory Data Analysis (Weeks 2-3)    
- Model selection and training (Weeks 4-6)     
- Model validation and optimization (Weeks 7-8)    
- Final report and presentation (Weeks 9-10)     

Tasks:       
- Data acquisition and preprocessing      
- Exploratory data analysis (EDA)      
- Feature engineering and selection      
- Model training and validation      
- Performance evaluation and optimization     
- Presentation of findings      

**Resources:** Python, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Jupyter Notebook  
    
---
### Research Plan
**Techniques and Methods:** Time-series analysis, regression models, machine learning techniques, and cross-validation methods.
Application: Apply time-series forecasting models to predict housing price trends, use feature engineering techniques to identify key drivers of price fluctuations, and validate results using robust evaluation metrics.      

**Hypothesis**
- **Hypothesis Statement:** Housing market trends can be predicted by analyzing historical price fluctuations, inventory changes, and sales patterns. By applying machine learning techniques, we can identify early indicators of market shifts, allowing for better-informed real estate investment decisions.

[Back to Top](#predicting-housing-market-trends-using-data-science)
    
---
### Research Milestone Timeline
<details>
  <summary>Show Research Milestone Timeline</summary>
    
#### **Week 1:**    
- **Tasks:**  
  - Collect and download the dataset.  
  - Perform initial data cleaning and preprocessing.  
- **Goal:** Establish a clean and structured dataset ready for exploratory analysis.  

#### **Week 2:**    
- **Tasks:**  
  - Conduct exploratory data analysis (EDA).  
  - Generate visualizations to understand market trends.  
- **Goal:** Identify key patterns and trends in housing data.  

#### **Week 3:**    
- **Tasks:**  
  - Perform feature engineering.  
  - Handle missing values and outliers.  
- **Goal:** Prepare a well-structured dataset for modeling.  

#### **Week 4:**   
- **Tasks:**  
  - Select baseline models for initial testing.  
  - Train and evaluate simple models.  
- **Goal:** Identify the best initial modeling approach.  

#### **Week 5:**    
- **Tasks:**  
  - Experiment with advanced models, including ensemble learning.  
  - Tune hyperparameters for improved performance.  
- **Goal:** Optimize predictive models for accuracy and robustness.  

#### **Week 6:**    
- **Tasks:**  
  - Implement time-series forecasting techniques.  
  - Compare results against baseline models.  
- **Goal:** Refine models based on forecasting effectiveness.  

#### **Week 7:**    
- **Tasks:**  
  - Validate models using cross-validation techniques.  
  - Ensure stability and reliability of predictions.  
- **Goal:** Improve generalizability of the final models.  

#### **Week 8:**    
- **Tasks:**  
  - Interpret model predictions.  
  - Extract actionable insights from the analysis.  
- **Goal:** Understand key factors driving housing market trends.  

#### **Week 9:**    
- **Tasks:**  
  - Finalize all results and conclusions.  
  - Start drafting the final report.  
- **Goal:** Synthesize research findings into a structured format.  

#### **Week 10:**    
- **Tasks:**  
  - Complete and refine the final report.  
  - Prepare any necessary visual presentations.  
- **Goal:** Submit a well-documented research project with clear insights.  

</details>

---
### Potential Scope Creep
- Challenges and Mitigation Strategies: 
    - Data quality issues → Implement thorough preprocessing and validation checks.       
    - Model complexity → Prioritize interpretability over excessive optimization.       
    - Expanding scope → Stick to predefined research objectives and avoid unnecessary additions.  
    
[Back to Top](#predicting-housing-market-trends-using-data-science)

---
    
## Acquisition Phase
- Data acquired from <a href="https://www.redfin.com/news/data-center/" target="_blank">Redfin Monthly Housing Market Data</a>
- 1099 rows x 15 columns
    
[Back to Top](#predicting-housing-market-trends-using-data-science)
    
---
## Preparation Phase
- Clean the data
    - Rename columns
        - lowercase
        - add a hyphen inbetween words
    - Fixed data type
        - Object to Int/float
        - Object to DataTime
    - Remove extra spaces/characters
        - remove character like... ex %, $, K, ","
        - remove space before and after the name
    - Remove nulls
        - 0% null
    
[Back to Top](#predicting-housing-market-trends-using-data-science)
    
---
## Exploration Phase

    
[Back to Top](#predicting-housing-market-trends-using-data-science)
    
---
## Modeling Phase


[Back to Top](#predicting-housing-market-trends-using-data-science)
    
---
## Conclusion


[Back to Top](#predicting-housing-market-trends-using-data-science)
    
---
## Steps to Reproduce

To set up the environment and run this project, follow these steps:

1. Clone the Repository
- Copy the SSH link from GitHub.
- Run the following command in your terminal:
    - git clone <SSH link/>
    - cd IU_Thesis_Project
2. Quick Run
This method uses a pre-cleaned dataset for quick execution.
- Install dependencies: pip install pandas-gbq
- Ensure your Python libraries are updated (Last update: Feb 2025).
- Open thesis_project.ipynb and verify the following function is present in the cell after imports: 
    redfin_df = check_file_exists_gbq("data.csv", "service-account-key.json", "iu-thesis-project.Redfin_Monthly_Housing_Market_Data.Redfin")
- Run thesis_project.ipynb.
- This will retrieve the dataset from Google BigQuery and load it.
- The dataset is ~630KB, so processing should be fast.
3. Set Up GitHub Authentication
- Generate a GitHub Personal Access Token:
    - Go to GitHub Token Settings.
    - Generate a token without selecting any scopes (leave all checkboxes unchecked).
- Save credentials in env.py:
    - Add the following variables:
        - github_token = "your_github_token_here"
        - github_username = "your_github_username_here"
- Ensure the REPOS list is in acquire.py, either by:
    - Manually adding repo names to the list, or
    - Using repo_list.txt.
4. Open the Jupyter Notebook
- Run Jupyter Lab (or equivalent environment):
    jupyter lab
5.  Run the Notebook
- Execute all cells in thesis_project.ipynb to complete data processing.
    
[Back to Top](#predicting-housing-market-trends-using-data-science)