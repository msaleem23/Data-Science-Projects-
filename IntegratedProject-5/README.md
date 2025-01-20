# IntegratedProject-5: Video Game Sales Analysis

## 📜 Project Overview  
This project analyzes video game sales data provided by the online store Ice, which sells video games globally. The dataset includes user and expert reviews, genres, platforms, and regional sales. The goal is to identify patterns that determine a game’s success, enabling better forecasting and advertising campaign planning for the year 2017.  

---

## 📂 Dataset Description  

The dataset (`games.csv`) contains the following columns:  
- **Name**: Name of the game.  
- **Platform**: Platform on which the game was released (e.g., Xbox, PlayStation).  
- **Year_of_Release**: Year the game was released.  
- **Genre**: Genre of the game (e.g., Action, Sports).  
- **NA_sales**: North American sales (in USD million).  
- **EU_sales**: European sales (in USD million).  
- **JP_sales**: Japanese sales (in USD million).  
- **Other_sales**: Sales in other regions (in USD million).  
- **Critic_Score**: Game review score by critics (maximum: 100).  
- **User_Score**: Game review score by users (maximum: 10).  
- **Rating**: ESRB rating (e.g., Teen, Mature).  

---

## 🚀 Project Objectives  

1. **Data Preprocessing**:  
   - Clean and standardize the dataset (e.g., lowercase column names, convert data types).  
   - Handle missing values and invalid entries (e.g., `TBD`).  
   - Add a new column for total global sales.  

2. **Exploratory Data Analysis (EDA)**:  
   - Analyze release trends across years and platforms.  
   - Identify leading platforms and genres globally and regionally.  
   - Explore the impact of user/critic reviews and ESRB ratings on sales.  

3. **Hypothesis Testing**:  
   - Test if average user ratings of Xbox One and PC platforms are the same.  
   - Test if average user ratings for Action and Sports genres differ.  

4. **Insights and Recommendations**:  
   - Summarize findings to guide advertising and product planning strategies.  

---

## 📊 Analysis Workflow  

### Step 1: Data Preparation  
- Standardize column names.  
- Handle missing and invalid data.  
- Create a new column for total global sales.  

### Step 2: Exploratory Data Analysis  
- Analyze sales trends by platform and year.  
- Identify top genres and platforms in North America, Europe, and Japan.  
- Evaluate the correlation between reviews and sales.  

### Step 3: Hypothesis Testing  
- Formulate and test hypotheses about user ratings.  
- Draw conclusions based on statistical analysis.  

### Step 4: Insights and Recommendations  
- Summarize actionable insights for 2017 campaign planning.  

---

## 🛠 Tools and Libraries Used  
- **Programming Language**: Python  
- **Libraries**:  
  - `pandas`: Data manipulation and preprocessing.  
  - `matplotlib` and `seaborn`: Data visualization.  
  - `scipy.stats`: Hypothesis testing.  

---

## 📄 File Structure  
- **`games.csv`**: Dataset containing video game sales data.  
- **`video_games_analysis.ipynb`**: Jupyter
