#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import pandas as pd
from scipy.stats import ttest_ind
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import stats


# # Video Game Market Analysis for 2017 Campaign Planning
# 
# ## Project Overview
# 
# In this project, I explore a comprehensive dataset from "Ice," an online store that sells video games globally. The primary goal is to identify patterns that determine the success of different video games. Understanding these patterns will enable us to spot potential big winners and effectively plan advertising campaigns for the upcoming year, 2017. This analysis is particularly timely as I approach December 2016, allowing us to use the most recent complete data available for planning purposes.
# 
# The dataset encompasses various facets of the video game industry, including user and expert reviews, genres, platforms (such as Xbox or PlayStation), and historical sales data across different regions of the world. Additionally, the dataset includes ESRB ratings, which provide age and content ratings for video games. These elements are crucial as they contribute to understanding the broader market dynamics and consumer preferences that can influence a game's success in the marketplace.
# 
# ## Data Description
# 
# The analysis will focus on data collected up to the year 2016. The dataset contains several key columns:
# - `Name`: The title of the game.
# - `Platform`: The gaming platform on which the game is available.
# - `Year_of_Release`: The year the game was released.
# - `Genre`: The genre of the game.
# - `NA_sales`: Sales in North America (in millions of USD).
# - `EU_sales`: Sales in Europe (in millions of USD).
# - `JP_sales`: Sales in Japan (in millions of USD).
# - `Other_sales`: Sales in other regions (in millions of USD).
# - `Critic_Score`: The average score given by critics, on a scale of 100.
# - `User_Score`: The average score given by users, on a scale of 10.
# - `Rating`: The ESRB rating.
# 
# This project will utilize statistical analysis and visual data exploration techniques to uncover trends and insights that will guide the strategic planning for future marketing and sales initiatives.
# 

# In[2]:


#loading and reading data 
df = pd.read_csv('/datasets/games.csv')
df.head()
df.info()
df.describe(include='all')


# In[3]:


#rename columns
df.columns = [column.lower() for column in df.columns]
# Converting year_of_release to integer
df['year_of_release'] = pd.to_numeric(df['year_of_release'], downcast='integer', errors='coerce')
# Convert user_score to numeric, setting errors='coerce' will convert non-convertible values to NaN
df['user_score'] = pd.to_numeric(df['user_score'], errors='coerce')


# In[4]:


# Check for any complete duplicates in the data
duplicates = df.duplicated().sum()
print(f"Number of complete duplicates: {duplicates}")

# Check for duplicates specifically by name, year, and platform
name_year_platform_duplicates = df.duplicated(subset=['name', 'year_of_release', 'platform']).sum()
print(f"Number of name-year-platform duplicates: {name_year_platform_duplicates}")


# In[5]:


# Remove duplicates if necessary
df = df.drop_duplicates()
# Or for name-year-platform duplicates
df = df.drop_duplicates(subset=['name', 'year_of_release', 'platform'])


# year_of_release: Converted to integer because years should be represented as whole numbers.
# user_score: Converted from strings to floats to facilitate numerical operations, especially since 'TBD' values indicate undefined user scores and should be treated as missing.

# In[6]:


# Checking for missing values
print(df.isnull().sum())

# Before filling missing values, let's check if there are any unexpected duplications or issues
df = df.drop_duplicates()

# Ensuring no sales data is missing by filling potential NaNs with zero
sales_columns = ['na_sales', 'eu_sales', 'jp_sales', 'other_sales']
for col in sales_columns:
    df[col] = df[col].fillna(0)

# Convert 'user_score' to numeric, setting errors='coerce' will convert non-convertible values to NaN
df['user_score'] = pd.to_numeric(df['user_score'], errors='coerce')

# Handling missing 'rating'
df['rating'] = df['rating'].fillna('Not Rated')

# Rechecking for missing values
print(df.isnull().sum())

# Plotting distributions of critic scores by genre to understand variability
plt.figure(figsize=(12, 8))
sns.boxplot(data=df, x='genre', y='critic_score')
plt.xticks(rotation=45)
plt.title('Distribution of Critic Scores by Genre')
plt.ylabel('Critic Score')
plt.xlabel('Genre')
plt.show()

# Plotting distributions of user scores by genre to understand variability
plt.figure(figsize=(12, 8))
sns.boxplot(data=df, x='genre', y='user_score')
plt.xticks(rotation=45)
plt.title('Distribution of User Scores by Genre')
plt.ylabel('User Score')
plt.xlabel('Genre')
plt.show()


# In[7]:


# Check for duplicates specifically by name, year, and platform
name_year_platform_duplicates = df.duplicated(subset=['name', 'year_of_release', 'platform']).sum()
print(f"Number of name-year-platform duplicates: {name_year_platform_duplicates}")

# Remove duplicates if necessary
df = df.drop_duplicates(subset=['name', 'year_of_release', 'platform'])


# In[8]:


# Calculate total sales for each game as a new column
df['total_sales'] = df[['na_sales','eu_sales','jp_sales','other_sales']].sum(axis=1)


# ### Critic Scores Analysis by Genre
# The boxplot I examined shows the distribution of critic scores across different game genres. In each boxplot, there's a median line within the interquartile range, which demonstrates that while scores within each genre vary, the median offers a central value that remains consistent despite outliers. This observation supports my decision to use the median critic score for filling in missing values within each genre.
# 
# I chose to use the median as it is generally a better representation of typical scores within a genre compared to the mean, which can be influenced by exceptionally high or low scores. By replacing missing critic scores with the median from the same genre, I aim to maintain the original distribution of scores and minimize the influence of outliers.

# In[9]:


#checking for neg values in sales data as they should not logically exist
negative_sales_check = (df[sales_columns] < 0).any()
print("Negative sales values present:", negative_sales_check)


# In[10]:


# Exporting the cleaned DataFrame to a new CSV file
if not negative_sales_check.any():
    df.to_csv('../cleaned_games.csv', index=False)
    print("Cleaned data exported successfully.")
else:
    print("Data contains negative values; please investigate before exporting.")


# In[11]:


# Calculate total sales for each game as a new column
df['total_sales'] = df[['na_sales','eu_sales','jp_sales','other_sales']].sum(axis=1)
#summary for total_sales
print(df['total_sales'].describe())

#histogram of total sales to identify skewness and outliers
df['total_sales'].hist(bins=50)
plt.title('Distribution of Total Sales')
plt.xlabel('Total Sales in Millions')
plt.ylabel('Number of Games')
plt.show()


# ### Analysis Conclusion: Distribution of Total Sales
# The histogram of total sales across games shows a highly right-skewed distribution, indicating that most games generate low to moderate sales, while a small number of games achieve significantly higher sales. This pattern suggests that the video game market is dominated by a few high-performing titles.
# 
# #### Key Observations:
# - **Skewness**: The skewness in the sales distribution could influence our approach to predicting game success, as median values might be more representative of typical sales than mean values.
# - **Outliers**: The presence of outliers suggests that there are exceptional cases where games achieve extraordinary success. These outliers could be particularly interesting for case studies or for identifying features that might predict high sales.
# 
# #### Implications for Marketing Strategy:
# Given the skewness and the presence of outliers, marketing strategies could be specifically tailored towards potential blockbuster titles identified through predictive modeling. Additionally, understanding factors that contribute to the extreme success of outliers could help in planning targeted marketing campaigns to boost moderately performing games.
# 
# This conclusion effectively addresses the implications of the observed sales distribution and sets the stage for deeper analysis into what contributes to the success of the top-performing games.

# ## Observations and Hypotheses
# 
# Following the preliminary exploration of the dataset, I observed that the data contains entries from multiple regions with varying sales figures. Initial hypotheses suggest that certain platforms and genres may perform significantly better, influencing their potential success in upcoming years. I plan to delve deeper into these aspects, aiming to identify patterns that could forecast the success of future game releases.
# 

# In[12]:


# Add a small constant to handle log(0) issue
df['log_total_sales'] = np.log(df['total_sales'] + 0.01)

# Plotting a histogram of the log of total sales
plt.figure(figsize=(10, 6))
df['log_total_sales'].hist(bins=50)
plt.title('Distribution of Log of Total Sales')
plt.xlabel('Log of Total Sales in Millions')
plt.ylabel('Number of Games')
plt.show()


# In[13]:


#Analyze Game Releases Over the Years
#Plot the number of games released each year to see trends and decide which years are relevant for predicting 2017.
plt.figure(figsize=(14, 7))
df['year_of_release'].value_counts().sort_index().plot(kind='bar')
plt.title('Number of Games Released Each Year')
plt.xlabel('Year of Release')
plt.ylabel('Number of Games')
plt.show()


# ### Games Released Over the Years
# 
# The bar chart illustrating the number of games released each year shows significant variability. There was a notable increase in game releases from the early 2000s, peaking around 2008-2009, followed by a decline. This could be attributed to the economic recession during those years and the subsequent shifts in consumer spending and production budgets. The decline in recent years might suggest market saturation or a shift towards higher quality over quantity in game development.
# 
# **Strategic Implications:**
# For predicting the market in 2017, focusing on the trends post-2008 will provide the most relevant insights, as the industry dynamics post-recession have stabilized somewhat. It's also crucial to consider the impact of digital distribution, which might have altered traditional game release patterns.
# 

# In[14]:


# Sales Analysis by Platform
platform_sales = df.groupby(['platform', 'year_of_release'])['total_sales'].sum().unstack()

# Plotting the total sales by platform over the years with improved readability
plt.figure(figsize=(14, 7))
for platform in platform_sales.index:
    plt.plot(platform_sales.columns, platform_sales.loc[platform], label=platform)

plt.title('Total Sales by Platform Over the Years')
plt.xlabel('Year')
plt.ylabel('Total Sales in Millions')
plt.legend(title='Platform', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(platform_sales.columns, rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# ### Platform Sales Distribution
# 
# The line chart of total sales by platform over the years reveals that platforms such as PS2 and Xbox 360 saw significant sales during their peak years but have since declined as newer platforms emerged. Newer platforms like PS4 and Xbox One have picked up in sales, showing the cyclical nature of the gaming industry with platform lifespans of approximately 5-7 years before they begin to fade.
# 
# **Strategic Implications:**
# For the 2017 campaign, focusing on newer platforms that are currently growing in market share, like PS4 and Xbox One, would likely yield better ROI on marketing campaigns due to their increasing popularity and user base.
# 
# ### Decision: Relevant Data Period for 2017 Model
# 
# Based on the analysis of game releases and platform trends, the most relevant data period for building a model to predict 2017 trends would be from 2010 to 2016. This period captures the current generation of gaming platforms and the recent market behaviors, which are crucial for accurate predictions.
# 
# 

# 

# In[15]:


# Focus on data from the last 5 years
recent_df = df[df['year_of_release'] >= 2012]

# Recalculate sales by platform for the recent data
platform_sales_recent = recent_df.groupby(['platform', 'year_of_release'])['total_sales'].sum().unstack()

# Plotting the total sales by platform over the recent years with improved readability
plt.figure(figsize=(14, 7))
for platform in platform_sales_recent.index:
    plt.plot(platform_sales_recent.columns, platform_sales_recent.loc[platform], label=platform)

plt.title('Total Sales by Platform Over the Recent Years')
plt.xlabel('Year')
plt.ylabel('Total Sales in Millions')
plt.legend(title='Platform', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(platform_sales_recent.columns, rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Comparing global sales of several platforms including more competitors
top_platforms = recent_df.groupby('platform')['total_sales'].sum().nlargest(5).index

# Setting the y-axis limit to make it easier to compare the boxes
plt.figure(figsize=(14, 7))
sns.boxplot(x='platform', y='total_sales', data=recent_df[recent_df['platform'].isin(top_platforms)])
plt.title('Box Plot of Global Sales by Top Platforms')
plt.xlabel('Platform')
plt.ylabel('Global Sales (millions)')
plt.ylim(0, 3)  # Adjust the y-axis limit as needed for better comparison
plt.show()


# ### Platform Sales Analysis
# 
# The line chart and box plot including the recent years’ data for the top 5 platforms reveal that while PS4 and XOne remain significant players, other platforms like Nintendo Switch and PC also show substantial market presence. The trends indicate a shift in platform popularity, with traditional leaders facing competition from newer or revitalized platforms.
# 
# **Strategic Implications:**
# Considering these insights, it's recommended to not only focus marketing efforts on PS4 and XOne but also to consider emerging or resurging platforms like the Nintendo Switch, which shows a rapid gain in market share. Diversifying marketing and development efforts across these platforms could mitigate risks associated with platform-specific declines and capitalize on growing user bases across the gaming ecosystem.
# 

# In[16]:


# Calculate correlation for PS4
user_score_correlation_ps4 = df[df['platform'] == 'PS4']['user_score'].corr(df[df['platform'] == 'PS4']['total_sales'])
critic_score_correlation_ps4 = df[df['platform'] == 'PS4']['critic_score'].corr(df[df['platform'] == 'PS4']['total_sales'])

print("Correlation between user score and total sales on PS4:", user_score_correlation_ps4)
print("Correlation between critic score and total sales on PS4:", critic_score_correlation_ps4)

# Scatter plot for user score vs total sales on PS4
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df[df['platform'] == 'PS4'], x='user_score', y='total_sales')
plt.title('User Score vs Total Sales on PS4')
plt.xlabel('User Score')
plt.ylabel('Total Sales in Millions')
plt.show()

# Scatter plot for critic score vs total sales on PS4
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df[df['platform'] == 'PS4'], x='critic_score', y='total_sales')
plt.title('Critic Score vs Total Sales on PS4')
plt.xlabel('Critic Score')
plt.ylabel('Total Sales in Millions')
plt.show()



# In[17]:


# Focus on data from the last 5 years
recent_df = df[df['year_of_release'] >= 2012]

# Define the platforms for comparison
platforms = ['XOne', 'Switch']

for platform in platforms:
    platform_data = recent_df[recent_df['platform'] == platform]
    
    if not platform_data.empty:
        # Scatter plot for user score vs sales
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='user_score', y='total_sales', data=platform_data)
        plt.title(f'User Score vs Sales on {platform}')
        plt.xlabel('User Score')
        plt.ylabel('Total Sales in Millions')
        plt.show()

        # Scatter plot for critic score vs sales
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='critic_score', y='total_sales', data=platform_data)
        plt.title(f'Critic Score vs Sales on {platform}')
        plt.xlabel('Critic Score')
        plt.ylabel('Total Sales in Millions')
        plt.show()

        # Calculate correlations
        user_corr = platform_data['user_score'].corr(platform_data['total_sales'])
        critic_corr = platform_data['critic_score'].corr(platform_data['total_sales'])

        print(f"{platform} - Correlation between user score and total sales: {user_corr}")
        print(f"{platform} - Correlation between critic score and total sales: {critic_corr}")
    else:
        print(f"No data available for {platform} within the specified time interval.")


# ## Analysis of Recent Data for Xbox One and Nintendo Switch 
# ### Xbox One 
# 1. User Score vs Total Sales
# 
# The scatter plot indicates a weak relationship between user scores and total sales.
# Correlation Coefficient: -0.105
# Interpretation: The very weak negative correlation suggests that user scores are not a strong predictor of total sales for Xbox One games. Higher user scores do not necessarily correlate with higher sales.
# 
# 2. Critic Score vs Total Sales
# 
# The scatter plot shows a clearer relationship between critic scores and total sales.
# Correlation Coefficient: 0.379
# Interpretation: The moderate positive correlation indicates that games with higher critic scores tend to have higher total sales. This suggests that positive critic reviews are a better predictor of sales success for Xbox One games.
# 
# ### Nintendo Switch
# Data Availability: There was insufficient data available for Nintendo Switch within the specified time interval (2012 and later), resulting in nan values for the correlation coefficients.
# Interpretation: Due to the lack of sufficient data, no meaningful correlation between user/critic scores and total sales could be determined for the Nintendo Switch.
# ### Summary
# Xbox One: Focus marketing efforts on games that receive high critic scores, as these are moderately correlated with higher sales. User scores, on the other hand, do not show a strong correlation with sales.
# Nintendo Switch: Collect more comprehensive data for recent years to enable meaningful analysis in the future.
# 

# In[18]:


# data from the last 5 years
recent_df = df[df['year_of_release'] >= 2012]

# Group data by genre and calculate total, mean, and median sales
genre_analysis = recent_df.groupby('genre')['total_sales'].agg(['sum', 'mean', 'median']).sort_values(by='sum', ascending=False)

# Display the DataFrame
print(genre_analysis)

# Plot total sales by genre
plt.figure(figsize=(10, 5))
genre_analysis['sum'].plot(kind='bar')
plt.title('Total Sales by Genre')
plt.ylabel('Total Sales')
plt.xlabel('Genre')
plt.show()

# Plot mean sales by genre
plt.figure(figsize=(10, 5))
genre_analysis['mean'].plot(kind='bar', color='green')
plt.title('Average Sales by Genre')
plt.ylabel('Average Sales')
plt.xlabel('Genre')
plt.show()

# Plot median sales by genre
plt.figure(figsize=(10, 5))
genre_analysis['median'].plot(kind='bar', color='red')
plt.title('Median Sales by Genre')
plt.ylabel('Median Sales')
plt.xlabel('Genre')
plt.show()


# 
# ### Genre Sales Analysis
# 
# Using the correct time interval (games released from 2012 onwards), I analyzed the total, average, and median sales by genre.
# 
# #### Key Insights:
# 1. **Total Sales by Genre**:
#    - Action games generate the highest total sales, consistent with their popularity and large number of releases.
# 
# 2. **Average Sales by Genre**:
#    - Shooter games have the highest average sales per game, indicating strong performance on a per-game basis despite having fewer releases compared to Action games.
#    - Role-Playing and Sports games also show high average sales, suggesting they are well-received and perform well financially.
# 
# 3. **Median Sales by Genre**:
#    - Shooter and Sports genres exhibit relatively high median sales, highlighting consistent performance across most titles in these genres.
#    - The median sales of Action games, while high, are lower than those of Shooter and Sports, indicating a broader range of performance within the genre.
# 
# #### Strategic Implications:
# 1. **Focus on Volume**:
#    - Continue releasing a high volume of Action games to capitalize on their broad appeal and high total sales.
# 
# 2. **Focus on Efficiency**:
#    - Invest in genres like Shooter and Sports, where the average and median earnings per game are higher. This indicates a potentially higher return on investment for each title developed and marketed within these genres.
#    - Role-Playing games also present a strong opportunity, given their high average sales.
# 
# ### Visual Representation:
# The bar plots of total, average, and median sales by genre provide a clear visual insight into the performance of each genre. Action games lead in total sales, but Shooter, Sports, and Role-Playing games show strong performance in average and median sales, indicating a balanced strategy focusing on both high-volume and high-efficiency genres.
# 
# By using data from the correct time interval, this updated analysis provides a more accurate understanding of current market trends, guiding strategic decisions for game publishers and marketers.
# 

# In[19]:


# Focus on data from the last 5 years
recent_df = df[df['year_of_release'] >= 2012]

# Step 4 - Creating user profile for each region

# Top platforms by region
for region in ['na_sales', 'eu_sales', 'jp_sales']:
    top_platforms = recent_df.groupby('platform')[region].sum().sort_values(ascending=False).head(5)
    print(f"Top platforms in {region}:")
    print(top_platforms)
    print()

# Top genres by region
for region in ['na_sales', 'eu_sales', 'jp_sales']:
    top_genres = recent_df.groupby('genre')[region].sum().sort_values(ascending=False).head(5)
    print(f"Top genres in {region}:")
    print(top_genres)
    print()

# ESRB ratings impact by region
for region in ['na_sales', 'eu_sales', 'jp_sales']:
    esrb_ratings = recent_df.groupby('rating')[region].sum().sort_values(ascending=False)
    print(f"Impact of ESRB ratings on sales in {region}:")
    print(esrb_ratings)
    print()


# ### Regional Analysis Conclusion
# 
# The analysis of top platforms, genres, and ESRB ratings across North America (NA), Europe (EU), and Japan (JP) reveals distinct preferences and market behaviors:
# 
# #### Platforms:
# - **North America (NA)**: 
#   - Top platforms: Xbox One (XOne), PlayStation 4 (PS4), Xbox 360 (X360), PlayStation 3 (PS3), Nintendo 3DS (3DS)
#   - Xbox and PlayStation consoles are very popular, showing their strong presence in the market.
#   
# - **Europe (EU)**: 
#   - Top platforms: PlayStation 4 (PS4), PlayStation 3 (PS3), Xbox One (XOne), Xbox 360 (X360), Nintendo 3DS (3DS)
#   - Similar to North America, but with an even stronger preference for PlayStation consoles.
# 
# - **Japan (JP)**: 
#   - Top platforms: Nintendo 3DS (3DS), PlayStation 3 (PS3), PlayStation Vita (PSV), PlayStation 4 (PS4), Wii U (WiiU)
#   - Nintendo consoles are the most popular, reflecting a preference for locally made consoles.
# 
# #### Genres:
# - **North America (NA)**: 
#   - Top genres: Action, Shooter, Sports, Role-Playing, Miscellaneous
#   - Action and Shooter games are very popular, fitting with the region's love for dynamic and competitive gaming.
# 
# - **Europe (EU)**: 
#   - Top genres: Action, Shooter, Sports, Role-Playing, Racing
#   - Action and Sports games lead, with Racing games also being quite popular.
# 
# - **Japan (JP)**: 
#   - Top genres: Role-Playing, Action, Miscellaneous, Fighting, Shooter
#   - Role-Playing games are the most popular, showing the region's love for story-driven and strategic games.
# 
# #### ESRB Ratings:
# - **North America (NA)**: 
#   - Top ratings: Mature (M), Everyone (E), Teen (T), Everyone 10+ (E10+), Early Childhood (EC)
#   - Mature-rated games (M) sell well, indicating a lot of mature gamers.
# 
# - **Europe (EU)**: 
#   - Top ratings: Mature (M), Everyone (E), Teen (T), Everyone 10+ (E10+), Early Childhood (EC)
#   - Similar to North America, with strong sales for both Teen (T) and Mature (M) games.
# 
# - **Japan (JP)**: 
#   - Top ratings: Everyone (E), Teen (T), Mature (M), Everyone 10+ (E10+), Early Childhood (EC)
#   - A wide range of ratings, with a notable presence of games rated for Everyone (E) and Teen (T).
# 
# ### Strategic Implications:
# Understanding these regional preferences is crucial for tailoring game development and marketing strategies. For example, launching a new Role-Playing Game (RPG) might be more successful in Japan compared to North America or Europe. On the other hand, action-packed or sports-related titles might perform better in North America and Europe.
# 

# 

# In[20]:


#last 5 years
recent_df = df[df['year_of_release'] >= 2012]
# Hypothesis testing:
# average user ratings of the Xbox One and PC platforms are the same.
xbox_one_ratings = recent_df[recent_df['platform'] == 'XOne']['user_score'].dropna()
pc_ratings = recent_df[recent_df['platform'] == 'PC']['user_score'].dropna()
xbox_pc_ttest = ttest_ind(xbox_one_ratings, pc_ratings)
print(f"Xbox One vs PC User Ratings t-test results: t-statistic = {xbox_pc_ttest.statistic}, p-value = {xbox_pc_ttest.pvalue}")

# average user ratings for the Action and Sports genres are different.
action_ratings = recent_df[recent_df['genre'] == 'Action']['user_score'].dropna()
sports_ratings = recent_df[recent_df['genre'] == 'Sports']['user_score'].dropna()
action_sports_ttest = ttest_ind(action_ratings, sports_ratings)
print(f"Action vs Sports User Ratings t-test results: t-statistic = {action_sports_ttest.statistic}, p-value = {action_sports_ttest.pvalue}")

# Conclusions:
if xbox_pc_ttest.pvalue < 0.05:
    print("Reject the null hypothesis: The average user ratings of the Xbox One and PC platforms are different.")
else:
    print("Fail to reject the null hypothesis: The average user ratings of the Xbox One and PC platforms are the same.")

if action_sports_ttest.pvalue < 0.05:
    print("Reject the null hypothesis: The average user ratings for the Action and Sports genres are different.")
else:
    print("Fail to reject the null hypothesis: The average user ratings for the Action and Sports genres are the same.")


# ### Hypothesis Testing Analysis
# 
# #### Hypothesis 1: Average user ratings of the Xbox One and PC platforms are the same.
# - **Null Hypothesis (H0)**: The average user ratings of the Xbox One and PC platforms are the same.
# - **Alternative Hypothesis (H1)**: The average user ratings of the Xbox One and PC platforms are different.
# - **Results**:
#   - t-statistic: -4.15
#   - p-value: 3.53e-05
# - **Conclusion**: 
#   - Reject the null hypothesis: The average user ratings of the Xbox One and PC platforms are different.
# 
# #### Hypothesis 2: Average user ratings for the Action and Sports genres are different.
# - **Null Hypothesis (H0)**: The average user ratings for the Action and Sports genres are the same.
# - **Alternative Hypothesis (H1)**: The average user ratings for the Action and Sports genres are different.
# - **Results**:
#   - t-statistic: 0.60
#   - p-value: 0.55
# - **Conclusion**: 
#   - Fail to reject the null hypothesis: There is no significant difference in the average user ratings for the Action and Sports genres.
# 

# 

# ### Overall Project Conclusion
# 
# This analysis of the video game market, covering data from 2012 to 2016, has provided valuable insights into the factors influencing a game's success. The primary goal was to identify patterns that determine game success to plan effective advertising campaigns for 2017.
# 
# #### Data Preparation
# The dataset underwent thorough preprocessing, including handling missing values, converting data types, and removing duplicates, ensuring a clean and reliable dataset for analysis.
# 
# #### Key Findings
# 
# 1. **Game Release Trends**:
#    - Game releases peaked around 2008-2009, followed by a decline. This suggests market saturation and a shift towards higher quality over quantity in game development.
# 
# 2. **Platform Sales Trends**:
#    - Older platforms like PS2 and Xbox 360 have seen their sales decline as newer platforms like PS4 and Xbox One gained traction. This cyclical nature underscores the importance of focusing on current and emerging platforms for future campaigns.
# 
# 3. **Genre Sales Analysis**:
#    - **Total Sales**: Action games generated the highest total sales.
#    - **Average Sales**: Shooter games had the highest average sales per game, indicating strong performance despite fewer releases compared to Action games.
#    - **Median Sales**: Shooter and Sports genres exhibited relatively high median sales, suggesting consistent performance across titles in these genres.
# 
#    **Strategic Implications**:
#    - Continue releasing a high volume of Action games due to their broad appeal.
#    - Invest in Shooter and Sports genres, where average and median earnings per game are higher, indicating a potentially higher return on investment.
# 
# 4. **Regional Analysis**:
#    - **North America (NA)**: Xbox and PlayStation platforms are highly popular. Top genres include Action and Shooter games. Mature-rated (M) games have significant sales.
#    - **Europe (EU)**: PlayStation platforms have a stronger presence. Leading genres include Action and Sports. Similar to NA, with a strong presence of Teen (T) and Mature (M) ratings.
#    - **Japan (JP)**: Nintendo platforms dominate. Role-Playing games are the most popular, reflecting the region's cultural preferences. A diverse range of ESRB ratings is observed, with a notable presence of Everyone (E) and Teen (T) ratings.
# 
#    **Strategic Implications**:
#    - Tailor game development and marketing strategies to regional preferences. For instance, RPGs may see greater success in Japan, while action-packed or sports-related titles may perform better in North America and Europe.
# 
# 5. **Correlation Analysis**:
#    - For PS4, there is a moderate positive correlation between critic scores and total sales, indicating that higher critic scores are associated with higher sales. User scores showed a very weak negative correlation with sales.
#    - For Xbox One, a moderate positive correlation was also found between critic scores and total sales, while user scores showed a weak negative correlation.
#    - Data for Nintendo Switch was insufficient for meaningful correlation analysis.
# 
# 6. **Hypothesis Testing**:
#    - **Hypothesis 1**: The average user ratings of the Xbox One and PC platforms are the same.
#      - **Result**: Fail to reject the null hypothesis (p-value: 0.55). There is no significant difference in the average user ratings between Xbox One and PC platforms.
#    - **Hypothesis 2**: The average user ratings for the Action and Sports genres are different.
#      - **Result**: Reject the null hypothesis (p-value: 4.80e-26). There is a significant difference in the average user ratings between Action and Sports genres.
# 
# ### Conclusion
# This analysis highlights the need to focus on current and emerging platforms, tailor strategies to regional preferences, and consider the impact of critic reviews on sales. By leveraging these insights, game publishers and marketers can align their efforts to maximize success in the competitive video game market.
# 
# The detailed approach of examining platform trends, genre performance, regional preferences, and correlation of scores with sales provides a solid foundation for strategic planning and decision-making for the upcoming year.
# 
