# Classification-Mobile-Price-Range-Prediction

Project Summary : This project aims to predict the price range of mobile phones based on their specifications. By analyzing the dataset, performing data preprocessing, and applying machine learning techniques, we can build a predictive model. This information helps companies understand market dynamics, make pricing decisions, and gain a competitive edge in the mobile phone industry.

Project Statement : Analyze the sales data of mobile phones and identify the relationship between mobile phone features (e.g., RAM, internal memory) and their price range, indicating the relative pricing level without predicting the actual price.

Here we start by importing libraries.
Import Libraries: This section includes the necessary libraries and packages imported for data analysis and visualization, data preprocessing, model training, evaluation, and visualization.
# Pandas for data manipulation,aggregation
# NumPy for efficient computational operations.
# Matplotlib and Seaborn for visualization and behaviour with respect to target variable
Dataset Loading :	load a data set from a Google Drive location using Google Colab. It mounts Google Drive and then reads the CSV file named "data_mobile_price_range
.csv" using pd.read_csv() from the pandas library. 
 
 .sample is used to display the random rows .
Data.shape is used to display the no of rows and col.The dataset contains 2000 rows and 21 columns’
Dataset information : To provide detailed information about the dataset, we can use the data.info() method. It provides an overview of the columns, their data types, and the number of non-null values.
Duplicate Values : The information does not mention any duplicate values in the dataset.
Missing Values/Null Values: The table shows the number of missing values (null values) in each column of the dataset. There is no missing/ null value in our dataset.
# Visualizing the missing values : it creates a heatmap visualization of missing values.The missing values are represented by a distinct color, allowing to identify the presence of missing values in the dataset visually. But there is no missing/ null value in our dataset.
What did you know about your dataset?
We have already discuss about this.
Understanding Variable name
Data.columns display the coln name
data.describe(include='all') it is a describe method which is used to generate descriptive statistics of the data within the DataFrame.
It would provide summary of the dataset, including statistical measures such as count, mean, standard deviation, minimum, quartiles, and maximum for numerical columns, as well as the count, unique values, top value, and frequency for categorical/object columns.
variables Description this is  Description about each column name.
Check Unique Values for each variable.
This loop will iterate over each column in data.columns.tolist() and print a statement indicating the number of unique values in that column.

Data Wrangling: 
This section focuses on several data transformations and preparations are performed to make the dataset analysis-ready
In the code above, data_num represents the numerical columns selected from the original dataset, including features such as battery power, clock speed, front camera megapixels (fc), internal memory (int_memory), mobile weight (mobile_wt), and others.
 Similarly, data_cat represents the categorical columns selected from the original dataset, including features such as Bluetooth (blue), dual SIM support (dual_sim), 4G support (four_g), 3G support (three_g), touch screen availability (touch_screen), and Wi-Fi availability (wifi).
the dictionary new_column_names maps the old column names to their corresponding new column names. The rename() function is then used to update the column names.
This code filters the DataFrame based on the conditions data.Screen_width == 0 and data.Pixel_height == 0  and then calculates the length of the resulting filtered DataFrame using the len() function. It prints the total number of phones satisfying each condition separately .  Screen_width is180 and Pixel_height is 2
By executing this code, the zero values in the Screen_width and Pixel_height columns will be replaced with their respective mean values.
Data Vizualization
Chart 1 : Pie Plot Of The Target Variable

The specific chart used in the provided code snippet is a pie chart. A pie chart is chosen to represent the distribution of the price range categories in the dataset. Here's why the pie chart was selected. 

Insight: 
Visualizing distribution: A pie chart is an effective way to visualize the distribution of categorical data. 

The chart visually represents the distribution of mobile phones across different price ranges. The labels indicate the categories of price ranges, which are 'Low Cost', 'Medium Cost', 'High Cost', and 'Very High Cost'. 

The chart provides an overview of the distribution of mobile phones across different price ranges, allowing viewers to quickly understand the relative proportions of low, medium, high, and very high-cost mobile phones in the dataset. 

Business Impact: 
companies can set optimal prices for their products. This can help them remain competitive in the market and attract customers. 

the dataset can help in identifying different customer segments based on their preferred price range.

Chart 2: Relationship Between Target Variable And Numerical Variables
Why  Box plots provide a quick and easy way to visualize the median, quartiles, and outliers in the data. 
Violin plots have been used to show the distribution of continuous variables (mobile weight and talk time) across different price ranges. Violin plots are similar to box plots, but they also show the probability density of the data at different values. 

Insight
Battery power, internal memory, and RAM have a positive relationship with price range. As the value of these features increases, the price range also increases. 
Screen height and width, pixel resolution height and width, and mobile weight do not have a clear relationship with price range. These features have a similar distribution across different price ranges. 
Talk time has a negative relationship with price range. As the value of talk time increases, the price range decreases.
By analyzing the relationship between the target variable (price range) and various features (battery power, internal memory, RAM, screen height and width, pixel resolution, mobile weight, and talk time),.
Business Impact:
businesses can better understand what features are important to customers when it comes to purchasing a mobile phone in different price ranges. 
This information can help businesses make informed decisions about product design, marketing, and pricing strategies. 


Char3.- Relationship Between Target Variable And Categorical Variables


Why  bar chart it helps to show the distribution of each category of the categorical variable and how it relates to the target variable. The use of multiple subplots in this code allows for easy comparison between the different categorical variables and their relationship with the target variable. 
Insights- 
Bluetooth: the proportion of phones with Bluetooth is slightly higher in the higher price range. 
4G: Almost all mobile phones in the higher price range have 4G connectivity. In contrast, the proportion of phones with 4G is lower in the lower price range. 
Dual_sim: The majority of mobile phones in all price ranges have dual sim support. However, the proportion of phones with dual sim is slightly higher in the lower price range. 
3G: The majority of mobile phones in the lower price range have 3G connectivity, while the proportion of phones with 3G decreases as the price range increases. 




Business perspective- 
The majority of the devices in the higher price range have 4G connectivity, a business can leverage this insight by focusing on developing and marketing devices with 4G connectivity to cater to the high-end market segment. 
Similarly, if we observe that devices with dual sim are more popular in the lower price range, a business can focus on developing devices with dual sim to target the budget-conscious segment. 


Chart - 4---Mobile Phones With Dual Sim Feature
Why  pie chart it is a good choice for this type of variable because it is a categorical variable with only two possible values ("yes" and "no"), and the pie chart can clearly show the proportion of each value in the dataset 
Insight- 
The chart shows that 51% of the mobile phones in the dataset have Dual sim feature while 49.1% do not have Dual sim feature. 
This may suggest that the presence or absence of Dual sim feature does not have a significant impact on the price range of mobile phones in the dataset. 
 
Business: 
Negative growth may arise if other variables in the dataset indicate decreasing sales, declining customer satisfaction, or increased competition, among other factors.
Chart 5 Frequency Of Each Category In The Variable.
 
Why  countplots it provide a quick and easy way to get an overview of the categorical variables in the dataset and how they are distributed among the phones . 
Insights- 
The majority of the phones in the dataset have Bluetooth, 4G, Dual_sim, 3G, Wifi, and Touch_screen features. 
The Dual_sim and Touch_screen features are the most common among the phones in the dataset. 
The 4G and Wifi features are also quite popular among the phones in the dataset. 
The 3G feature is less common among the phones in the dataset , but still present in a significant number of them. 
Business- It may indicate that there is a high demand for that feature in the market, which can lead to positive business impact if the company offers products or services that cater to that demand. 


Chart 6- 
Histograms are a useful tool for displaying the distribution of a numerical variable. They display the frequency or count of observations within a specific range or bin of values. 
Insights- 
Battery_power: The majority of phones have battery power is 600 mAh to 1600 mAh. 
Front_camera and Primary_camera: The majority of phones have camera resolutions between 0-5 megapixels. 
Screen_height and Screen_width: The distribution of screen height and screen width is relatively normal. Most phones have a screen size of 20 cm. 
Pixel_width and Pixel_height: The majority of phones have a pixel resolution between 500-1000*. 
Internal_memory and RAM: The distribution of internal memory and RAM is skewed to the right. Most phones have internal memory and RAM less than 32 GB and 2GB, respectively. 
Mobile_depth: The majority of phones have a depth between 0.2-0.5 cm. 
Number_of_cores and Processor_speed: Most phones have 4 cores and a processor speed between 1-2 GHz. 
Mobile_weight: The distribution of mobile weight is skewed to the right. Most phones have a weight between 90-200 grams. 
Talk_time: The majority of phones have a talk time between 5-20 hours. 
Business- 
by analyzing the various features of the mobile phones, businesses can understand what customers value the most and what features they are willing to pay a premium for. 
This can inform product development and marketing strategies to meet customer needs and increase sales. 
Chart 7- 
This chart helps to understand the relationship between the binary features and the price range of mobile phones in the dataset. It can help to identify any trends or patterns between the two variables, which can be useful for making business decisions or developing predictive models 
Insights- 
The majority of mobile phones in the dataset support 4G, with only a small percentage not supporting it. 
There is a higher proportion of mobile phones in the high and very high price ranges that support 4G compared to the low and medium price ranges. 
Most of the mobile phones in the dataset also support 3G, with only a small percentage not supporting it. 
The distribution of mobile phones by price range is similar for those that support and do not support 3G. 
The chart suggests that having 4G support is more closely related to the price range of a mobile phone than having 3G support. 
Business- 
The relationship between the features and the price range can help mobile phone manufacturers make informed decisions about which features to prioritize and which ones to improve or eliminate to meet consumer demand. 
Chart8- 
Why Distplot it is a convenient way to visualize the distribution of a single variable, which can be helpful in identifying patterns and understanding the underlying structure of the data . 
The dashed lines at the mean and median values also provide information about the data's central tendency. 
Insights- 
We can see if a variable is normally distributed or if it has a skewed distribution. We can also see if there are any outliers in the data that may need to be addressed before building a predictive model. 
Business- 
The analysis shows that customers highly value battery life and camera quality, the company can focus on improving these features in their products to attract more customers and gain a competitive advantage. 
The analysis shows that customers are not interested in certain features, such as 3G connectivity, but the company continues to invest in these features, it may lead to negative growth. 


Chart 9: Relationship Between Primary Camera And Front Camera.


The histogram is a good choice for this dataset because it allows us to visualize the distribution of the data for each variable. 
Insight- 
the Primary camera variable has a wider range of megapixels than the Front camera variable, as indicated by the spread of the histogram. 
This suggests that the primary camera on mobile phones tends to have a wider range of megapixels than the front camera. 
Business- 
if a mobile phone manufacturer decides to prioritize the front camera's mega-pixels over the primary camera's mega-pixels based on the above visualization, they might produce phones with high front camera mega-pixels but low primary camera mega-pixels. 
Mega-pixels alone do not determine camera quality, and other factors like sensor size and image processing algorithms also play a crucial role. 


Chart - 10--Relationship Between Pixel Width And Price Range .
The KDE plot estimates the probability density function of the data and provides a smooth curve that highlights the shape of the distribution. 
The boxplot gives a quick visual summary of the distribution of the data and shows the median, quartiles, and any outliers in each price range. 
Insight-
KDE
The distribution of Pixel_width varies across different Price_ranges. For example, high-end phones (Price_range 3) may have higher Pixel_width than low-end phones (Price_range 0). 
There may be overlap between the distributions of Pixel_width for different Price_ranges, indicating that Pixel_width may not be a strong predictor of Price_range on its own. 
BOX Plot: 
The median Pixel_width may differ across different Price_ranges. 
The interquartile range (IQR) of Pixel_width may differ across different Price_ranges, indicating that there may be different levels of variability in Pixel_width across different Price_ranges. 
The presence of outliers in Pixel_width may differ across different Price_ranges, indicating that there may be extreme values of Pixel_width that are more common in some Price_ranges than others. 
Business- 
This insight could help businesses prioritize investing in higher quality displays in their higher-end phones to differentiate themselves in the market. 
 




 
Chart11- 
KDE plots are useful for visualizing the distribution of a variable. It provides an estimate of the probability density function of a random variable. 
Insight- 
Battery_power: Most of the phones in the dataset have a battery power between 900 mAh to 2100 mAh. 
Processor_speed: The processor speed distribution seems to be approximately normally distributed, with most phones having a processor speed around 1.5 GHz to 2.5 GHz. 
Front_camera: The majority of phones in the dataset have a front camera with 5 to 10 mega-pixels. 
Internal_memory: The internal memory distribution is right-skewed, with most phones having 8GB to 32GB of internal memory. 
Mobile_depth: The mobile depth distribution is approximately normal, with most phones having a depth of around 8 mm to 9 mm. 
Mobile_weight: The majority of phones in the dataset have a weight between 80g to 200g. 
Number_of_cores: The number of cores distribution is right-skewed, with most phones having 4 to 8 cores. 
Primary_camera: The primary camera distribution is also right-skewed, with most phones having a primary camera with 8 to 16 mega-pixels. 
Pixel_height and Pixel_width: The pixel height and width distributions are approximately normal, with most phones having a resolution of 1000 to 2000 pixels. 
RAM: The RAM distribution seems to be approximately normally distributed, with most phones having 1GB to 4GB of RAM. Screen_height and Screen_width: The screen height and width distributions are approximately normal, with most phones having a screen size of around 5 to 6 inches. 
Talk_time: The talk time distribution is right-skewed, with most phones having a talk time of around 10 to 18 hours. 
Business- 
The analysis shows that smartphones with higher battery power and better cameras tend to have a higher price range, businesses can focus on improving these features in their products to attract customers who are willing to pay a higher price. 
Chart12- 
The KDE plot provides a visual representation of the density of weight values for each price range, allowing us to compare the distributions and identify any differences or similarities. 
Insight- 
The variable Mobile_weight is related to the price range of mobile phones. 
For example, if the KDE plot shows that the distribution of Mobile_weight is different for different price ranges, we can conclude that Mobile_weight is a significant variable in determining the price range of mobile phones. 
If the box plot shows that the median Mobile_weight decrease as the price range decrease, we can conclude that there is a negative correlation between Mobile_weight and price range. 
Business- 
If the weight of a phone becomes excessive, it may also turn off some consumers who prioritize portability and convenience, which could lead to negative growth for the manufacturer. 
Chart - 13---Relationship Between Screen Size And Price Range .
The chart includes a kernel density estimation (KDE) plot and a boxplot to explore the relationship between the screen size (sc_size) and the price range of the mobile phone. 
Insights- 
It appears that the sc_size distribution is higher for higher-priced phones (price range 2 and 3) compared to lower-priced phones (price range 0 and 1). 
Business- 
The analysis shows that larger screen size is associated with higher price ranges, a business might choose to prioritize larger screens in their product line. 
Chart14- 
The heatmap is a graphical representation of the correlation matrix that shows how strongly each variable is related to the other variables. 
Insights- 
Battery power, RAM, and price range have a strong positive correlation, which means that as the battery power and RAM of a phone increase, so does its price range. 
Internal memory, primary camera, pixel width, and screen height also have a moderate positive correlation with price range. 
The number of cores and talk time have a weak positive correlation with price range. 
Pixel height and screen width have a weak negative correlation with price range. 
RAM and internal memory have a strong positive correlation, which means that as the RAM of a phone increases, so does its internal memory. 
Pixel height and pixel width have a strong positive correlation, which means that as the pixel height of a phone increases, so does its pixel width. 
 
 
Chart15- 
 Pair Plot generates pairwise scatter plots for all the variables in the dataset and allows us to see the relationships between variables, as well as the distribution of each variable  
Insights- 
The 'RAM' variable has a strong positive correlation with the 'Price_range' variable, which means that mobile phones with higher RAM tend to have higher price ranges. 
There is a weak positive correlation between the 'Battery_power' and 'Price_range' variables, which means that mobile phones with higher battery power tend to have slightly higher price ranges. 
The 'Pixel_height' and 'Pixel_width' variables have a positive correlation with each other, which is expected since higher pixel heights and widths typically result in better screen resolution. 
The 'Screen_height' and 'Screen_width' variables also have a positive correlation with each other, which means that mobile phones with larger screen heights also tend to have larger screen widths. 
There are no strong correlations between the other variables in the dataset. 
 
Hypothesis Testing
1)Hypothetical Statement--To test whether there is a significant difference in the mean battery power between mobile phones with 3G and those without 3G:
The null hypothesis for this t-test would be that there is no significant difference in the average battery power between mobile phones with 3G and those without 3G.
The alternative hypothesis would be that there is a significant difference in the average battery power between the two groups.
Method:  t-test (two-sample t-test) is used to compare
Result: The p-value of 0.6066 indicates that there is no significant difference in price range between the two groups at a 5% significance level.
2)Hypothetical Statement - Is there a significant difference in internal memory between mobile phones with touch screens and mobile phones without touch screens;
Null hypothesis: There is no significant difference in internal memory between mobile phones with touch screens and mobile phones without touch screens.
Alternative hypothesis: There is a significant difference in internal memory between mobile phones with touch screens and mobile phones without touch screens.
Method:  t-test (two-sample t-test) is used to compare
Result: 
The obtained t-value in this case is -1.2073, indicating that the difference between the means of the two groups is -1.2073 times the standard error of the difference. The calculated p-value is 0.2275, which exceeds the commonly used significance level of 0.05.
As a result, the null hypothesis that there is no significant difference in internal memory between mobile phones with and without a touch screen is not rejected.
3)Hypothetical Statement --Is there a significant difference in battery power between mobile phones with high cost and mobile phones with low cost?
Null hypothesis: There is no significant difference in battery power between mobile phones with high cost and mobile phones with low cost.
Alternative hypothesis: There is a significant difference in battery power between mobile phones with high cost and mobile phones with low cost.
Method:  t-test (two-sample t-test) is used to compare
Result: The t-value of 10.0743 indicates that the means of the two groups differ significantly.
The p-value of 0.0000 indicates strong evidence against the null hypothesis, which states that there is no significant difference in battery power between high-cost and low-cost mobile phones.
Therefore, we can reject the null hypothesis and conclude that there is a significant difference in battery power between the high-cost and low-cost mobile phones.

Feature Manipulation: Removing irrelevant or unnecessary columns from the dataset can be beneficial for the machine learning model. It helps in reducing noise, improving computational efficiency, and focusing on the most relevant features that contribute to the target variable.
The new feature 'total_resolution' will be created by multiplying the 'Pixel_height' and 'Pixel_width' columns. The original 'Pixel_height' and 'Pixel_width' columns will be dropped from the dataset to minimize feature correlation.

Feature Selection:By using this approach, you can select a subset of features that are most relevant for predicting the target variable, which can help avoid overfitting and improve the performance of your model.
the SelectKBest method is used with the f_classif score function to rank the features based on their importance for predicting the target variable. The code selects the top k features with the highest scores and stores them in the selected_features array. The original feature matrix X is then subsetted to include only the selected features, creating a new DataFrame X_selected. Finally, the selected feature names are printed.
Data Transformation: new column 'sc_ratio' will be created by dividing the 'Screen_height' by the 'Screen_width'. The original columns 'Screen_width' and 'Screen_height' will be dropped from the dataset. The 'Price_range' column will be label encoded using the LabelEncoder, where each unique value will be assigned a numerical label. Finally, the first five rows of the transformed dataset will be printed for inspection.
Data Scaling : This scaling process ensures that the variables are on a similar scale, which can be beneficial for various machine learning algorithms that are sensitive to the magnitude of variables.
The specified columns will be scaled using Min-Max scaling, which transforms the values to a range between 0 and 1. The scaled values will replace the original values in the dataset. The head() function is used to display the first five rows of the scaled data for inspection.
Data Splitting : the features are assigned to the variable X, and the target variable 'Price_range' is assigned to the variable y. The train_test_split function is then used to split the data into training and testing sets, with a test size of 0.2 (20% of the data) and a random state of 42 for reproducibility. The resulting training sets are assigned to X_train and y_train, while the testing sets are assigned to X_test and y_test. Finally, the shape of the training and testing sets is printed to verify the split.
Handling Imbalanced Dataset : 
The RandomUnderSampler is used to randomly remove samples from the majority class (higher frequency) to balance the dataset. The resampled features are stored in X_resampled, and the corresponding target values are stored in y_resampled. The shape of the original and resampled datasets is printed to compare the sample sizes.
Lastly, the countplot from the seaborn library is used to visualize the distribution of the resampled target variable (Price_range). This helps verify if the undersampling process has effectively balanced the classes.


ML Model - 1 LogisticRegression
Here, y_test represents the true target labels of the test set. The accuracy_score function compares the predicted labels (y_pred) with the true labels (y_test) and calculates the accuracy of the model.
Accuracy: 0.895 ( 89%)

Evaluation metric Score: Evaluation metric scores help in comparing different models or variations of the same model and selecting the one that performs the best for a specific task. They provide insights into the model's strengths and weaknesses and guide further improvements or adjustments to enhance the model's performance.
The evaluation metric scores (accuracy, precision, recall, and F1 score) are stored in a dictionary called metric_scores. The dictionary is then converted to a pandas DataFrame, where the metric names are used as a column called 'Metric' and the scores are stored in a column called 'Score'. The DataFrame is then plotted using Seaborn's barplot function.

Cross-validation: cross-validation is a valuable tool in machine learning for robust model evaluation, model selection, and generalization assessment. It provides a more realistic estimate of a model's performance and helps in building reliable and effective predictive models.
By performing hyperparameter optimization using GridSearchCV, the code helps in finding the best combination of hyperparameters for the logistic regression model, resulting in improved accuracy compared to using default hyperparameter values.

ML Model - 2---DecisionTreeClassifier
The Decision Tree classifier is a popular machine learning algorithm for classification tasks. It builds a tree-like model of decisions and their possible consequences, based on the training data. The trained model can then be used to make predictions on new, unseen data.
Accuracy: 0.8375

ML Model - 3--RandomForestClassifier
Random Forest is an ensemble learning method that combines multiple decision trees to create a more robust and accurate model. It leverages the concept of bagging and random feature selection to reduce overfitting and improve generalization. Random Forest classifiers are widely used in various machine learning tasks, including classification, regression, and feature importance estimation.
Create an instance of the Random Forest classifier: The code creates an instance of the RandomForestClassifier class, setting the random_state parameter to 42. This ensures that the random processes in the classifier are reproducible.
Fit the model on the training data: The fit() function is called to train the Random Forest classifier on the training data. The X_train variable represents the features of the training set, and y_train represents the corresponding target variable.
Make predictions on the test data: The trained model is used to make predictions on the test data using the predict() function. The X_test variable contains the features of the test set.
Calculate the accuracy of the model: The accuracy_score() function is used to calculate the accuracy of the model by comparing the predicted labels (y_pred) with the actual labels (y_test). The accuracy score represents the percentage of correctly predicted labels in the test set.
Print the accuracy: Accuracy: 0.88
3. Explain the model which you have used and the feature importance using any model explainability tool?
!pip install shap : This command will install the shap library, allowing you to use it for model interpretation and explainability.
Logistic regression is a statistical model that utilizes a logistic function to estimate the probability of a binary outcome. It is commonly used for classification tasks, such as predicting mobile phone price ranges. In this implementation, precision, recall, and F1 score were computed as evaluation metrics to assess the model's performance. The confusion matrix provides insights into the true positive rate for each class, while the accuracy was determined to be 0.965, indicating the proportion of correctly predicted instances.
Conclusion
Based on the analysis of the provided dataset, the following conclusions can be drawn:
RAM, battery power, internal memory, and screen size (px_height and px_width) exhibit a strong positive correlation with the price range of mobile phones.
Among the implemented models, logistic regression achieved the highest accuracy of 0.927, followed by Random Forest Classifier with an accuracy of 0.875 and decision tree with 0.8475 accuracy.
The most influential features for predicting the price range of mobile phones are RAM, battery power, internal memory, and screen size.
Manufacturers can focus on enhancing these features in their mobile phones to increase their price range and effectively target specific customer segments.
The dataset was found to be clean, devoid of missing values or outliers, facilitating smooth analysis and model development.
Overall, the project successfully predicted the price range of mobile phones based on their features, empowering companies to make informed decisions and improve their competitive position in the market.
