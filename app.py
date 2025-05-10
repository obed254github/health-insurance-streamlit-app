import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import joblib
import pickle

# ------------------------ #
#         HEADER           #
# ------------------------ #
st.markdown("---")
st.markdown(
    "<h2 style='color: #B22220;'>📊 Data Analysis</h1>",
    unsafe_allow_html=True,
)
st.markdown("---")

st.markdown(
    "<h4 style='color: #B22220;'>Loading all necessary libraries</h4>",
    unsafe_allow_html=True,
)

st.markdown("""
            ```python
            import pandas as pd
            import numpy as np
            import seaborn as sns
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import OneHotEncoder, StandardScaler
            from sklearn.compose import ColumnTransformer
            from sklearn.linear_model import LinearRegression
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import GridSearchCV
            from lightgbm import LGBMRegressor
            from xgboost import XGBRegressor
            from scipy import stats
            from sklearn.metrics import r2_score, mean_squared_error
            import matplotlib.pyplot as plt
            import joblib
            ```
            """)


# ------------------------ #
#       LOAD DATA          #
# ------------------------ #
def load_data():
    return pd.read_csv("data/HealthData.csv")


df = load_data()

st.markdown(
    "<h4 style='color: #B22220;'>Loading data</h4>",
    unsafe_allow_html=True,
)

st.markdown("""
            ```python
            data = pd.read_csv("data/HealthData.csv")
            ```
            """)
# ------------------------ #
#      DATA PREVIEW        #
# ------------------------ #

st.markdown(
    "<h4 style='color: #B22220;'>Dataset preview</h4>",
    unsafe_allow_html=True,
)

st.markdown("""
            ```python
            #Displaying the first 5 rows of the data.
            data.head()
            ```
            """)
st.table(df.head())


st.markdown(
    "<h4 style='color: #B22220;'>Feature description</h4>",
    unsafe_allow_html=True,
)

st.markdown("""
            ```python
            #Summary statistics of numerical features
            data.describe().transpose()
            ```
            """)
st.write(df.describe().transpose())

# ------------------------ #
#    CHARGES SUMMARY       #
# ------------------------ #

st.markdown(
    "<h4 style='color: #B22220;'>Insurance charges summary</h4>",
    unsafe_allow_html=True,
)

st.markdown("""
            ```python
            summary = {
            "Minimum charges": df["charges"].min(),
            "Mean charges": df["charges"].mean(),
            "Maximum charges": df["charges"].max(),
            }
            summary_df = pd.DataFrame(summary.items(), columns=["Metric", "Value"])
            table(summary_df) 
            ```   
            """)
summary = {
    "Minimum charges": df["charges"].min(),
    "Mean charges": df["charges"].mean(),
    "Maximum charges": df["charges"].max(),
}
summary_df = pd.DataFrame(summary.items(), columns=["Metric", "Value"])
st.table(summary_df)

# ------------------------ #
#     VISUALIZATIONS       #
# ------------------------ #
st.markdown("---")
st.markdown(
    "<h2 style='color: #B22220;'>Data Vizualisations</h1>",
    unsafe_allow_html=True,
)


def age_converter(age):
    if 18 <= age <= 35:
        return "Young"
    elif 36 <= age <= 55:
        return "Middle-aged"
    else:
        return "Senior"


df["age_group"] = df["age"].apply(age_converter)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Histogram", "Boxplot", "Scatter Plot", "Correlation"]
)

with tab1:
    st.markdown(
        "<h5 style='color: #B22220;'>Distribution of selected feature</h5>",
        unsafe_allow_html=True,
    )
    df1 = df[["age", "charges", "bmi"]]
    feature = st.selectbox("Select feature", df1.columns, key="hist_feature")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(df1[feature], kde=True, color="black", ax=ax)
    ax.set_title(f"Distribution of {feature}")
    fig.tight_layout()
    st.pyplot(fig)

with tab2:
    st.markdown(
        "<h5 style='color: #B22220;'>Boxplot: Charges by category</h5>",
        unsafe_allow_html=True,
    )
    category_option = st.selectbox(
        "Select category to compare against charges:",
        options=["Smoking Status", "Age Group"],
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    if category_option == "Smoking Status":
        sns.boxplot(y="charges", x="smoker", data=df, palette="Set1", ax=ax)
        ax.set_title("Charges by Smoking Status")
        ax.set_xlabel("Smoking Status")
    elif category_option == "Age Group":
        sns.boxplot(y="charges", x="age_group", data=df, palette="Set1", ax=ax)
        ax.set_title("Charges by Age Group")
        ax.set_xlabel("Age Group")
    ax.set_ylabel("Charges")
    fig.tight_layout()
    st.pyplot(fig)

with tab3:
    st.markdown(
        "<h5 style='color: #B22220;'>Scatter plot: Charges vs continuous features</h5>",
        unsafe_allow_html=True,
    )
    numerical_features = ["age", "bmi", "children"]
    selected_feature = st.selectbox(
        "Select feature to plot against charges", numerical_features
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        x=selected_feature,
        y="charges",
        hue="age_group",
        palette="Set1",
        size="age_group",
        data=df,
        ax=ax,
    )
    ax.set_xlabel(selected_feature.capitalize())
    ax.set_ylabel("Insurance Charges ($)")
    ax.set_title(f"Charges vs {selected_feature.capitalize()} by Age Group")
    ax.legend().set_title("Age group")
    fig.tight_layout()
    st.pyplot(fig)

with tab4:
    st.markdown(
        "<h5 style='color: #B22220;'>Correlation Matrix: Age, BMI, Children, and Charges</h5>",
        unsafe_allow_html=True,
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    selected_cols = ["age", "bmi", "children", "charges"]
    correlation = df[selected_cols].corr()
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", mask=mask, ax=ax)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    st.pyplot(fig)

# ------------------------ #
#   MODEL BUILDING INFO    #
# ------------------------ #
st.markdown("---")
st.markdown(
    "<h2 style='color: #B22220;'>Model building</h2>",
    unsafe_allow_html=True,
)

st.markdown("---")


# --------------------------------------------------- #
#   All the code from evaluation jupyter notebook     #
# --------------------------------------------------- #
data = load_data()


# Converting smoker to binary values 1 for smoking = Yes and 0 for not smoking = No
def smoker_conversion(variable):
    if variable == "yes":
        return 1
    else:
        return 0


data["smoker"] = data["smoker"].apply(smoker_conversion)


# categorizing individuals into 'Young' (18-35), 'Middle-aged' (36-55), and 'Senior' (56+), then calculate the average charges for each age group.
def age_converter(age):
    if age >= 18 | age <= 35:
        return "Young"
    elif age >= 36 | age <= 55:
        return "Middle-aged"
    else:
        return "Senior"


data["age_group"] = data["age"].apply(age_converter)
# Creating a variable named bmi_smoker which is a product of smoker and bmi
data["bmi_smoker"] = data["bmi"] * data["smoker"]
# Creating a variable named age_smoker which is a product of age and smoker.
data["age_smoker"] = data["age"] * data["smoker"]
# Creating a variable named children age_bmi which is a product of age and bmi
data["age_bmi"] = data["age"] * data["bmi"]
# Creating a variable named children_per_age which divides the number of children by age
data["children_per_age"] = data["children"] / data["age"]

# Define features and target variable
X = data.drop("charges", axis=1)
y = data["charges"]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            [
                "age",
                "bmi",
                "children",
                "bmi_smoker",
                "age_smoker",
                "age_bmi",
                "children_per_age",
            ],
        ),
        ("cat", OneHotEncoder(drop="first"), ["age_group", "sex", "smoker", "region"]),
    ]
)

# Apply preprocessing to features
X_processed = preprocessor.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
# Make predictions on test set
y_pred = linear_model.predict(X_test)

# Evaluate performance using R² and RMSE
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))


# Function to calculate adjusted R2
def adjusted_r2(r2_calc):
    n = len(y_test)  # Number of observations
    p = X.shape[1]  # Number of predictors
    adjusted_r2 = 1 - ((1 - r2_calc) * (n - 1) / (n - p - 1))
    return adjusted_r2


st.markdown(
    "<h3 style='color: #B22220;'>Feature Engineering</h3>", unsafe_allow_html=True
)
st.markdown("""
- Created interaction terms: `bmi_smoker`, `age_smoker`, `age_bmi`, `children_per_age` 
-These terms allows the model to capture interactive effects,
where the combined effect of two variables is more pronounced than their
individual contributions.For example `bmi_smoker` seeks to determine
what health effects does smoking have on insurance charges if it increases or 
decreases the costs.They are useful in linear models,which can’t capture complex 
interactions unless they are explicitly created. 
- Encoded categorical variables using `OneHotEncoder`  
- Scaled numerical features with `StandardScaler`  

```python
#Creating features that can help confound interactive effects on  charges.
#Creating a variable named bmi_smoker which is a product of smoker and bmi
df["bmi_smoker"] = df["bmi"] * df["smoker"].map({'yes': 1, 'no': 0})
#Creating a variable named age_smoker which is a product of age and smoker.
df["age_smoker"] = df["age"] * df["smoker"].map({'yes': 1, 'no': 0})
#Creating a variable named children age_bmi which is a product of age and bmi
df["age_bmi"] = df["age"] * df["bmi"]
#Creating a variable named children_per_age which divides the number of children by age of the dependent
#This will show the dependency per age
input_df["children_per_age"] = input_df.apply(
            lambda x: x["children"] / x["age"] if x["age"] != 0 else 0, axis=1
        )
```

```python
# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['age', 'bmi', 'children', 'bmi_smoker', 'age_smoker', 'age_bmi', 'children_per_age']),
        ('cat', OneHotEncoder(drop='first'), ['age_group', 'sex', 'smoker', 'region'])
    ]
)
```
""")
st.markdown("<h3 style='color: #B22000;'>Model Selection</h3>", unsafe_allow_html=True)
st.markdown("""
- Trained: `Linear Regression`, `RandomForestRegressor`, `LightGBM`, and `XGBoost`  
- Tuned `Random Forest` using grid search 
-`GridSearchCV` came in handy in determining the optimal parameters for the optimal 
RandomForest model. Grid Search tries to search systematically through hyperparameter
values while evaluating performance of each combination.
-For a RandomForest model hyperparameters such as;
    - n_estimators: Number of trees to build
    - max_depth: Determines maximum depth for each tree.
    - min_samples_split: Refers to the minimum number of samples required 
    to split an internal node.
    - max_features: The number of features that should be considered while looking
    for the best split.
    - bootstrap: Determines whether bootstrap samples will be used when building the trees.
    Setting bootstrap to `True` performs random sampling with replacement, while `False`
    will use the whole dataset for each tree.
- Visualized feature importance in determining the cost of insurance charges for optimal
RandomForest model.
""")
st.markdown("<h3 style='color: #B22000;'>Model Intuition</h3>", unsafe_allow_html=True)

st.markdown(
    "<h4 style='color: #B22000;'>Linear Regression</h4>", unsafe_allow_html=True
)
st.markdown("""
- Fits a straight line to model the relationship between features and charges.
- Assumes the effect of each feature is additive and constant.
- Easy to interpret but limited when the relationship is non-linear.
""")
st.markdown(
    "<h4 style='color: #B22000;'>Fitting Linear Regression model</h4>",
    unsafe_allow_html=True,
)
st.markdown("""
```python
model = LinearRegression()
model.fit(X_train, y_train)

#Displaying model coefficients and the intercept
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)

#Making predictions on test set
y_pred = model.predict(X_test)

# Evaluating performance using R² and RMSE
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
#Calculating adjusted R2
def adjusted_r2(r2_calc):
    n = len(y_test)  # Number of observations
    p = X.shape[1]  # Number of predictors
    adjusted_r2 = 1 - ((1 - r2_calc) * (n - 1) / (n - p - 1))
    return adjusted_r2

print(f"Linear model R² Score: {r2:.3f}")
print(f"Linear model RMSE:{rmse:.2f}")
r2_adj = adjusted_r2(r2)
print(f"linear model adjusted R2: {r2_adj:.3f}")
```
""")
st.markdown(f"Linear model R² Score: {r2:.3f}")
st.markdown(f"Linear model RMSE:{rmse:.2f}")
r2_adj = adjusted_r2(r2)
st.markdown(f"linear model adjusted R2: {r2_adj:.3f}")
st.markdown("""
```python
#Plotting actual vs predicted charges
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Reference line (y=x)
plt.xlabel('Actual charges')
plt.ylabel('Predicted charges')
plt.title('Actual vs predicted charges')
plt.show()
```
""")
# Creating the figure
fig, ax = plt.subplots(figsize=(10, 6))
# Making a scatter plot
ax.scatter(y_test, y_pred, alpha=0.6, color="#FF7F7F")
# Plotting a reference line
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])
# Setting labels
ax.set_xlabel("Actual charges")
ax.set_ylabel("Predicted charges")
ax.set_title("Actual vs Predicted Charges")
# Displaying the plot.
st.pyplot(fig)

st.markdown(
    "<h4 style='color: #B22000;'>Random Forest Regressor</h4>",
    unsafe_allow_html=True,
)

st.markdown("""
- Builds multiple decision trees and averages their results.
- Captures complex interactions and non-linear patterns.
- More robust than a single tree and less prone to overfitting.
""")

st.markdown(
    "<h4 style='color: #B22000;'>Fitting base random forest regressor</h4>",
    unsafe_allow_html=True,
)
st.markdown("""
```python
#Initializing Random Forest Regressor
rf_model = RandomForestRegressor(
    n_estimators=100, #Number of trees
    max_depth=None, #Depth of each tree (None = full depth)
    random_state=42
)

#Fitting the model
rf_model.fit(X_train, y_train)

#Making predictions
y_pred = rf_model.predict(X_test)

#Evaluating the model
RMSE = mean_squared_error(y_test, y_pred, squared = False)
R2 = r2_score(y_test, y_pred)

rf_predictions = rf_model.predict(X_test)
RMSE = mean_squared_error(y_test, rf_predictions, squared = False)
R2 = r2_score(y_test, rf_predictions)
print(f"Base Random Forest Model RMSE: {RMSE:.2f}")
print(f"Base Random Forest model R2: {R2:.2f}")
r2_adj = adjusted_r2(R2)
print(f"Base Random Forest model adjusted R2: {r2_adj:.3f}")
```
""")
# Initializing Random Forest Regressor
rf_model = RandomForestRegressor(
    n_estimators=100,  # Number of trees
    max_depth=None,  # Depth of each tree (None = full depth)
    random_state=42,
)

# Fitting the model
rf_model.fit(X_train, y_train)

# Making predictions
y_pred = rf_model.predict(X_test)

# Evaluating the model
RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
R2 = r2_score(y_test, y_pred)

rf_predictions = rf_model.predict(X_test)
RMSE = np.sqrt(mean_squared_error(y_test, rf_predictions))
R2 = r2_score(y_test, rf_predictions)
st.markdown(f"Base Random Forest Model RMSE: {RMSE:.2f}")
st.markdown(f"Base Random Forest model R2: {R2:.2f}")
r2_adj = adjusted_r2(R2)
st.markdown(f"Base Random Forest model adjusted R2: {r2_adj:.2f}")

st.markdown(
    "<h4 style='color: #B22000;'>Performing a Grid Search on Random Forest Model</h4>",
    unsafe_allow_html=True,
)

st.markdown("""
```python
#Defining parameter grid
param_grid = {
    'n_estimators': [10,100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap':[True,False]
}

rf = RandomForestRegressor(random_state=42)

#GridSearch Search
grid_search = GridSearchCV(
    estimator=rf, 
    param_grid=param_grid,
    cv=5, 
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose = 0
    )
grid_search.fit(X_train, y_train)
predictions = grid_search.predict(X_test)
mse = mean_squared_error(y_test , predictions, squared = False)
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best R² score: {grid_search.best_score_}")
```

""")
# Initializing Random Forest Regressor
rf_model = RandomForestRegressor(
    n_estimators=100,  # Number of trees
    max_depth=None,  # Depth of each tree (None = full depth)
    random_state=42,
)

# Fitting the model
rf_model.fit(X_train, y_train)

# Making predictions
y_pred = rf_model.predict(X_test)

# Evaluating the model
RMSE = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred,
    )
)
R2 = r2_score(y_test, y_pred)

rf_predictions = rf_model.predict(X_test)
RMSE = np.sqrt(mean_squared_error(y_test, rf_predictions))
R2 = r2_score(y_test, rf_predictions)
# Tuned RF model
final_model = RandomForestRegressor(
    bootstrap=True,
    max_depth=10,
    max_features="sqrt",
    min_samples_leaf=4,
    min_samples_split=10,
    n_estimators=100,
)
final_model.fit(X_train, y_train)

st.markdown(
    "<h4 style='color: #B22000;'>Fitting the tuned RandomForest Model</h4>",
    unsafe_allow_html=True,
)

st.markdown("""
```python
final_model = RandomForestRegressor(
    bootstrap = True,
    max_depth = 10,
    max_features = 'sqrt',
    min_samples_leaf = 4,
    min_samples_split= 10, 
    n_estimators = 100
)
final_model.fit(X_train, y_train)
final_model.feature_importances_
```
```python
#Getting predictions
predictions = final_model.predict(X_test)
#Calculating root mean squared error
RMSE = mean_squared_error(y_test, predictions, squared = False)
#Calculating R2
R2 = r2_score(y_test, predictions)
print(f"Tuned Random Forest Model RMSE: {RMSE:.2f}")
print(f"Tuned Random Forest model R2: {R2:.2f}")
r2_adj = adjusted_r2(R2)
print(f"Tuned Random Forest model adjusted R2: {r2_adj:.3f}")   
```

""")

predictions = final_model.predict(X_test)
RMSE = np.sqrt(mean_squared_error(y_test, predictions))
R2 = r2_score(y_test, predictions)
st.write(f"Tuned Random Forest Model RMSE: {RMSE:.2f}")
st.write(f"Tuned Random Forest model R2: {R2:.2f}")
r2_adj = adjusted_r2(R2)
st.write(f"Tuned Random Forest model adjusted R2: {r2_adj:.3f}")
st.markdown("""
**Plotting feature importances for the tuned model**
```python
#Getting numeric features and categorical features from the original data.
numeric_features = ['age', 'bmi', 'children', 'bmi_smoker', 'age_smoker', 'age_bmi', 'children_per_age']
categorical_features = ['age_group', 'sex', 'smoker', 'region']
#Accessing OneHotEncoder from the fitted preprocessor
ohe = preprocessor.named_transformers_['cat']
encoded_cat_features = ohe.get_feature_names_out(categorical_features)
all_feature_names = list(numeric_features) + list(encoded_cat_features)
importances = final_model.feature_importances_
assert len(importances) == len(all_feature_names),
feature_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=True)
plt.figure(figsize=(10, 6))
plt.barh(feature_df['Feature'], feature_df['Importance'])
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importances")
plt.tight_layout()
plt.show()
```
""")
# Defining numeric and categorical features
numeric_features = [
    "age",
    "bmi",
    "children",
    "bmi_smoker",
    "age_smoker",
    "age_bmi",
    "children_per_age",
]
categorical_features = ["age_group", "sex", "smoker", "region"]
# Accessing OneHotEncoder from the fitted preprocessor
ohe = preprocessor.named_transformers_["cat"]
encoded_cat_features = ohe.get_feature_names_out(categorical_features)

# Combining all feature names
all_feature_names = list(numeric_features) + list(encoded_cat_features)

# Getting feature importances
importances = final_model.feature_importances_
assert len(importances) == len(all_feature_names), (
    "Mismatch between importances and feature names"
)

# Creating a dataframe
feature_df = pd.DataFrame(
    {"Feature": all_feature_names, "Importance": importances}
).sort_values(by="Importance", ascending=True)

# Creating the figure
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(feature_df["Feature"], feature_df["Importance"])
ax.set_xlabel("Feature Importance")
ax.set_title("Random Forest Feature Importances")
plt.tight_layout()

# Displaying the plot.
st.pyplot(fig)

st.markdown(
    "<h4 style='color: #B22000;'>LightGBM Regressor</h4>",
    unsafe_allow_html=True,
)

st.markdown("""
- Gradient boosting algorithm that builds trees leaf-wise for high accuracy.
- Fast and efficient, especially on large datasets.
- Learns from previous errors and improves iteratively.
""")

st.markdown(
    "<h4 style='color: #B22000;'>Fitting LIGHTGBM model</h4>",
    unsafe_allow_html=True,
)
st.markdown("""
```python
#Initializing lightGBM regressor
lgbm_model = LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    verbose = -1
)
#Fitting the model
lgbm_model.fit(X_train, y_train)
#Making predictions and evaluation
y_pred_lgbm = lgbm_model.predict(X_test)
mse_lgbm = mean_squared_error(y_test, y_pred_lgbm , squared = False)
r2_lgbm = r2_score(y_test, y_pred_lgbm)
print(f"LightGBM RMSE: {mse_lgbm:.2f}")
print(f"LightGBM R²: {r2_lgbm:.2f}")
r2_adj = adjusted_r2(r2_lgbm)
print(f"light gbm model adjusted R2: {r2_adj:.3f}")
```
""")
# Initializing lightGBM regressor
lgbm_model = LGBMRegressor(
    n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, verbose=-1
)
# Fitting the model
lgbm_model.fit(X_train, y_train)
# Making predictions and evaluation
y_pred_lgbm = lgbm_model.predict(X_test)
mse_lgbm = np.sqrt(mean_squared_error(y_test, y_pred_lgbm))
r2_lgbm = r2_score(y_test, y_pred_lgbm)
st.markdown(f"LightGBM RMSE: {mse_lgbm:.2f}")
st.markdown(f"LightGBM R²: {r2_lgbm:.2f}")
r2_adj = adjusted_r2(r2_lgbm)
st.markdown(f"light gbm model adjusted R2: {r2_adj:.3f}")


st.markdown(
    "<h4 style='color: #B22000;'>XGBoost Regressor</h4>",
    unsafe_allow_html=True,
)
st.markdown("""
- Another boosting model that adds regularization for better generalization.
- Builds trees sequentially to correct mistakes.
- Known for high accuracy in structured/tabular datasets.
""")

st.markdown(
    "<h4 style='color: #B22000;'>Fitting XGBOOST Regressor model</h4>",
    unsafe_allow_html=True,
)

st.markdown("""
```python
# Initialize XGBoost Regressor
xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
#Fitting the model
xgb_model.fit(X_train, y_train)
#Predictions and evaluation
y_pred_xgb = xgb_model.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb, squared = False)
r2_xgb = r2_score(y_test, y_pred_xgb)
print(f"XGBoost MSE: {mse_xgb:.2f}")
print(f"XGBoost R²: {r2_xgb:.2f}")
r2_adj = adjusted_r2(r2_xgb)
print(f"linear model adjusted R2: {r2_adj:.3f}")
```
""")
# Initialize XGBoost Regressor
xgb_model = XGBRegressor(
    n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
)
# Fitting the model
xgb_model.fit(X_train, y_train)
# Predictions and evaluation
y_pred_xgb = xgb_model.predict(X_test)
mse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)
st.markdown(f"XGBoost MSE: {mse_xgb:.2f}")
st.markdown(f"XGBoost R²: {r2_xgb:.2f}")
r2_adj = adjusted_r2(r2_xgb)
st.markdown(f"linear model adjusted R2: {r2_adj:.3f}")
st.markdown("---")

st.markdown(
    "<h4 style='color: #B22000;'>Model evaluation summary</h4>",
    unsafe_allow_html=True,
)

st.markdown("""
            
| Model                     | R² Score | Adjusted R² | RMSE     |
|--------------------------|----------|-------------|----------|
| Linear Regression        | 0.865    | 0.859       | 4577.96  |
| Base Random Forest       | 0.870    | 0.863       | 4509.22  |
| **Tuned Random Forest**  | **0.880**| **0.875**   | **4315.11** |
| LightGBM                 | 0.880    | 0.870       | 4400.60  |
| XGBoost                  | 0.870    | 0.867       | 4442.12  |

""")
st.markdown("---")

st.markdown("<h4 style='color: #B22000; '>Conclusion</h4>", unsafe_allow_html=True)

st.markdown("""

> **Final Model Chosen**: *Tuned Random Forest Regressor* with best performance across RMSE and R².

**Saving tuned `RandomForest` regressor model**
```python
    #Saving the model as a pickel file.
    joblib.dump(final_model, "final_model.pkl")
    #Saving the preprocessor as a pickel file.
    joblib.dump(preprocessor, "preprocessor.pkl")
```

**Note: Do not forget to review my app on the next page**
""")
st.markdown("---")
