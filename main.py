import iris
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder

sns.set_theme(color_codes=True)
##ST
st.set_page_config(page_title="Behavioural_Analysis", page_icon=":tada:", layout="wide")
df = pd.read_csv('Training Data.csv')
df.head()
st.header("Printing head()")
st.dataframe(df.head())

st.header("Data Preprocessing Part 1")
# Remove unnecessary attribute
df.drop(columns=['Id','CITY'], inplace=True)
df.head()
st.dataframe(df.head())

#Check the number of unique value from all of the object datatype
st.write("Check the number of unique value from all of the object datatype")
df.select_dtypes(include='object').nunique()
st.dataframe(df.select_dtypes(include='object').nunique())

st.header("Segment Profession into smaller unique value")
st.dataframe(df['Profession'].unique())

def segment_profession(profession):
    if profession in ['Mechanical_engineer', 'Design_Engineer', 'Chemical_engineer', 'Biomedical_Engineer',
                      'Computer_hardware_engineer', 'Petroleum_Engineer', 'Civil_engineer',
                      'Industrial_Engineer', 'Technology_specialist']:
        return 'Engineering'
    elif profession in ['Software_Developer', 'Technical_writer', 'Graphic_Designer', 'Web_designer']:
        return 'IT/Software'
    elif profession in ['Civil_servant', 'Politician', 'Police_officer', 'Magistrate', 'Official', 'Army_officer']:
        return 'Government'
    elif profession in ['Librarian', 'Teacher']:
        return 'Education'
    elif profession in ['Economist', 'Financial_Analyst']:
        return 'Finance'
    elif profession in ['Flight_attendant', 'Air_traffic_controller', 'Aviator']:
        return 'Aviation'
    elif profession in ['Architect', 'Designer', 'Fashion_Designer']:
        return 'Design'
    elif profession in ['Physician', 'Dentist', 'Surgeon']:
        return 'Medical'
    elif profession in ['Hotel_Manager', 'Chef']:
        return 'Hospitality'
    elif profession == 'Artist':
        return 'Art'
    elif profession in ['Comedian', 'Psychologist']:
        return 'Entertainment'
    elif profession in ['Secretary', 'Computer_operator']:
        return 'Administration'
    elif profession in ['Chartered_Accountant', 'Analyst']:
        return 'Finance/Accounting'
    elif profession in ['Technician', 'Microbiologist', 'Scientist', 'Geologist', 'Statistician']:
        return 'Science/Research'
    else:
        return 'Other'

# Apply the segmentation function to the array of unique values
df['Profession'] = df['Profession'].apply(segment_profession)

plt.figure(figsize=(10,5))
df['Profession'].value_counts().plot(kind='bar')
fig, ax = plt.subplots(figsize=(10, 5))
ax = df['Profession'].value_counts().plot(kind='bar')
st.pyplot(fig)


st.header("Segment State into smaller unique value")
df['STATE'].unique()
st.dataframe(df['STATE'].unique())

def segment_state(state):
    if state in ['Madhya_Pradesh', 'Maharashtra', 'Kerala', 'Odisha', 'Tamil_Nadu']:
        return 'South/Central India'
    elif state in ['Gujarat', 'Rajasthan']:
        return 'West India'
    elif state in ['Telangana', 'Andhra_Pradesh']:
        return 'Telugu States'
    elif state in ['Bihar', 'West_Bengal', 'Haryana', 'Puducherry', 'Uttar_Pradesh']:
        return 'North India'
    elif state in ['Himachal_Pradesh', 'Punjab', 'Uttarakhand']:
        return 'Northwest India'
    elif state in ['Tripura', 'Jharkhand', 'Mizoram', 'Assam', 'Jammu_and_Kashmir']:
        return 'Northeast India'
    elif state in ['Delhi', 'Chhattisgarh', 'Chandigarh']:
        return 'Central India'
    elif state in ['Uttar_Pradesh[5]', 'Manipur', 'Sikkim']:
        return 'Other'
    else:
        return 'Unknown'

# Apply the segmentation function to the array of unique values
df['STATE'] = df['STATE'].apply(segment_state)

plt.figure(figsize=(10,5))
df['STATE'].value_counts().plot(kind='bar')
fig, ax = plt.subplots(figsize=(10, 5))
ax = df['STATE'].value_counts().plot(kind='bar')
st.pyplot(fig)

st.header("After processing data of STATE and PROFESSION")
st.dataframe(df.head())

st.header("Exploratory Data Analysis")
# list of categorical variables to plot
cat_vars = ['Married/Single', 'House_Ownership', 'Car_Ownership',
            'Profession', 'STATE']

# create figure with subplots
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axs = axs.flatten()

# create barplot for each categorical variable
for i, var in enumerate(cat_vars):
    sns.countplot(x=var, hue='Risk_Flag', data=df, ax=axs[i])
    axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=90)

# adjust spacing between subplots
fig.tight_layout()

# remove the sixth subplot
fig.delaxes(axs[5])

# show plot
plt.show()

cat_vars = ['Married/Single', 'House_Ownership', 'Car_Ownership',
            'Profession', 'STATE']

# create figure with subplots
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axs = axs.flatten()

# create barplot for each categorical variable
for i, var in enumerate(cat_vars):
    sns.countplot(x=var, hue='Risk_Flag', data=df, ax=axs[i])
    axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=90)

# adjust spacing between subplots
fig.tight_layout()

# remove the sixth subplot
fig.delaxes(axs[5])

st.pyplot(fig)

import warnings
warnings.filterwarnings("ignore")
# get list of categorical variables
cat_vars = ['Married/Single', 'House_Ownership', 'Car_Ownership',
            'Profession', 'STATE']

# create figure with subplots
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axs = axs.flatten()

# create histplot for each categorical variable
for i, var in enumerate(cat_vars):
    sns.histplot(x=var, hue='Risk_Flag', data=df, ax=axs[i], multiple="fill", kde=False, element="bars", fill=True, stat='density')
    axs[i].set_xticklabels(df[var].unique(), rotation=90)
    axs[i].set_xlabel(var)

# adjust spacing between subplots
fig.tight_layout()

# remove the sixth subplot
fig.delaxes(axs[5])

# show plot
plt.show()

cat_vars = ['Married/Single', 'House_Ownership', 'Car_Ownership',
            'Profession', 'STATE']

# create figure with subplots
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axs = axs.flatten()

# create histplot for each categorical variable
for i, var in enumerate(cat_vars):
    sns.histplot(x=var, hue='Risk_Flag', data=df, ax=axs[i], multiple="fill", kde=False, element="bars", fill=True, stat='density')
    axs[i].set_xticklabels(df[var].unique(), rotation=90)
    axs[i].set_xlabel(var)

# adjust spacing between subplots
fig.tight_layout()

# remove the sixth subplot
fig.delaxes(axs[5])

st.pyplot(fig)

# Specify the maximum number of categories to show individually
max_categories = 5

cat_vars = ['Married/Single', 'House_Ownership', 'Car_Ownership',
            'Profession', 'STATE']

# Create a figure and axes
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 15))

# Create a pie chart for each categorical variable
for i, var in enumerate(cat_vars):
    if i < len(axs.flat):
        # Count the number of occurrences for each category
        cat_counts = df[var].value_counts()

        # Group categories beyond the top max_categories as 'Other'
        if len(cat_counts) > max_categories:
            top_categories = cat_counts[:max_categories]
            other_count = cat_counts[max_categories:].sum()
            cat_counts = pd.concat([top_categories, pd.Series(other_count, index=['Other'])])

        # Create a pie chart
        axs.flat[i].pie(cat_counts, labels=cat_counts.index, autopct='%1.1f%%', startangle=90)

        # Set a title for each subplot
        axs.flat[i].set_title(f'{var} Distribution')

# Adjust spacing between subplots
fig.tight_layout()

# remove sixth plot
fig.delaxes(axs[1][2])

# Display the plot using Streamlit
st.pyplot(fig)

num_vars = ['Income', 'Age', 'Experience', 'CURRENT_JOB_YRS', 'CURRENT_HOUSE_YRS']

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
axs = axs.flatten()

for i, var in enumerate(num_vars):
    sns.boxplot(x=var, data=df, ax=axs[i])

fig.tight_layout()

# remove the sixth subplot
fig.delaxes(axs[5])

# Display the plot using Streamlit
st.pyplot(fig)

num_vars = ['Income', 'Age', 'Experience', 'CURRENT_JOB_YRS', 'CURRENT_HOUSE_YRS']

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 20))
axs = axs.flatten()

for i, var in enumerate(num_vars):
    sns.boxplot(y=var, x='Risk_Flag', data=df, ax=axs[i])

fig.tight_layout()

# remove the sixth subplot
fig.delaxes(axs[5])

# Display the plot using Streamlit
st.pyplot(fig)

num_vars = ['Income', 'Age', 'Experience', 'CURRENT_JOB_YRS', 'CURRENT_HOUSE_YRS']

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
axs = axs.flatten()

for i, var in enumerate(num_vars):
    sns.violinplot(x=var, data=df, ax=axs[i])

fig.tight_layout()

# remove the sixth subplot
fig.delaxes(axs[5])

# Display the plot using Streamlit
st.pyplot(fig)

num_vars = ['Income', 'Age', 'Experience', 'CURRENT_JOB_YRS', 'CURRENT_HOUSE_YRS']

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 20))
axs = axs.flatten()

for i, var in enumerate(num_vars):
    sns.violinplot(y=var, data=df, x='Risk_Flag', ax=axs[i])

fig.tight_layout()

# remove the sixth subplot
fig.delaxes(axs[5])

# Display the plot using Streamlit
st.pyplot(fig)

st.header("Data Preprocessing Part 2")

#Check missing value
check_missing = df.isnull().sum() * 100 / df.shape[0]
check_missing[check_missing > 0].sort_values(ascending=False)

# Loop over each column in the DataFrame where dtype is 'object'
for col in df.select_dtypes(include=['object']).columns:
    # Print the column name and the unique values
    print(f"{col}: {df[col].unique()}")

    # Loop over each column in the DataFrame where dtype is 'object'
    for col in df.select_dtypes(include=['object']).columns:
        # Get the unique values for the column
        unique_values = df[col].unique()

        # Display the column name
        st.write(f"Unique values for column '{col}':")

        # Display each unique value in a formatted way
        for val in unique_values:
            st.write(f"- {val}")

        # Add a separator between columns
        st.write("---")

# Encode categorical columns to numeric
for col in df.select_dtypes(include=['object']).columns:
    label_encoder = LabelEncoder()
    df[col] = label_encoder.fit_transform(df[col])

# Display the correlation heatmap
plt.figure(figsize=(20, 16))
heatmap = sns.heatmap(df.corr(), annot=True, fmt='.2g')

# Show the heatmap in Streamlit
st.pyplot(heatmap.figure)

st.header("Train Test Split")

from sklearn.model_selection import train_test_split
# Select the features (X) and the target variable (y)
X = df.drop('Risk_Flag', axis=1)
y = df['Risk_Flag']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

st.write("Decision Tree")
