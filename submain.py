import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import shap
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder

sns.set_theme(color_codes=True)


# Function to get or create SessionState
def get_state():
    return st.session_state

# Create functions for each page
def dataset_page():
    st.title('Dataset Page')

    state = get_state()

    if 'data' not in state:
        state.data = None

    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        state.data = data

    if state.data is not None:
        st.write("## Input Dataset")
        st.write(state.data)

        st.write("\n\n[Go to Preprocessing Page](#preprocessing)")

# Create function for the preprocessing page
def preprocessing_page():
    st.title('Preprocessing Page')

    state = get_state()

    if state.data is not None:
        st.write("## Input Dataset")
        st.write(state.data)

        st.write("## Preprocessed Data")
        st.write("Remove unnecessary attribute")
        # Drop columns 'Id' and 'CITY'
        state.data.drop(columns=['Id', 'CITY'], inplace=True)
        st.write(state.data.head())
        st.write("Check the number of unique value from all of the object datatype")
        st.dataframe(state.data.select_dtypes(include='object').nunique())
        st.write("## Segment Profession into smaller unique value")
        st.dataframe(state.data['Profession'].unique())

        def segment_profession(profession):
            if profession in ['Mechanical_engineer', 'Design_Engineer', 'Chemical_engineer', 'Biomedical_Engineer',
                              'Computer_hardware_engineer', 'Petroleum_Engineer', 'Civil_engineer',
                              'Industrial_Engineer', 'Technology_specialist']:
                return 'Engineering'
            elif profession in ['Software_Developer', 'Technical_writer', 'Graphic_Designer', 'Web_designer']:
                return 'IT/Software'
            elif profession in ['Civil_servant', 'Politician', 'Police_officer', 'Magistrate', 'Official',
                                'Army_officer']:
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

        st.write("Apply the segmentation function to the array of unique values")
        state.data['Profession'] = state.data['Profession'].apply(segment_profession)
        plt.figure(figsize=(10, 5))
        state.data['Profession'].value_counts().plot(kind='bar')
        fig, ax = plt.subplots(figsize=(10, 5))
        ax = state.data['Profession'].value_counts().plot(kind='bar')
        st.pyplot(fig)

        st.write("## Segment State into smaller unique value")
        state.data['STATE'].unique()
        st.dataframe(state.data['STATE'].unique())

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

        st.write("Apply the segmentation function to the array of unique values")
        state.data['STATE'] = state.data['STATE'].apply(segment_state)

        plt.figure(figsize=(10, 5))
        state.data['STATE'].value_counts().plot(kind='bar')
        fig, ax = plt.subplots(figsize=(10, 5))
        ax = state.data['STATE'].value_counts().plot(kind='bar')
        st.pyplot(fig)
        st.header("After processing data of STATE and PROFESSION")
        st.dataframe(state.data.head())

        st.write("## Exploratory Data Analysis")
        # list of categorical variables to plot
        cat_vars = ['Married/Single', 'House_Ownership', 'Car_Ownership',
                    'Profession', 'STATE']

        # create figure with subplots
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
        axs = axs.flatten()

        st.write("barplot for each categorical variable")
        for i, var in enumerate(cat_vars):
            sns.countplot(x=var, hue='Risk_Flag', data=state.data, ax=axs[i])
            axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=90)

        # adjust spacing between subplots
        fig.tight_layout()

        # remove the sixth subplot
        fig.delaxes(axs[5])

        st.pyplot(fig)

        cat_vars = ['Married/Single', 'House_Ownership', 'Car_Ownership',
                    'Profession', 'STATE']

        # create figure with subplots
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
        axs = axs.flatten()

        st.write("histplot for each categorical variable")
        for i, var in enumerate(cat_vars):
            sns.histplot(x=var, hue='Risk_Flag', data=state.data, ax=axs[i], multiple="fill", kde=False, element="bars",
                         fill=True, stat='density')
            axs[i].set_xticklabels(state.data[var].unique(), rotation=90)
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

        st.write("pie chart for each categorical variable")
        for i, var in enumerate(cat_vars):
            if i < len(axs.flat):
                # Count the number of occurrences for each category
                cat_counts = state.data[var].value_counts()

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
            sns.boxplot(x=var, data=state.data, ax=axs[i])

        fig.tight_layout()

        # remove the sixth subplot
        fig.delaxes(axs[5])

        # Display the plot using Streamlit
        st.pyplot(fig)

        num_vars = ['Income', 'Age', 'Experience', 'CURRENT_JOB_YRS', 'CURRENT_HOUSE_YRS']

        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 20))
        axs = axs.flatten()

        for i, var in enumerate(num_vars):
            sns.boxplot(y=var, x='Risk_Flag', data=state.data, ax=axs[i])

        fig.tight_layout()

        # remove the sixth subplot
        fig.delaxes(axs[5])

        # Display the plot using Streamlit
        st.pyplot(fig)

        num_vars = ['Income', 'Age', 'Experience', 'CURRENT_JOB_YRS', 'CURRENT_HOUSE_YRS']

        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
        axs = axs.flatten()

        for i, var in enumerate(num_vars):
            sns.violinplot(x=var, data=state.data, ax=axs[i])

        fig.tight_layout()

        # remove the sixth subplot
        fig.delaxes(axs[5])

        # Display the plot using Streamlit
        st.pyplot(fig)

        num_vars = ['Income', 'Age', 'Experience', 'CURRENT_JOB_YRS', 'CURRENT_HOUSE_YRS']

        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 20))
        axs = axs.flatten()

        for i, var in enumerate(num_vars):
            sns.violinplot(y=var, data=state.data, x='Risk_Flag', ax=axs[i])

        fig.tight_layout()

        # remove the sixth subplot
        fig.delaxes(axs[5])

        # Display the plot using Streamlit
        st.pyplot(fig)


        st.write("## Data Preprocessing Part 2")

        # Check missing value
        check_missing = state.data.isnull().sum() * 100 / state.data.shape[0]
        check_missing[check_missing > 0].sort_values(ascending=False)

        # Loop over each column in the DataFrame where dtype is 'object'
        for col in state.data.select_dtypes(include=['object']).columns:
            # Print the column name and the unique values
            print(f"{col}: {state.data[col].unique()}")

            # Loop over each column in the DataFrame where dtype is 'object'
            for col in state.data.select_dtypes(include=['object']).columns:
                # Get the unique values for the column
                unique_values = state.data[col].unique()

                # Display the column name
                st.write(f"Unique values for column '{col}':")

                # Display each unique value in a formatted way
                for val in unique_values:
                    st.write(f"- {val}")

                # Add a separator between columns
                st.write("---")

        # Encode categorical columns to numeric
        for col in state.data.select_dtypes(include=['object']).columns:
            label_encoder = LabelEncoder()
            state.data[col] = label_encoder.fit_transform(state.data[col])

        # Display the correlation heatmap
        plt.figure(figsize=(20, 16))
        heatmap = sns.heatmap(state.data.corr(), annot=True, fmt='.2g')

        # Show the heatmap in Streamlit
        st.pyplot(heatmap.figure)



def testing_page():
    st.title('Testing Page')
    # Add content for the testing page
    state = get_state()

    if state.data is not None:
        st.write("## Input Dataset")
        st.write(state.data)

        st.write("## Train Test Split")

        from sklearn.model_selection import train_test_split
        # Select the features (X) and the target variable (y)
        X = state.data.drop('Risk_Flag', axis=1)
        y = state.data['Risk_Flag']

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        from sklearn.tree import DecisionTreeClassifier
        dtree = DecisionTreeClassifier(random_state=0, max_depth=4, min_samples_leaf=1, min_samples_split=2,
                                       class_weight='balanced')
        dtree.fit(X_train, y_train)

        from sklearn.metrics import accuracy_score
        y_pred = dtree.predict(X_test)

        # Calculate and display accuracy score in Streamlit
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy Score: {round(accuracy * 100, 2)}%")

        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, jaccard_score, log_loss
        st.write('F-1 Score : ', (f1_score(y_test, y_pred, average='micro')))
        st.write('Precision Score : ', (precision_score(y_test, y_pred, average='micro')))
        st.write('Recall Score : ', (recall_score(y_test, y_pred, average='micro')))
        st.write('Jaccard Score : ', (jaccard_score(y_test, y_pred, average='micro')))
        st.write('Log Loss : ', (log_loss(y_test, y_pred)))

        # Create imp_df with feature importances
        imp_df = pd.DataFrame({
            "Feature Name": X_train.columns,
            "Importance": dtree.feature_importances_
        })
        fi = imp_df.sort_values(by="Importance", ascending=False)
        fi2 = fi.head(10)

        # Create the plot
        plt.figure(figsize=(10, 8))
        sns.barplot(data=fi2, x='Importance', y='Feature Name')
        plt.title('Top 10 Feature Importance Each Attributes (Decision Tree)', fontsize=18)
        plt.xlabel('Importance', fontsize=16)
        plt.ylabel('Feature Name', fontsize=16)

        # Display the plot in Streamlit
        st.pyplot(plt)
        st.set_option('deprecation.showPyplotGlobalUse', False)

        # Create SHAP explainer and calculate SHAP values
        explainer = shap.TreeExplainer(dtree)
        shap_values = explainer.shap_values(X_test)

        # Display SHAP summary plot using Matplotlib and Streamlit
        shap.summary_plot(shap_values, X_test, show=False)
        st.pyplot(bbox_inches='tight')

        # Create SHAP explainer and calculate SHAP values
        explainer = shap.TreeExplainer(dtree)
        shap_values = explainer.shap_values(X_test)

        # Display SHAP summary plot using Matplotlib and Streamlit
        shap.summary_plot(shap_values[1], X_test, feature_names=X_test.columns, show=False)
        st.pyplot(bbox_inches='tight')

        from sklearn.metrics import confusion_matrix
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Display confusion matrix heatmap using Matplotlib and Streamlit
        plt.figure(figsize=(5, 5))
        sns.heatmap(data=cm, linewidths=.5, annot=True, cmap='Blues')
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        all_sample_title = 'Accuracy Score for Decision Tree: {0}'.format(dtree.score(X_test, y_test))
        plt.title(all_sample_title, size=15)

        # Display the plot in Streamlit
        st.pyplot(plt)

        from sklearn.metrics import roc_curve, roc_auc_score
        y_pred_proba = dtree.predict_proba(X_test)[:, 1]
        # Create a DataFrame combining actual and predicted values
        df_actual_predicted = pd.concat([pd.DataFrame(np.array(y_test), columns=['y_actual']),
                                         pd.DataFrame(y_pred_proba, columns=['y_pred_proba'])], axis=1)
        df_actual_predicted.index = y_test.index

        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(df_actual_predicted['y_actual'], df_actual_predicted['y_pred_proba'])
        auc = roc_auc_score(df_actual_predicted['y_actual'], df_actual_predicted['y_pred_proba'])

        # Plot ROC curve with AUC
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label='AUC = %0.4f' % auc)
        plt.plot(fpr, fpr, linestyle='--', color='k')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve', size=15)
        plt.legend()

        # Display the plot in Streamlit
        st.pyplot(plt)

        st.write("## Random Forest")
        from sklearn.ensemble import RandomForestClassifier
        rfc = RandomForestClassifier(random_state=0, max_features='sqrt', n_estimators=200, class_weight='balanced')
        rfc.fit(X_train, y_train)

        y_pred = rfc.predict(X_test)
        st.write("Accuracy Score :", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, jaccard_score, log_loss
        st.write('F-1 Score : ', (f1_score(y_test, y_pred, average='micro')))
        st.write('Precision Score : ', (precision_score(y_test, y_pred, average='micro')))
        st.write('Recall Score : ', (recall_score(y_test, y_pred, average='micro')))
        st.write('Jaccard Score : ', (jaccard_score(y_test, y_pred, average='micro')))
        st.write('Log Loss : ', (log_loss(y_test, y_pred)))

        from sklearn.ensemble import RandomForestClassifier
        imp_df = pd.DataFrame({
            "Feature Name": X_train.columns,
            "Importance": rfc.feature_importances_
        })
        fi = imp_df.sort_values(by="Importance", ascending=False)
        fi2 = fi.head(10)

        # Create the plot
        plt.figure(figsize=(10, 8))
        sns.barplot(data=fi2, x='Importance', y='Feature Name')
        plt.title('Top 10 Feature Importance Each Attributes (Random Forest)', fontsize=18)
        plt.xlabel('Importance', fontsize=16)
        plt.ylabel('Feature Name', fontsize=16)

        # Display the plot in Streamlit
        st.pyplot(plt)

# Create a function to handle page navigation
def main():
    st.sidebar.title('Navigation')
    page = st.sidebar.selectbox('Select a page', ['Dataset', 'Preprocessing', 'Testing'])

    if page == 'Dataset':
        dataset_page()
    elif page == 'Preprocessing':
        preprocessing_page()
    elif page == 'Testing':
        testing_page()

if __name__ == '__main__':
    main()
