import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu 
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px
from plotly import graph_objs as go 


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
#from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from prophet import Prophet
from dowhy import CausalModel
import networkx as nx
import numpy as np
import pickle
import time
from io import BytesIO
import utils

#-----------Web page setting-------------------#
page_title = "Gritap AMR"
page_icon = "💊💉🧫🦠"
picker_icon = "👇"
#layout = "centered"

#--------------------Page configuration------------------#
st.set_page_config(page_title = page_title, page_icon = page_icon, layout = "wide")

#--------------------Web App Design----------------------#

selected = option_menu(
    menu_title = page_title + " " + page_icon,
    options = ['Data Analysis', 'Train Model', 'Make a Forecast', 'Make Prediction'],
    icons = ["bi-bar-chart-fill", "bi-database-fill-gear", "bi-activity", "bi-filetype-ai"],
    default_index = 0,
    orientation = "horizontal"
)

@st.cache_data
# Load data function
def load_data(antibiotic):
    link = antibiotic + "_subset_clean.csv"
    data = pd.read_csv(link)
    return data

# Some lists
# 1. Antibiotic List
antiotics_list = ['Amikacin', 'Azithromycin']    


if selected == 'Data Analysis':
    with st.sidebar:
        st.header("🧪 Data Analysis Guide")
        st.markdown("This page helps you analyze antimicrobial resistance across demographics, bacteria, and antibiotics.")

        with st.expander("Step-by-Step Instructions"):
            st.markdown("""
            1. **Select an Antibiotic** from the dropdown list.
            2. **Choose an Analysis Type**:
                - **Demographical Analysis**: Understand how resistance varies by age, gender, etc.
                - **Bacterial Analysis**: Explore resistance across different bacterial species.
                - **Antibiotic Resistance Analysis**: See how resistance patterns evolve over time for the selected antibiotic.
            3. **Interpret Results**:
                - Hover over the interactive charts for more details.
                - Below each chart, read:
                    - 🧐 **Observations**
                    - 💡 **Implications**
                    - ✅ **Recommendations**
            """)

        st.success("")

    # Anaysis starts here
    st.title("Data Analysis")
    st.subheader("Explore and visualize antibiotic resistance data")
    # Select Antibiotic
    antibiotic = st.selectbox("Select Antibiotic", antiotics_list)

    # Load data
    data = load_data(f"{antibiotic}/{antibiotic}")

    # Analysis options
    analysis_type = st.selectbox("Select Analysis Type", ["Demographical Analysis", "Bacterial Species Analysis", "Antibiotic Resistance Analysis"])

    if analysis_type == "Demographical Analysis":
        st.subheader("Demographical Analysis")
        st.write("This section allows you to analyze the demographic data of patients")
        
        #Plot Gender Distribution
        utils.gender_distribution(data)

        #Plot Age Distribution
        utils.age_distribution(data)

        #Country Analysis
        utils.country_analysis(data)
        
        # Continent Analysis
        utils.continent_analysis(data)

        # Patient type Analysis
        utils.patient_type_analysis(data)

    elif analysis_type == "Bacterial Species Analysis":
        st.subheader("Bacteria (Species) Analysis")
        st.write("This section allows you to analyze the bacterial species data")
        
        # Top 10 Species Analysis
        utils.top_10_species_analysis(data, antibiotic)
        
        # Species per Country Analysis
        utils.organism_distribution_by_country(data)

        # Gender Distribution
        utils.organism_distribution_by_gender(data)

        # Bacteria Distribution and age
        utils.organism_distribution_by_age(data)

        # Yearly Bacteria Distribution
        utils.yearly_organism_distribution(data)

        # Bacteria distribution by family
        utils.organism_distribution_by_family(data)

        # Bacteria Distribution by resistance status
        utils.species_by_resistance_status(data, antibiotic)
        
        # Bacteria Distribution to MIC Value
        utils.species_distribution_by_mic(data, antibiotic)


    elif analysis_type == "Antibiotic Resistance Analysis":
        st.subheader("Antibiotic Resistance Analysis")
        st.write("This section allows you to analyze antibiotic resistance data")

        # Antibiotic Resistance Distribution
        utils.resistance_distribution(data, antibiotic)

        # Plot MIC Distribution
        utils.mic_distribution(data, antibiotic)

        # Plot Yearly Resistance Status
        utils.yearly_resistance_status(data, antibiotic)

        # Plot Bacteria Resistance Status
        utils.bacteria_resistance_status(data, antibiotic)
        

elif selected == 'Statistical Analysis':
    with st.sidebar:
        st.header("📐 Statistical Analysis Guide")
        st.markdown("View statistical charts and highlights based on advanced Bayesian modeling.")

        with st.expander("Step-by-Step Instructions"):
            st.markdown("""
            1. **Select an Antibiotic** to analyze.
            2. **Review BHM Results**:
                - BHM-generated plots and charts will load automatically.
            3. **Interpret Each Chart**:
                - Read the following provided for every chart:
                    - 🔍 **Observations**
                    - 📌 **Key Insights**
                    - 🧠 **Implications**
            """)

        st.success("Statistical power unlocked! Let the Bayesian magic guide your decisions 🧙‍♂️📊")

    # Statistical Analysis starts here
    st.title("Statistical Analysis")

    # Select Antibiotic
    antibiotic = st.selectbox("Select Antibiotic", antiotics_list)

    # Show statistical analysis
    utils.statistic_analysis(antibiotic)

elif selected == 'Train Model':
    with st.sidebar:
        st.header("🤖 Train a Model Guide")
        st.markdown("Train classification models and evaluate them with interactive metrics and charts.")

        with st.expander("Step-by-Step Instructions"):
            st.markdown("""
            1. **Select Inputs**:
                - Choose an **Antibiotic**.
                - Select a **Bacteria (Species)**.
                - Pick one of the 9 available **ML Algorithms**.
            2. **Training Conditions**:
                - If the selected bacteria only has one resistance status, a message will notify you — no training needed.
            3. **Click the 'Train Model' Button** to:
                - Train the model and make predictions.
                - View:
                    - 📈 **Accuracy, Precision, Recall, F1-Score**
                    - 📊 **Interactive Confusion Matrix**
                    - ⭐ **Feature Importance Chart** (if available)
                    - 🧪 **Causal Effects Chart** showing the influence of Phenotype, Source, and Country.
            4. **Download the Trained Model** as a `.pkl` file using the provided button.
            """)

        st.success("")

    # Train Model starts here
    st.title("Train Model")
    st.subheader("Train machine learning models on antibiotic resistance data")
    # Select Antibiotic
    antibiotic = st.selectbox("Select Antibiotic", antiotics_list)

    # Load data
    data = load_data(f"{antibiotic}/{antibiotic}")
    # Fill all `NaN` values with mode in all columns that have NaN values
    for col in data.columns:
        if data[col].dtype == 'object' and data[col].isnull().any():
            data[col].fillna(data[col].mode()[0], inplace=True)

    # Prediction
    utils.train_model(data, antibiotic)
    
# Make a Forecast page
elif selected == 'Make a Forecast':
    with st.sidebar:
        st.header("📅 Forecasting Guide")
        st.markdown("Predict future resistance trends using time-series modeling.")

        with st.expander("Step-by-Step Instructions"):
            st.markdown("""
            1. **Select Inputs**:
                - Choose an **Antibiotic**.
                - Select a **Bacteria (Species)**.
                - Enter the **Number of Years** you want to forecast.
            2. **Check for Data Availability**:
                - If not enough historical data is available, you’ll be notified.
            3. **Click 'Make Forecast' Button**:
                - See an **interactive line plot** of future trends.
                - Review:
                    - 🔍 **Observations**
                    - 🧠 **Implications**
                    - ✅ **Recommendations**
            """)

        st.success("")

    # Make a Forecast starts here
    st.title("Make a Forecast")

    # Select Antibiotic
    antibiotic = st.selectbox("Select Antibiotic", antiotics_list)
    # Load data
    data = load_data(f"{antibiotic}/{antibiotic}")

    # Forecast
    utils.forecast(data, antibiotic)
    

elif selected == 'Make Prediction':
    with st.sidebar:
        st.header("🔍 Make a Prediction Guide")
        st.markdown("Simulate conditions and predict bacteria resistance to selected antibiotics.")

        with st.expander("Step-by-Step Instructions"):
            st.markdown("""
            1. **Select Prediction Inputs**:
                - Choose an **Antibiotic**.
                - Select a **Bacteria (Species)**.
                - Pick additional conditions like **Year**, **Type of Study**, etc.
            2. **Click 'Make Prediction' Button**:
                - You’ll receive a message indicating the predicted resistance status:
                    - 🚦 **Resistant**, **Intermediate**, or **Susceptible**.
            3. **Explore Causal Effects**:
                - Choose a **Bacteria (Species)**.
                - Select **Treatment Factors** (e.g., Source, Phenotype).
                - Click **'Show Causal Effects'** to:
                    - View a **summary table**.
                    - See an **interactive horizontal bar chart**.
                    - Read accompanying highlights:
                        - 🔍 **Observations**
                        - 🧠 **Implications**
            """)

        st.success("")

    # Make Prediction starts here
    st.title("Make Prediction")
    st.subheader("Predict antibiotic resistance using trained models")
    # Select Antibiotic
    antibiotic = st.selectbox("Select Antibiotic", antiotics_list)
    # Load data
    data = load_data(f"{antibiotic}/{antibiotic}")

    # Prediction
    utils.make_prediction(data, antibiotic)

    # Show causal effect estimation
    st.subheader("Causal Effect Estimation")
    utils.check_causal_effect(data, antibiotic)
  







