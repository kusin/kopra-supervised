# library manipulation dataset
import pandas as pd
import numpy as np

# import library streamlit
import streamlit as st
import streamlit_extras.add_vertical_space as avs

# lib untuk analisa statistik
import scipy.stats as sc
import statsmodels.api as sm

# library data visualization
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# lib untuk praproses dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# lib untuk klasifikasi data
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, NuSVC, LinearSVC

# library untuk evaluasi model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix

# lib disble warning
import warnings
warnings.filterwarnings('ignore')

# config web streamlit
st.set_page_config(
  page_title="My Dasboard - Covid 19",
  page_icon="",
  layout="wide",
  initial_sidebar_state="auto",
  menu_items={
    "Get Help": "https://www.github.com/kusin",
    "Report a bug": "https://www.github.com/kusin",
    "About": "### Copyright 2022 all rights reserved by Aryajaya Alamsyah"
  }
)

# load dataset
dataset = pd.read_csv("dataset/ekstrasi-fitur-kopra-nonTelungkup_norm.csv")

# container-header
with st.container():
  st.markdown("## Finding the best model classification for copra type using supervised learning algorithms")
  avs.add_vertical_space(2) 

# container-dataset
with st.container():
  st.info("The dataset of copra type")
  st.dataframe(dataset, use_container_width=True)
  avs.add_vertical_space(2)

# container-model and EDA
with st.container():
  # split two columns
  col1, col2 = st.columns([0.3,0.7], gap="small")

  # create form
  with col1:
    st.info("Supervised Learning Algorithms")
    with st.form("my-form"):
      algorithms = st.selectbox(
        "Choose an algorithm", (
          "K-Nearest Neighbor",
          "Support Vector Machine",
          "Decision Tree",
          "Naive Bayes", 
          "Logistic Regression"
        ), 
        placeholder="Choose an algorithm", index=None
      )
      experiments = st.selectbox(
        "Choose an experiments",(
          "Experiment 1", 
          "Experiment 2", 
          "Experiment 3", 
          "Experiment 4", 
          "Experiment 5", 
          "Experiment 6", 
          "Experiment 7"
         ), 
        placeholder="Choose an experiments", index=None
      )
      splitting = st.selectbox(
        "Choose an train and test", ("90 - 10", "80 - 20"), 
        placeholder="Choose an train and test", index=None
      )
      submitted = st.form_submit_button(label="Submit", type="secondary", use_container_width=False)
      #st.caption("Execution time is about 5 minutes")

  # ploting EDA and results 
  with col2:

    # split to 5 tab
    st.info("Exploration Data Analysis")
    tab1, tab2, tab3, tab4,tab5= st.tabs(["Results", "Color", "Shape", "Texture", "Class"])

    # KDE-color
    with tab2:
      fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(20,12),constrained_layout=True)
      sns.kdeplot(dataset, x="Mean_R", hue="Class",  fill=True, ax=ax[0,0])
      sns.kdeplot(dataset, x="Mean_G", hue="Class", fill=True, ax=ax[0,1])
      sns.kdeplot(dataset, x="Mean_B", hue="Class", fill=True, ax=ax[0,2])
      sns.kdeplot(dataset, x="Mean_H", hue="Class",  fill=True, ax=ax[1,0])
      sns.kdeplot(dataset, x="Mean_S", hue="Class",  fill=True, ax=ax[1,1])
      sns.kdeplot(dataset, x="Mean_V", hue="Class",  fill=True, ax=ax[1,2])
      sns.kdeplot(dataset, x="Mean_Gray", hue="Class",  fill=True, ax=ax[2,0])
      sns.kdeplot(dataset, x="Standar_Deviasi", hue="Class",  fill=True, ax=ax[2,1])
      fig.tight_layout()
      st.pyplot(fig, use_container_width=True)

    # KDE-shape
    with tab3:
      fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11,3),constrained_layout=True)
      sns.kdeplot(dataset, x="Luas", hue="Class",  fill=True, ax=ax[0])
      sns.kdeplot(dataset, x="Perimeter", hue="Class",  fill=True, ax=ax[1])
      fig.tight_layout()
      st.pyplot(fig, use_container_width=True)

    # KDE-texture
    with tab4:
      fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20,8))
      sns.kdeplot(dataset, x="Contrast", hue="Class",  fill=True, ax=ax[0,0])
      sns.kdeplot(dataset, x="Dissimilarity", hue="Class",  fill=True, ax=ax[0,1])
      sns.kdeplot(dataset, x="Homogeneity", hue="Class",  fill=True, ax=ax[0,2])
      sns.kdeplot(dataset, x="Energy", hue="Class",  fill=True, ax=ax[1,0])
      sns.kdeplot(dataset, x="Correlation", hue="Class",  fill=True, ax=ax[1,1])
      fig.tight_layout()
      st.pyplot(fig, use_container_width=True)

    # Barplot
    with tab5:
      fig = px.histogram(dataset, x="Class")
      fig.update_traces(marker_color=px.colors.diverging.Tropic)
      fig.update_layout(
        xaxis_title_text='', # xaxis label
        yaxis_title_text='', # yaxis label
      )
      st.plotly_chart(fig, use_container_width=True)

    with tab1:
      # # Jika Submit
      # result_accuracy=0; result_precision=0; result_recall=0;
      # trainX=None; testX=None; trainY=None; testY=None;
      # result_KNN=None; result_SVM=None; result_C45=None; result_GNB=None; result_LR=None;
      if submitted and algorithms and experiments and splitting:
        # set variabel warna, bentuk, texture
        color = ["Mean_R", "Mean_G", "Mean_B", "Mean_H", "Mean_V", "Mean_S", "Mean_Gray", "Standar_Deviasi"]
        shape = ["Luas", "Perimeter"]
        texture = ["Contrast", "Dissimilarity", "Homogeneity", "Energy", "Correlation"]
        
        # 1. Set feature and Labels
        # --------------------------------------------
        if experiments == "Experiment 1":
          #x = color
          x = dataset[color].values
        if experiments == "Experiment 2":
          x = dataset[shape].values
        if experiments == "Experiment 3":
          x = dataset[texture].values
        if experiments == "Experiment 4":
          x = dataset[color + shape].values
        if experiments == "Experiment 5":
          x = dataset[color + texture].values
        if experiments == "Experiment 6":
          x = dataset[shape + texture].values
        if experiments == "Experiment 7":
          x = dataset[color + shape + texture].values
        
        # the target class
        y = dataset[["Class"]].values

        # 2. train and test
        # --------------------------------------------
        # split validation
        if splitting == "90 - 10":
          trainX, testX, trainY, testY = train_test_split(x, y, train_size=0.9, test_size=0.1, random_state=0, shuffle=True)
          
        # split validation
        if splitting == "80 - 20":
          trainX, testX, trainY, testY = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=0, shuffle=True)

        # 3. algorithms
        # --------------------------------------------
        if algorithms == "K-Nearest Neighbor":
          result = KNeighborsClassifier(n_neighbors=3).fit(trainX, trainY).predict(testX)

        if algorithms == "Support Vector Machine":
          result = SVC(kernel='linear').fit(trainX, trainY).predict(testX)
        
        if algorithms == "Decision Tree":
          result = DecisionTreeClassifier(criterion="gini", random_state=0).fit(trainX, trainY).predict(testX)
        
        if algorithms == "Naive Bayes":
          result = GaussianNB().fit(trainX, trainY).predict(testX)
        
        if algorithms == "Logistic Regression":
          result = LogisticRegression(max_iter=1000).fit(trainX, trainY).predict(testX)


        # Results Accuracy
        st.text("Accuracy \t :"+str(
          np.round(accuracy_score(testY, result),5)
        ))
        
        # Results Precisiom
        st.text("Precision \t :"+str(
          np.round(precision_score(testY, result, average="macro"),5)
        ))

        # Results Recall
        st.text("Recall \t\t :"+str(
          np.round(recall_score(testY, result, average="macro"),5)
        ))

        # Results F1-Score
        st.text("F1-Score \t :"+str(
          np.round(f1_score(testY, result, average="macro"),5)
        ))
    