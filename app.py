import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
import seaborn as sns
#Streamlit confiturations
st.set_page_config(page_title='Student performace predictore', layout='centered')
st.title('Student performance predictore')
st.write('This is web app uses a simple machine learning to predict a student average score based on input subject markes.')
#input sliders
math = st.slider('Math.Score', 0,100, 50)
science = st.slider('Science.Score', 0,100, 50)
english = st.slider('english.Score', 0,100, 50)


training = pd.DataFrame({
    'math': np.random.randint(30, 100, 100),
    'science': np.random.randint(30, 100, 100),
    'english': np.random.randint(30, 100, 100)
})

# Average for each student
training['average'] = training.mean(axis=1)
X = training[['math', 'science', 'english']]
y = training['average']

# Split into trainint and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2)

# buidl the model
model = LinearRegression()

# train the model
model.fit(X_train, y_train)

#Prediction

input_data = [[math, science, english]]
prediction = model.predict(input_data)

st.success(f'Predicted score is: {prediction}')


#Visualization
#Plotting training dat ana prediction
st.subheader('Visual comparion')
fig, ax = plt.subplots()
ax.scatter(training['math'], training['average'], color = 'blue', label = 'Training')
ax.scatter(math,prediction, color = 'red', label = 'Input', s = 100)
ax.set_xlabel('math score')
ax.set_ylabel('Predicted Average')
ax.set_title('Math vs Average prediction')
ax.legend()
st.pyplot(fig)

