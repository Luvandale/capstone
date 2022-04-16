import numpy as np
import pandas as pd
# from IPython.display import Image
import streamlit as st
# import matplotlib.pyplot as plt
# import plotly.graph_objs as go 
# from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# init_notebook_mode(connected=True)
# import seaborn as sns #remove?
from sklearn.model_selection import train_test_split

# Import label encoder
from sklearn import preprocessing
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

# scaling the data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

# Data visualization
# import matplotlib.pyplot 
from sklearn.preprocessing import StandardScaler
from keras.utils.np_utils import to_categorical
# import seaborn as sns
import numpy as np
import pandas as pd 
import tensorflow as tf
# import seaborn as sns
# Keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
from tensorflow.keras.optimizers import SGD, Adam, Adadelta, RMSprop
import keras.backend as K
# Train-Test
from sklearn.model_selection import train_test_split
# Scaling data
from sklearn.preprocessing import StandardScaler
# Classification Report
from sklearn.metrics import classification_report



country_dict = {8: 'Sudan', 7: 'South Africa', 2: 'Kenya', 6: 'Somalia', 0: 'Angola', 4: 'Mozambique', 9: 'Uganda', 5: 'Nigeria', 3: 'Mali', 1: 'Burundi', 10: 'Democratic Republic of the Congo'}

def data_processing(data_df):
  """
  used to process our data so that it can be used by the model
  """
  data_df= pd.read_csv(data_df)
  # Filling Nan values with the mean
  data_df['latitude'].fillna(-0.231070	, inplace=True)
  data_df['longitude'].fillna(25.892334, inplace=True)
  data_df['Weapon_type'].fillna(6.710219, inplace=True)
  data_df = data_df.drop([ 'Region','Motive','Summary'], axis=1)

  # Encode labels in column 'species'.
  data_df['Month']= label_encoder.fit_transform(data_df['Month'])
  # Encode labels in column 'species'.
  data_df['Group']= label_encoder.fit_transform(data_df['Group'])
  # Encode labels in column 'species'.
  data_df['Country_no']= label_encoder.fit_transform(data_df['Country_no'])
  # Encode labels in column 'species'.
  # Sub['city']= label_encoder.fit_transform(Sub['city'])

  one_hot_encoded_data = pd.get_dummies(data_df, columns = ['Month'])
  data_x = one_hot_encoded_data.drop(['Country','city','Group','Country_no'],axis=1)
  data_x = pd.DataFrame(sc_X.fit_transform(data_x))
  return data_x

def model_prediction(data_x):
  loaded_model = tf.keras.models.load_model('keras_model.h5')
  predictions = loaded_model.predict(data_x)
  return predictions

def post_prediction(file_df):
  data_x = data_processing(file_df)
  y_pred = model_prediction(data_x)
  y_pred_prob = np.max(y_pred,axis = 1)
  y_pred_class = np.argmax(y_pred,axis=1)
  # y_test_class = np.argmax(y_test, axis=1)
  y_pred_class_ev = label_encoder.inverse_transform(y_pred_class)
  conc_data = pd.concat([pd.DataFrame(data_x),pd.Series(y_pred_class)], axis =1)
  conc_data = pd.concat([conc_data,pd.Series(y_pred_prob)], axis =1)
  conc_data.columns = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,'Country','Probability']
  conc_data['Country_name']= pd.Series(country_dict)
  conc_data['Occurrence'] = np.where(conc_data['Probability']>0.8, 'yes', 'no')
  conc_data = conc_data[['Country','Probability','Country_name','Occurrence']]
  return conc_data

# title of the app
st.title("Subsaharan terrorism Predictor app")
st.markdown("Upload a csv file and will predict whether an attack will occur or not in a Subsaharan country")

# button to upload image and submit for prediction
csv_file = st.file_uploader("Choose csv file to upload", type="csv")
submit = st.button('Predict')

# on predict button clicksre
if submit:
      if csv_file is not None:
          d = post_prediction(csv_file)
          st.dataframe(d.head())
          d.to_csv("d.csv")
      #     st.download_button(
      #     label="Download data as CSV",
      #     data=d,
      #     file_name='large_df.csv',
      #     mime='text/csv',
      # )
df = pd.read_csv("d.csv")         
def convert_df(df):
       return df.to_csv().encode('utf-8')


csv = convert_df(df)

st.download_button(
   "Press to Download",
   csv,
   "file.csv",
   "text/csv",
   key='download-csv'
)

      
