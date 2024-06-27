# import pandas as pd
# import numpy as np
# import pickle
# import streamlit as st

# st.write("""
#          # Penguin Prediction App
#          This app predict **Palmer Penguin** species!
#          """)
# st.sidebar.header('User input features')

# uploaded_file = st.sidebar.file_uploader('upload csv file', type=["csv"])
# if uploaded_file is not None:
#     input_df = pd.read_csv(uploaded_file)
# else:
#     def user_input_features():
#         island =st.sidebar.selectbox('island',('Biscoe', 'Dream', 'Torgersen')),
#         sex = st.sidebar.selectbox('sex',('male','female'))
#         bill_length_mm = st.sidebar.slider('bill_length_mm',32,59,43),
#         bill_depth_mm = st.sidebar.slider('bill_depth_mm',13,21,17),
#         flipper_length_mm = st.sidebar.slider('flipper_length_mm', 172, 231, 201),
#         body_mass_g = st.sidebar.slider('body_mass_g', 2700, 6300, 4200),
#         data = {
#             'island':island,
#             'bill_length_mm': bill_length_mm,
#              'bill_depth_mm': bill_depth_mm,
#              'flipper_length_mm':flipper_length_mm,
#              'body_mass_g':body_mass_g,
#              'sex':sex  
#         }
#         features = pd.DataFrame(data, index=[0])
#         return features
#     input_df = user_input_features()
    
# # combine user input feature with entire penguins dataset
# penguins_raw = pd.read_csv(r"C:\Users\User\Downloads\penguins_cleaned.csv")
# penguins = penguins_raw.drop(columns=['species'])
# df = pd.concat([input_df, penguins], axis=1)

# encode = ['sex', 'island']
# for col in encode:
#     dummy = pd.get_dummies(df[col], prefix=col)
#     df = pd.concat([df,dummy], axis=1)
#     del df[col]
# df = df[:1]

# st.subheader('User input features')
# if uploaded_file is not None:
#     st.write(df)
# else:
#     st.write('waiting for csv file to be uploaded. currently using examplr parameters')
#     st.write(df)
    
# load_clf = pickle.load(open('penguins.pkl', 'rb'))

# prediction = load_clf.predict(df)
# prediction_proba = load_clf.predict_proba(df)

# st.subheader('Prediction')
# penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
# st.write(penguins_species[prediction])
# st.subheader('prediction_probablity')
# st.write(prediction_proba)


# import pandas as pd
# import numpy as np
# import pickle
# import streamlit as st

# st.write("""
#          # Penguin Prediction App
#          This app predicts **Palmer Penguin** species!
#          """)
# st.sidebar.header('User input features')

# uploaded_file = st.sidebar.file_uploader('Upload CSV file', type=["csv"])
# if uploaded_file is not None:
#     input_df = pd.read_csv(uploaded_file)
# else:
#     def user_input_features():
#         island = st.sidebar.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
#         sex = st.sidebar.selectbox('Sex', ('male', 'female'))
#         bill_length_mm = st.sidebar.slider('Bill Length (mm)', 32, 59, 43)
#         bill_depth_mm = st.sidebar.slider('Bill Depth (mm)', 13, 21, 17)
#         flipper_length_mm = st.sidebar.slider('Flipper Length (mm)', 172, 231, 201)
#         body_mass_g = st.sidebar.slider('Body Mass (g)', 2700, 6300, 4200)
#         data = {
#             'island': island,
#             'bill_length_mm': bill_length_mm,
#             'bill_depth_mm': bill_depth_mm,
#             'flipper_length_mm': flipper_length_mm,
#             'body_mass_g': body_mass_g,
#             'sex': sex
#         }
#         features = pd.DataFrame(data, index=[0])
#         return features
#     input_df = user_input_features()

# # Combine user input features with entire penguins dataset
# penguins_raw = pd.read_csv(r"C:\Users\User\Downloads\penguins_cleaned.csv")
# penguins = penguins_raw.drop(columns=['species'])
# df = pd.concat([input_df, penguins], axis=0)

# # Encode categorical features
# encode = ['sex', 'island']
# for col in encode:
#     dummy = pd.get_dummies(df[col], prefix='', prefix_sep='')  # No prefix
#     df = pd.concat([df, dummy], axis=1)
#     del df[col]
# df = df[:1]  # Select only the first row (the user input data)

# st.subheader('User Input Features')
# if uploaded_file is not None:
#     st.write(df)
# else:
#     st.write('Waiting for CSV file to be uploaded. Currently using example parameters.')
#     st.write(df)

# # Load the model
# load_clf = pickle.load(open('penguins.pkl', 'rb'))

# # Ensure the input features match the training features
# missing_cols = set(load_clf.feature_names_in_) - set(df.columns)
# for col in missing_cols:
#     df[col] = 0
# df = df[load_clf.feature_names_in_]

# # Make predictions
# prediction = load_clf.predict(df)
# prediction_proba = load_clf.predict_proba(df)

# st.subheader('Prediction')
# penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
# st.write(penguins_species[prediction])
# st.subheader('Prediction Probability')
# st.write(prediction_proba)





# ----------------------------
import pandas as pd
import numpy as np
import pickle
import streamlit as st

st.write("""
         # Penguin Prediction App
         This app predicts **Palmer Penguin** species!
         """)
st.sidebar.header('User input features')

uploaded_file = st.sidebar.file_uploader('Upload CSV file', type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
        sex = st.sidebar.selectbox('Sex', ('male', 'female'))
        bill_length_mm = st.sidebar.slider('Bill Length (mm)', 32, 59, 43)
        bill_depth_mm = st.sidebar.slider('Bill Depth (mm)', 13, 21, 17)
        flipper_length_mm = st.sidebar.slider('Flipper Length (mm)', 172, 231, 201)
        body_mass_g = st.sidebar.slider('Body Mass (g)', 2700, 6300, 4200)
        data = {
            'island': island,
            'bill_length_mm': bill_length_mm,
            'bill_depth_mm': bill_depth_mm,
            'flipper_length_mm': flipper_length_mm,
            'body_mass_g': body_mass_g,
            'sex': sex
        }
        features = pd.DataFrame(data, index=[0])
        return features
input_df = user_input_features()

# Encode categorical features
encode = ['sex', 'island']
for col in encode:
    dummy = pd.get_dummies(input_df[col], prefix='',    prefix_sep='')  # No prefix
    input_df = pd.concat([input_df, dummy], axis=1)
    del input_df[col]

st.subheader('User Input Features')
if uploaded_file is not None:
    st.write(input_df)
else:
    st.write('Waiting for CSV file to be uploaded. Currently using example parameters.')
    st.write(input_df)

# Load the model
load_clf = pickle.load(open('penguins.pkl', 'rb'))

# Ensure the input features match the training features
missing_cols = set(load_clf.feature_names_in_) - set(input_df.columns)
for col in missing_cols:
    input_df[col] = 0
input_df = input_df[load_clf.feature_names_in_]

# Make predictions
prediction = load_clf.predict(input_df)
prediction_proba = load_clf.predict_proba(input_df)

st.subheader('Prediction')
penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.write(penguins_species[prediction])
st.subheader('Prediction Probability')
st.write(prediction_proba)
