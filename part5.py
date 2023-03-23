import sklearn
import numpy as np
import streamlit as st
import joblib

model = joblib.load("final_model.pkl")

input_data = (67.00,282.12,78.85,95.79,13.96,17.94,27.14,0.30,254.75,20.31,752.94,0.00,22.95,0.0,0.0)


def rh_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    prediction = model.predict(input_data_reshaped)

    return prediction


def main():
    # giving a title
    st.title('Prediction of Relative humidity using air quality data ')
    st.subheader('Main Project-Group 3')

    st.sidebar.info("This web app is made as part of Presence Of Main Project")
    st.sidebar.info("Give all the information in beside coloumn")
    st.sidebar.info("Click on the 'Predict' button to get relative humidity ")

    # getting the input data from the user
    pm2_5=st.text_input("PM 2.5") #1
    pm10=st.text_input("PM 10") #2
    no=st.text_input("NO")#3
    nox = st.text_input("NOx")  # 4
    ozone = st.text_input("OZONE")  # 5
    toluene = st.text_input("TOLUENE")  # 6
    temp = st.text_input("Temp")  # 7
    ws= st.text_input("WIND SPEED")  # 8
    wd = st.text_input("WIND DIRECTION")  # 9
    sr = st.text_input("SOLAR RADIATION")  # 10
    bp= st.text_input("BAROMETRIC PRESSURE")  # 11
    vws = st.text_input("vws")  # 12
    at=st.text_input("AT")#13
    rf=st.text_input("RAINFALL")#14
    totrf = st.text_input("TOTAL RAINFALL")#15

    # code for Prediction
    diagnosis = ''

    # creating a button for Prediction

    if st.button('Predict'):
        diagnosis = rh_prediction([pm2_5,pm10,no,nox,ozone,toluene,temp,ws,wd,sr,bp,vws,at,rf,totrf])
    st.success(diagnosis)


if __name__ == '__main__':
    main()