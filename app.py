import requests
import streamlit as st
from PIL import Image
import time
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def perform_inference(criteria):
    raw_data = pd.read_csv('kobe_dataset.csv', delimiter=',')

    data = raw_data.dropna()
    request_data = data.query('shot_type == "{}"'.format(criteria))
    # request_data = data[[
    #     'shot_type',
    #     'minutes_remaining',
    #     'period',
    #     'playoffs',
    #     'shot_distance',
    #     'shot_made_flag'
    # ]]

    samples = request_data.sample(n=5412, random_state=time.localtime().tm_sec)
    request_samples = samples
    response_samples = samples
    request_samples = request_samples[[
        'lat',
        'lon',
        'minutes_remaining',
        'period',
        'playoffs',
        'shot_distance',
    ]]
    # request_samples = request_samples.drop('shot_made_flag', axis=1)

    payload = {
        'dataframe_records': request_samples.to_dict(orient='records')
    }
    # st.warning(request_data.shape[0])

    url = 'http://127.0.0.1:5001/invocations'
    response = requests.post(url, json=payload)
    # results = response.json()
    # pred = results['predictions'][0]
    
    # st.success('Done!')
    if response.status_code == 200:
        predicted_series = pd.json_normalize(response.json())
        # y_pred = predicted_series['predictions'].squeeze()

        response_samples = response_samples[[
            'shot_type',
            # 'minutes_remaining',
            # 'period',
            # 'playoffs',
            # 'shot_distance',
            'shot_made_flag'
        ]]
        
        response_samples['prediction'] = predicted_series['predictions'][0]

        # st.warning(predicted_series['predictions'][0])
        # st.warning(response_samples.shape[0])
        st.success('Previsão "{}" realizada!'.format(criteria))
        st.warning(
            'Total de arremessos: {}\n\rArremessos convertidos: {}\n\rArremessos perdidos: {}'.format(
                request_samples.shape[0],
                response_samples['prediction'].value_counts()[0],
                response_samples['prediction'].value_counts()[1]
            )
        )

        st.dataframe(response_samples, use_container_width=True, height=150)

        # Define as classes e a matriz de confusão
        classes = ['Arremessos perdidos', 'Arremessos convertidos']
        cm = confusion_matrix(response_samples['shot_made_flag'], response_samples['prediction'])

        # Plota a matriz de confusão usando matplotlib
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        # Configura os rótulos dos eixos e o título
        ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=classes, yticklabels=classes,
            xlabel='Predição', ylabel='Real',
            title='Matriz de Confusão')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # Adiciona os valores aos quadrados
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        # Mostra a figura usando Streamlit
        st.divider()
        st.pyplot(fig)
    else: 
        st.error('Ops! Não deu para adivinhar...')

    # return pred
    # return response.status_code
    return samples

st.set_page_config(page_title='Projeto Engenharia de Machine Learning - [23E1_3]', layout='wide')

col1, col2, col3 = st.columns([1,3,1])

with col2:
    image = Image.open('marca-infnet.png')
    st.image(image, width=80)
    st.write('## Projeto Engenharia de Machine Learning - [23E1_3]')
    st.write('_Bruno Meletti_')
    st.divider()

    st.info('Prevendo resultado dos arremessos...')
    col2_1, col2_2 = st.columns([1, 1])
    with col2_1:
        predictions = perform_inference("2PT Field Goal")
    with col2_2:
        predictions = perform_inference("3PT Field Goal")

    # with st.spinner('Wait for it...'):
    #     time.sleep(5)

    #     st.success('Done!')
   
