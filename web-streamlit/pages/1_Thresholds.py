import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.metrics import roc_auc_score, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt


st.set_page_config(layout="wide")


@st.cache_data
def thresh():
    # Чтение файлов csv
    y_prob = pd.read_csv('C:/Users/zorik/PycharmProjects/LZA/data_to_streamlit/val_proba.csv').to_numpy()
    y_true = pd.read_csv('C:/Users/zorik/PycharmProjects/LZA/data_to_streamlit/val_y.csv').to_numpy()

    # Имена колонок
    columns = pd.read_csv('C:/Users/zorik/PycharmProjects/LZA/data_to_streamlit/val_y.csv').columns

    # Сортировка по убыванию вероятностей y_prob и y_true
    for i in range(len(columns)):
        p = y_prob[:, i].argsort()
        y_prob[:, i] = y_prob[:, i][p]
        y_true[:, i] = y_true[:, i][p]

    y_prob = y_prob[::-1, :]
    y_true = y_true[::-1, :]

    # Куммулятивные суммы для 1 и 0
    y_true_cum = np.cumsum(y_true == 1, axis=0)
    y_true_inv_cum = np.cumsum(y_true == 0, axis=0)

    # Классы TP TN FP FN
    tp = y_true_cum
    fp = y_true_inv_cum
    tn = y_true_cum.copy()
    fn = y_true_cum.copy()

    for i in range(len(columns)):
        tn[:, i] = y_true_inv_cum[:, i][-1] - y_true_inv_cum[:, i]
        fn[:, i] = y_true_cum[:, i][-1] - y_true_cum[:, i]

    # Вычисление Precision
    prec = np.divide(tp, np.add(tp, fp))

    # Вычисление Recall
    rec = np.divide(tp, np.add(tp, fn))

    # Вычисление FPR
    fpr = np.divide(fp, np.add(fp, tn))

    # Имена таргетов
    columns = pd.read_csv('C:/Users/zorik/PycharmProjects/LZA/data_to_streamlit/val_proba.csv').columns

    dct = dict()
    for i, j in enumerate(columns):
        dct[j] = {'y_prob': y_prob[:, i],
                  'y_true': y_true[:, i],
                  'Precision': prec[:, i],
                  'Recall': rec[:, i],
                  'FPR': fpr[:, i]}

    return dct


dct = thresh()

st.markdown('# Подбор оптимальных метрик для таргетов')

targets = st.selectbox("##### Выберите таргет", tuple(dct.keys()))

if not targets:
    st.error("Please select at least one feature.")

else:
    data = dict()

    data[targets] = dct[targets]

    for i, j in data.items():
        st.markdown(f'### Таргет - {i.lower()}')
        st.write(f'##### Значение AUC-ROC для таргета составляет',
                 np.round(roc_auc_score(j['y_true'], j['y_prob']), 3))

        threshold = st.slider('##### Выберите отсечку (threshold):',
                              min_value=0.0, max_value=1.0, step=0.001, value=0.5)

        index = np.argmax(j['y_prob'] < threshold)
        index_med = np.argmax(j['y_prob'] < 0.5)

        col1, col2, col3 = st.columns([1, 1, 1])

        col1.subheader("Precision-Recall кривая")
        col1.line_chart(pd.DataFrame(j), x='Recall', y='Precision', color=['#87CEEB'])

        col2.subheader("ROC кривая")
        col2.line_chart(pd.DataFrame(j).rename(columns={"Recall": "TPR"}), x='FPR', y='TPR', color=['#87CEEB'])

        col3.subheader("Confusion matrix")
        conf_matrix = confusion_matrix(j['y_true'], j['y_prob'] > threshold)
        fig, ax = plt.subplots(figsize=(8, 6))
        display = ConfusionMatrixDisplay(conf_matrix)
        plt.grid(False)
        plt.style.use('dark_background')
        params = {"ytick.color": "w",
                  "xtick.color": "w",
                  "axes.labelcolor": "w",
                  "axes.edgecolor": "w"}
        plt.rcParams.update(params)
        display.plot(ax=ax, cmap=plt.cm.Blues)
        col3.pyplot(fig)

        # st.write(f'##### Значение Precision для таргета {i.lower()} при отсечке {threshold} составляет',
        #          np.round(j['Precision'][index], 3))
        # st.write(f'##### Значение Recall для таргета {i.lower()} при отсечке {threshold} составляет',
        #          np.round(j['Recall'][index], 3))
        # st.write(f'##### Значение F1-score для таргета {i.lower()} при отсечке {threshold} составляет',
        #          np.round(2 * j['Recall'][index] * j['Precision'][index] /
        #                   (j['Recall'][index] + j['Precision'][index]), 3))

        st.write(f'##### Поведение метрик при пороге классификации :blue[{threshold}]:')

        source = pd.DataFrame({'Метрики': ['Precision', 'Recall'],
                               'Значения': [np.round(j['Precision'][index], 3), np.round(j['Recall'][index], 3)]})

        col1, col2, col3, col4 = st.columns([8, 1, 1, 1])

        chart = alt.Chart(source).mark_bar().encode(
            x='Значения',
            y='Метрики',
            color='Метрики'
        ).properties(height=150)

        col1.altair_chart(chart, theme="streamlit", use_container_width=True)

        prec_metric = np.round(j['Precision'][index], 3)
        rec_metric = np.round(j['Recall'][index], 3)
        f1_metric = np.round(
            2 * j['Recall'][index] * j['Precision'][index] / (j['Recall'][index] + j['Precision'][index]), 3)

        prec_minus = np.round(j['Precision'][index_med], 3)
        rec_minus = np.round(j['Recall'][index_med], 3)
        f1_minus = np.round(
            2 * j['Recall'][index_med] * j['Precision'][index_med] / (
                        j['Recall'][index_med] + j['Precision'][index_med]), 3)

        prec_delta = prec_metric - prec_minus
        rec_delta = rec_metric - rec_minus
        f1_delta = f1_metric - f1_minus

        col2.metric("Precision",
                    f"{prec_metric}",
                    f"{np.round(prec_delta, 3)}",
                    help='Изменение метрики относительно порога 0.5')
        col3.metric("Recall",
                    f"{rec_metric}",
                    f"{np.round(rec_delta, 3)}",
                    help='Изменение метрики относительно порога 0.5')
        col4.metric("F1-score",
                    f"{f1_metric}",
                    f"{np.round(f1_delta, 3)}",
                    help='Изменение метрики относительно порога 0.5')
