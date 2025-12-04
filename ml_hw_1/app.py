import streamlit as st
import numpy as np
import pandas as pd
import pickle
import json
import plotly.express as px
import os

from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

# Для смены текущей рабочей директории
os.chdir(os.path.dirname(__file__))

def preprocess_df(df: pd.DataFrame, light_mode=False):
    """
    Предобработка загруженного датасета.
    Режимы:
        light_mode = True:
            - преобразовавыает некоторые объектовые столбцы в числовые
            - разделяет torque
        light_mode = False:
            - создает новые признаки для использования инференса модели
            
    df: Подгружаемый датасет, тип - pd.DataFrame
    light_mode: Режим работы, False по умолчанию
    """
    def clean_torque(text):
        """
        Вспомогательная функция для очистки torque признака
        """
        try:
            text = text.lower()
        except:
            return np.nan, np.nan
        # Удаляем ненужные символы
        text = text.replace("~", "-")
        text = text.replace("/", "")
        text = text.replace("@", "")
        text = text.replace(",", "")
        text = text.replace("(kgm rpm)", "kgm")
        text = text.replace("(nm rpm)", "nm")
        text = text.replace("rpm", "")
        text = text.replace("(", " ")
        
        if "nm" in text:
            text = text.replace("nm", " ")
            max_torque = float(text.split()[0])
        
        elif "kgm" in text:
            text = text.replace("kgm", " ")
            max_torque = float(text.split()[0]) * 9.80665
        
        elif int(text.split()[0]) > 40:
            max_torque = float(text.split()[0])
        
        else:
            max_torque = float(text.split()[0]) * 9.80665

        if "+-" in text:
            text = text.split()[1]
            rpm = float(text.split('+-')[0]) + float(text.split('+-')[1])
        else:
            rpm = float(text.split()[-1].split('-')[-1])
        
        if max_torque == rpm:
            rpm = np.nan
            
        return max_torque, rpm

    # Действия для light_mode
    if 'max_torque_rpm' not in df.columns:
        for col in ['mileage', 'engine', 'max_power']:
            df[col] = pd.to_numeric(df[col].str.split(' ', n = 1, expand = True)[0], errors='coerce')

        errors_in = []
        for row in df.itertuples():
            a, b = clean_torque(row.torque)
            if a > 1000 or b > 10000:
                errors_in.append(row.Index)
        df.loc[errors_in, 'torque'] = df.loc[errors_in, 'torque'].str.replace('kgm', 'nm')
        df[['torque', 'max_torque_rpm']] = df['torque'].apply(lambda x: pd.Series(clean_torque(x)))
    if light_mode:
        return df

    # Генерация дополнительных признаков
    with open('train_medians.pkl', 'rb') as f:
        train_medians = pickle.load(f)
    for col in df.columns[df.isna().any()].tolist() + ['mileage']:
        model_medians = df.groupby('name')[col].median()
        df[col] = df.apply(lambda row: model_medians[row['name']] if (pd.isna(row[col])) or (row[col] == 0) else row[col], axis=1)
        df[col] = df[col].apply(lambda x: np.nan if x==0 else x) 
        df[col] = df[col].fillna(train_medians[col])

    df[['engine', 'seats']] = df[['engine', 'seats']].astype(int)
    df['brand'] = df['name'].str.split().str[0]
    
    with open('cars.json', 'r') as f:
        cars = json.load(f)
    df['car_type'] = df['name'].map(cars)

    premium_brand = ['Audi', 'Lexus', 'Jaguar', 
                        'Land', 'Mercedes-Benz', 'BMW', 
                        'Volvo', 'Ford', 'Jeep', 'Toyota']
    df['premium'] = df['brand'].map(lambda x: 1 if x in premium_brand else 0)

    df['age'] = max(df['year']) + 1 - df['year']
    df['usage_ratio'] = df['km_driven'] / df['age']

    poly_features = ['age', 'torque', 'max_power', 'mileage', 'km_driven']
    poly = PolynomialFeatures(degree=2, include_bias=False)
    df = pd.concat([
        df.drop(columns=poly_features),
        pd.DataFrame(poly.fit_transform(df[poly_features]), columns=poly.get_feature_names_out())], axis=1)
    return df


def eda_tabs_df(df: pd.DataFrame, train):
    """
    Табы для EDA

    df: Подгружаемый датасет, тип - pd.DataFrame
    train: Флаг переключатель для заголовка, тип - Bool
    """
    if train:
        st.subheader("Анализ обучающей выборки.")
    else :
        st.subheader("Анализ тестовой выборки.")
        
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Обзор", "Числовые признаки", "Категориальные признаки", "Пропуски", "Типы данных"])
        
    # Первый таб - базовая информация о датафрейме
    with tab1:
        st.subheader("Основная информация")
        st.write(f"**Размер:** {df.shape[0]} строк × {df.shape[1]} столбцов")
        st.write(f"**Индексация:** от {df.index[0]} до {df.index[-1]}")
        st.subheader("Первые 10 строк датасета")
        st.dataframe(df.head(10))

    # Второй таб - исследование числовых и категориальных признаков 
    with tab2:
        # Числовые признаки
        st.subheader("Статистики числовых признаков")
        numeric_cols = df.select_dtypes(exclude='object').columns.tolist()
        if len(numeric_cols) > 0:
            st.dataframe(df[numeric_cols].describe())
            st.subheader("Гистограммы числовых признаков")
            if "selling_price" in numeric_cols:
                fig_hist = px.histogram(
                            df, x="selling_price", nbins=30, title=f"Распределение целевого признака",
                        )
                fig_hist.update_layout(
                    showlegend=False,
                    margin=dict(l=10, r=10, t=40, b=10),
                    height=300
                )
                st.plotly_chart(fig_hist)
                numeric_cols.remove("selling_price")
            col1, col2 = st.columns(2)
            for col in numeric_cols:
                # Гистограмма распределения
                with col1:
                    fig_hist = px.histogram(
                        df, x=col, nbins=30, title=f"Распределение {col}",
                    )
                    fig_hist.update_layout(
                        showlegend=False,
                        margin=dict(l=10, r=10, t=40, b=10),
                        height=300
                    )
                    st.plotly_chart(fig_hist)
                # Scatter с ценой
                with col2:
                    fig_scatter = px.scatter(
                        df, x=col, y="selling_price",
                        title=f"{col} vs selling_price",
                        trendline="ols",
                        trendline_color_override="red"
                    )
                    fig_scatter.update_layout(
                        margin=dict(l=10, r=10, t=40, b=10),
                        height=300
                    )
                    st.plotly_chart(fig_scatter)
        else:
            st.info("Нет числовых столбцов")

    with tab3:
        # Категориальные признаки
        st.subheader("Статистики категориальных признаков")
        object_cols = df.select_dtypes(include='object').columns.to_list()
        if len(object_cols) > 0:
            st.dataframe(df[object_cols].describe())
            st.subheader("Гистограммы категориальных признаков")
            if "name" in object_cols:
                object_cols.remove("name")

            col1, col2 = st.columns(2)
            for i, col in enumerate(object_cols):
                current_col = col1 if i % 2 == 0 else col2
                with current_col:
                    fig = px.violin(
                        df,
                        x=col,
                        y="selling_price",
                        color=col,
                        box=True,
                        points="outliers",
                        title=f"Цена по {col}"
                    )

                    fig.update_layout(
                        height=480,
                        margin=dict(t=60, b=20, l=10, r=10),
                        showlegend=False,
                        violingap=0.1,
                        violinmode='overlay'
                    )
                    st.plotly_chart(fig)
        else:
            st.info("Нет категориальных столбцов")

    with tab4:
        st.subheader("Пропущенные значения")
        missing = df.isnull().sum()
        missing_percent = (missing / len(df)) * 100
        missing_df = pd.DataFrame({
            'Колонка': missing.index,
            'Пропусков': missing.values,
            '% пропусков': missing_percent.values.round(2)
        })
        missing_df = missing_df[missing_df['Пропусков'] > 0]
        
        if len(missing_df) > 0:
            st.dataframe(missing_df)
            st.subheader("Визуализация пропусков")

            filtered_missing = missing_percent[missing_percent > 0].sort_values()
            fig = px.bar(
                x=filtered_missing.values,
                y=filtered_missing.index,
                orientation='h',
                title='Процент пропусков по столбцам',
                labels={'x': '% пропусков', 'y': 'Столбцы'}
            )

            st.plotly_chart(fig)
        else:
            st.success("Пропущенных значений нет")

    with tab5:
        st.subheader("Типы данных")
        dtype_df = pd.DataFrame({
            'Колонка': df.columns,
            'Тип данных': df.dtypes.values,
            'Уникальных значений': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(dtype_df)

    st.divider()


def load_model(path):
    """
    Подгружаем модель для пресказания
    
    path: Путь до pickle файла, тип - строка
    """
    with open(path, 'rb') as f:
        model = pickle.load(f)
    if model is not None:
        st.success("Модель успешно загружена и готова к работе!")
    return model

st.title("Прогноз стоимости продажи автомобиля на предобученной модели линейной регрессии.")
st.set_page_config(
    page_title="Дашборд прогноза стоимости автомобилей",
    layout="wide"
)

# Главные табы для датасетов и ручного ввода.
tab1_global, tab2_global = st.tabs(["Датасеты", "Ручной ввод"])
# Для первой табы, подгружаем датасеты и визуализируем. После загрузки теста - получаем предсказание.
with tab1_global:
    col1, col2 = st.columns(2)
    with col1:
        file_train = st.file_uploader("Загрузите CSV файл c обучающим датасетом", type=["csv"])
    with col2:
        file_test = st.file_uploader("Загрузите CSV файл c тестовым датасетом", type=["csv"])
    # Легенда, если загружен хоть один из двух
    if file_train is not None or file_test is not None:
        st.divider()
        st.subheader("Параметры датасета:")
        col1, col2 = st.columns(2)
        with col1:
            st.write("`name` - название автомобиля,")
            st.write("`year` - год выпуска автомобиля,")
            st.write("`selling_price` - цена продажи автомобиля,")
            st.write("`km_driven` - пробег автомобиля,")
            st.write("`fuel` - тип топлива автомобиля,")
            st.write("`seller_type` - кто продает автомобиль,")
            st.write("`transmission` - тип коробки передач,")
        with col2:
            st.write("`owner` - количество владельцев автомобиля,")
            st.write("`mileage` - расход топлива,")
            st.write("`engine` - объем двигателя,")
            st.write("`max_power` - максимальная мощность двигателя,")
            st.write("`torque` - вращательный момент,")
            st.write("`seats` - количество мест.")

    # Грузим датасеты, даем light_mode обработку, показываем табы
    if file_train is not None:
        df_train = pd.read_csv(file_train)
        df_train = preprocess_df(df_train, light_mode=True)
        eda_tabs_df(df_train, train=True)
    if file_test is not None:
        df_test = pd.read_csv(file_test)
        df_test = preprocess_df(df_test, light_mode=True)
        eda_tabs_df(df_test, train=False)

        # Для теста генерим доп. фичи
        st.warning('Часть требуемых признаков отсутствует. Для повышения точности прогноза сгенерированы дополнительные признаки.')
        df_test = preprocess_df(df_test)
        st.success("Датасет преобразован и готов к работе!")

        # Отдель блок с результатом прогноза, дает R2, RMSE, веса признаков и анализ остатков
        st.divider()
        st.header("Прогноз стоимости продажи автомобиля по загруженным данным.")
        inference_model = load_model("model_pipeline.pkl")
        st.subheader("Результаты прогнозирования с использованием загруженных данных.")
        if st.button("Предсказать", key="loaded_data_predict_button"):
            true = df_test['selling_price'].values
            preds = inference_model.predict(df_test)
            st.write(f"**Метрика R2:** {r2_score(true, preds):.3f}")
            st.write(f"**Метрика RMSE:** {root_mean_squared_error(true, preds):.3f}")

            # Важность признаков из веса инфересна  
            model_weights = inference_model.named_steps['ridge'].coef_
            weights_df = pd.DataFrame({
                'Feature': df_test.columns,
                'Weight': model_weights
            }).sort_values('Weight', ascending=True)
            fig_weights = px.bar(
                weights_df,
                x='Weight',
                y='Feature',
                orientation='h',
                title='Важность признаков',
                color='Weight',
                color_continuous_scale='RdBu'
            )
            st.plotly_chart(fig_weights)
            
            # Остатки из разности с реальными значениями теста
            residuals = true - preds
            fig_hist = px.histogram(
                x=residuals,
                nbins=30,
                title='Распределение остатков',
                labels={'x': 'Остатки', 'count': 'Частота'}
            )
            fig_scatter = px.scatter(
                x=preds,
                y=residuals,
                title='Остатки vs Предсказания',
                labels={'x': 'Предсказанные значения', 'y': 'Остатки'}
            )
            fig_scatter.add_hline(y=0, line_dash="dash", line_color="red")
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_hist)
            with col2:
                st.plotly_chart(fig_scatter)

# Вторая глобальная таба, ввод параметров и прогноз по ним
with tab2_global:
    # Грузим модель
    st.subheader("Прогноз по параметрам, введенным вручную.")
    inference_model = load_model("model_pipeline.pkl")
    st.write("Введите параметры автомобиля для прогнозирования его стоимости")

    # Читаем ввод в двух колонках
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("**Название модели**", placeholder="Например: Hyundai Creta")
        year = st.slider("**Год выпуска**", min_value=1950, max_value=2020, value=2020, step=1)
        km_driven = st.number_input("**Пробег (км)**", min_value=0, value=50000, step=1000)
        fuel = st.selectbox("**Тип топлива**", ["Diesel", "Petrol", "CNG", "LPG"])
        seller_type = st.selectbox("**Тип продавца**", ["Dealer", "Individual", "Trustmark Dealer"])
        transmission = st.selectbox("**Трансмиссия**", ["Manual", "Automatic"])
    with col2:
        owner = st.selectbox(
            "**Количество владельцев**", 
            ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"]
        )
        mileage = st.number_input("**Расход топлива (км/л)**", min_value=0.0, value=20.0, step=0.1)
        engine = st.number_input("**Объем двигателя (см³)**", min_value=0, value=1500, step=1)
        max_power = st.number_input("**Мощность (л.с.)**", min_value=0.0, value=100.0, step=1.0)
        torque = st.text_input("**Крутящий момент**", placeholder="Например: 190Nm@ 2000rpm")
        seats = st.slider("**Количество мест**", min_value=2, max_value=30, value=5, step=1)

    # После нажатия на кнопку, генерим фичи и прогнозируем стоимость
    if st.button("Предсказать", key="manual_predict_button"):
        new_df = pd.DataFrame({
            'name': [name],
            'year': [year],
            'km_driven': [km_driven],
            'fuel': [fuel],
            'seller_type': [seller_type],
            'transmission': [transmission],
            'owner': [owner],
            'mileage': [str(mileage)],
            'engine': [str(engine)],
            'max_power': [str(max_power)],
            'torque': [torque],
            'seats': [seats]
        })
        new_df = preprocess_df(new_df)
        pred = inference_model.predict(new_df)[0]
        st.write(f"**Предсказанная стоимость:** {pred:.0f}")