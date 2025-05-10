# Импорт нужных библиотек
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

# Кэшируем функцию загрузки данных (чтобы не загружались заново каждый раз)
@st.cache_data
def load_data():
    return pd.read_csv("data.csv", sep=";")

# Загружаем данные
df = load_data()

# Заголовок веб-приложения
st.title("Прогноз потребления электроэнергии")

# Выбор региона и отрасли пользователем
regions = df['регион'].unique().tolist()
columns = [col for col in df.columns if col not in ['регион', 'год']]
selected_region = st.selectbox("Выберите регион", regions)
selected_column = st.selectbox("Выберите отрасль", columns)
forecast_horizon = st.slider("Горизонт прогноза (лет)", 1, 10, 6)

# Фильтруем данные под выбранный регион
df_filtered = df[df['регион'] == selected_region].sort_values('год')
values = df_filtered[selected_column].astype(float).values
years = df_filtered['год'].values

# Создаём функции-признаки (лаги, разности, кризисы)
def create_features(data, years, n_lags=3):
    rows = []
    for i in range(n_lags, len(data)):
        row = {
            f'lag_{l}': data[i - l] for l in range(1, n_lags + 1)
        }
        row['diff_1_2'] = row['lag_1'] - row['lag_2']
        row['diff_2_3'] = row['lag_2'] - row['lag_3']
        row['year'] = years[i]
        row['year_sq'] = years[i] ** 2
        row['crisis_2009'] = int(years[i] == 2009)
        row['covid_2020'] = int(years[i] == 2020)
        row['svo_2022'] = int(years[i] >= 2022)
        rows.append(row)
    return pd.DataFrame(rows), data[3:], years[3:]

X, y, valid_years = create_features(values, years)

# Объявляем модели
baseline = LinearRegression()
rf = RandomForestRegressor(n_estimators=100, random_state=42)
gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)
ridge = RidgeCV()

# Кросс-валидация с MSE
tscv = TimeSeriesSplit(n_splits=5)
baseline_mse = -np.mean(cross_val_score(baseline, X, y, cv=tscv, scoring='neg_mean_squared_error'))
rf_mse = -np.mean(cross_val_score(rf, X, y, cv=tscv, scoring='neg_mean_squared_error'))
gbr_mse = -np.mean(cross_val_score(gbr, X, y, cv=tscv, scoring='neg_mean_squared_error'))

# Объединяем модели в стеккинг
stack = StackingRegressor(
    estimators=[('rf', rf), ('gbr', gbr)],
    final_estimator=ridge,
    passthrough=True,
    n_jobs=-1
)
stack.fit(X, y)

# Прогноз на будущее
future_preds = []
last_window = list(values[-3:])
future_years = np.arange(years[-1] + 1, years[-1] + 1 + forecast_horizon)

for year in future_years:
    row = {
        'lag_1': last_window[-1],
        'lag_2': last_window[-2],
        'lag_3': last_window[-3],
        'diff_1_2': last_window[-1] - last_window[-2],
        'diff_2_3': last_window[-2] - last_window[-3],
        'year': year,
        'year_sq': year ** 2,
        'crisis_2009': 0,
        'covid_2020': 0,
        'svo_2022': int(year >= 2022)
    }
    pred = stack.predict(pd.DataFrame([row]))[0]
    future_preds.append(pred)
    last_window.append(pred)
    last_window.pop(0)

# Метрики качества прогноза
X_eval = X[-6:]
y_eval = y[-6:]
y_pred_eval = stack.predict(X_eval)

mae = mean_absolute_error(y_eval, y_pred_eval)
rmse = np.sqrt(mean_squared_error(y_eval, y_pred_eval))
mape = np.mean(np.abs((y_eval - y_pred_eval) / y_eval)) * 100
mse = mean_squared_error(y_eval, y_pred_eval)

# Выводим метрики
st.subheader("Сравнение моделей (MSE):")
st.write(f"LinearRegression MSE: {baseline_mse:.2f}")
st.write(f"RandomForest MSE: {rf_mse:.2f}")
st.write(f"GradientBoosting MSE: {gbr_mse:.2f}")

st.subheader("Качество stacking-модели:")
st.write(f"**MAE:** {mae:.2f}")
st.write(f"**MSE:** {mse:.2f}")
st.write(f"**RMSE:** {rmse:.2f}")
st.write(f"**MAPE:** {mape:.2f}%")

# График прогноза
st.subheader("График прогноза:")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(years, values, label="Фактические", marker='o')
ax.plot(future_years, future_preds, label="Прогноз", marker='o')
ax.axvline(years[-1] + 0.5, linestyle='--', color='gray', label='Начало прогноза')
ax.set_xlabel("Год")
ax.set_ylabel("Значение")
ax.set_title(f"{selected_region} — {selected_column}")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Важность признаков (feature importance)
st.subheader("Feature Importance (RandomForest):")
rf.fit(X, y)
importances = rf.feature_importances_
imp_df = pd.DataFrame({"Feature": X.columns, "Importance": importances}).sort_values("Importance", ascending=False)
st.dataframe(imp_df)
fig_imp, ax_imp = plt.subplots()
ax_imp.barh(imp_df['Feature'], imp_df['Importance'])
ax_imp.invert_yaxis()
ax_imp.set_title("Важность признаков")
st.pyplot(fig_imp)
