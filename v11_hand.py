#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Версия 2.1.0
Прогнозирование риска опоздания с техническим осмотром самоходных машин в июне 2026.
Используется CatBoost для классификации на исторических данных.
В итоговый файл добавлен столбец Kod_insp (код инспекции владельца).
"""

import pandas as pd #обработка и анализ структурированных данных
import numpy as np #инструменты для работы с многомерными массивами, матрицами
from datetime import datetime #функции для  работы с датами, временем
from catboost import CatBoostClassifier #модель градиентного бустинга
from sklearn.metrics import roc_auc_score, log_loss #оценка качества классификации и метрика качества
import warnings
warnings.filterwarnings('ignore')

print("Запуск скрипта версии 2.1.0")

# ---------------------------- ШАГ 1: ЗАГРУЗКА ДАННЫХ ----------------------------
print("1. Загрузка данных из Excel-файлов...")

# Загружаем таблицу актуальных машин
tractors = pd.read_excel('tractors.xlsx', dtype={'Owner': str, 'GRZ': str})
print(f"   - tractors.xlsx: {tractors.shape[0]} строк, столбцы: {list(tractors.columns)}")

# Загружаем таблицу техосмотров
to = pd.read_excel('to.xlsx', dtype={'Owner': str, 'GRZ': str, 'Kod_insp': str})
print(f"   - to.xlsx: {to.shape[0]} строк, столбцы: {list(to.columns)}")

# Приводим текстовые поля к единому формату (убираем пробелы, делаем str)
for df in [tractors, to]:
    if 'Owner' in df.columns:
        df['Owner'] = df['Owner'].astype(str).str.strip()
    if 'GRZ' in df.columns:
        df['GRZ'] = df['GRZ'].astype(str).str.strip()
    if 'Kod_insp' in df.columns:
        df['Kod_insp'] = df['Kod_insp'].astype(str).str.strip()

# Преобразуем даты
to['Date_TO'] = pd.to_datetime(to['Date_TO'], errors='coerce')
to['Date_TO_next'] = pd.to_datetime(to['Date_TO_next'], errors='coerce')

# ---------------------------- ШАГ 2: ФИЛЬТРАЦИЯ АКТУАЛЬНЫХ МАШИН ----------------------------
print("2. Оставляем только машины, присутствующие в tractors.xlsx (актуальные)...")
active_grz = set(tractors['GRZ'].unique())
to_active = to[to['GRZ'].isin(active_grz)].copy()
print(f"   - После фильтрации в to.xlsx осталось {to_active.shape[0]} записей.")

# ---------------------------- ШАГ 3: РАСЧЁТ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ ДЛЯ ИСТОРИЧЕСКИХ ОСМОТРОВ ----------------------------
print("3. Расчёт целевой переменной (опоздание) для каждого осмотра, начиная со второго...")

# Сортируем по машине и дате осмотра
to_active = to_active.sort_values(['GRZ', 'Date_TO']).reset_index(drop=True)

# Список для хранения записей с вычисленным опозданием
records = []

# Группируем по машине
for grz, group in to_active.groupby('GRZ'):
    if len(group) < 2:
        continue  # для одной записи нет предыдущей, пропускаем
    # Для каждого осмотра, начиная со второго
    for i in range(1, len(group)):
        prev_row = group.iloc[i-1]
        curr_row = group.iloc[i]
        
        # Ожидаемая дата текущего осмотра = Date_TO_next из предыдущего
        expected_date = prev_row['Date_TO_next']
        if pd.isna(expected_date):
            continue  # если нет ожидаемой даты, пропускаем
        
        actual_date = curr_row['Date_TO']
        if pd.isna(actual_date):
            continue
        
        # Целевая переменная: 1 если опоздание, иначе 0
        is_late = int(actual_date > expected_date)
        
        # Сохраняем все данные текущего осмотра и дополнительную информацию
        record = curr_row.to_dict()
        record['expected_date'] = expected_date
        record['is_late'] = is_late
        record['prev_Date_TO'] = prev_row['Date_TO']
        record['prev_Date_TO_next'] = prev_row['Date_TO_next']
        records.append(record)

df_hist = pd.DataFrame(records)
print(f"   - Сформировано {len(df_hist)} исторических событий с известным опозданием.")
if len(df_hist) == 0:
    raise ValueError("Нет данных для обучения. Проверьте, что есть машины с несколькими осмотрами.")

# ---------------------------- ШАГ 4: ФОРМИРОВАНИЕ ПРИЗНАКОВ ДЛЯ МОДЕЛИ ----------------------------
print("4. Создание признаков для модели...")

# Добавляем базовые признаки из данных
df_hist['month_expected'] = df_hist['expected_date'].dt.month
df_hist['dayofweek_expected'] = df_hist['expected_date'].dt.dayofweek
df_hist['day_expected'] = df_hist['expected_date'].dt.day

# Отсортируем все исторические события по дате осмотра (глобально)
df_hist = df_hist.sort_values('Date_TO').reset_index(drop=True)

# Словари для накопления статистики по владельцам и машинам
owner_stats = {}   # ключ: Owner, значение: {'total_inspections': int, 'total_late': int}
machine_stats = {} # ключ: GRZ, значение: {'inspections': int, 'total_late': int, 'total_days_late': float}

# Функции для обновления статистики
def update_owner_stats(owner, is_late):
    if owner not in owner_stats:
        owner_stats[owner] = {'total_inspections': 0, 'total_late': 0}
    owner_stats[owner]['total_inspections'] += 1
    owner_stats[owner]['total_late'] += is_late

def update_machine_stats(grz, is_late, days_late=0):
    if grz not in machine_stats:
        machine_stats[grz] = {'inspections': 0, 'total_late': 0, 'total_days_late': 0.0}
    machine_stats[grz]['inspections'] += 1
    machine_stats[grz]['total_late'] += is_late
    machine_stats[grz]['total_days_late'] += days_late

# Список для хранения строк с признаками
feature_rows = []

# Проходим по строкам в хронологическом порядке (от старых к новым)
for idx, row in df_hist.iterrows():
    grz = row['GRZ']
    owner = row['Owner']
    actual_date = row['Date_TO']
    expected_date = row['expected_date']
    is_late = row['is_late']
    
    # Признаки, основанные на статистике до этого события
    # Статистика по владельцу (все предыдущие осмотры всех его машин)
    if owner in owner_stats:
        owner_feats = {
            'owner_total_inspections': owner_stats[owner]['total_inspections'],
            'owner_total_late': owner_stats[owner]['total_late'],
            'owner_late_rate': owner_stats[owner]['total_late'] / max(1, owner_stats[owner]['total_inspections'])
        }
    else:
        owner_feats = {'owner_total_inspections': 0, 'owner_total_late': 0, 'owner_late_rate': 0.0}
    
    # Статистика по машине (все предыдущие осмотры этой машины)
    if grz in machine_stats:
        machine_feats = {
            'machine_inspections': machine_stats[grz]['inspections'],
            'machine_late_count': machine_stats[grz]['total_late'],
            'machine_late_rate': machine_stats[grz]['total_late'] / max(1, machine_stats[grz]['inspections']),
            'machine_avg_days_late': machine_stats[grz]['total_days_late'] / max(1, machine_stats[grz]['inspections'])
        }
    else:
        machine_feats = {
            'machine_inspections': 0,
            'machine_late_count': 0,
            'machine_late_rate': 0.0,
            'machine_avg_days_late': 0.0
        }
    
    # Количество дней между предыдущим осмотром и ожидаемой датой (характеризует планируемый интервал)
    days_between = (expected_date - row['prev_Date_TO']).days if not pd.isna(row['prev_Date_TO']) else np.nan
    machine_feats['days_since_prev_TO'] = days_between if pd.notna(days_between) else 0
    
    # Категориальные признаки (можно взять из исходных данных)
    cat_features_values = {
        'Type_TO': row.get('Type_TO', 'unknown'),
        'Resilt': row.get('Resilt', 'unknown'),
        'Mark': row.get('Mark', 'unknown'),
        'Group': row.get('Group', 'unknown'),
        'Operation': row.get('Operation', 'unknown')
    }
    
    # Собираем все признаки в одну строку
    feature_row = {
        'month_expected': row['month_expected'],
        'dayofweek_expected': row['dayofweek_expected'],
        'day_expected': row['day_expected'],
        'owner_total_inspections': owner_feats['owner_total_inspections'],
        'owner_total_late': owner_feats['owner_total_late'],
        'owner_late_rate': owner_feats['owner_late_rate'],
        'machine_inspections': machine_feats['machine_inspections'],
        'machine_late_count': machine_feats['machine_late_count'],
        'machine_late_rate': machine_feats['machine_late_rate'],
        'machine_avg_days_late': machine_feats['machine_avg_days_late'],
        'days_since_prev_TO': machine_feats['days_since_prev_TO'],
        'Type_TO': cat_features_values['Type_TO'],
        'Resilt': cat_features_values['Resilt'],
        'Mark': cat_features_values['Mark'],
        'Group': cat_features_values['Group'],
        'Operation': cat_features_values['Operation'],
        'is_late': is_late   # целевая переменная
    }
    feature_rows.append(feature_row)
    
    # После обработки строки обновляем статистики для использования в будущих событиях
    # Вычисляем количество дней опоздания (если было)
    days_late = max(0, (actual_date - expected_date).days) if is_late else 0
    update_machine_stats(grz, is_late, days_late)
    update_owner_stats(owner, is_late)

df_features = pd.DataFrame(feature_rows)
print(f"   - Создано {len(df_features)} строк с признаками.")

# ---------------------------- ШАГ 5: ОБУЧЕНИЕ МОДЕЛИ ----------------------------
print("5. Обучение модели CatBoost...")

# Добавим исходную дату осмотра для временной валидации
df_features['Date_TO'] = df_hist['Date_TO'].values
train_mask = df_features['Date_TO'] < datetime(2026, 1, 1)
valid_mask = df_features['Date_TO'] >= datetime(2026, 1, 1)

X_train = df_features[train_mask].drop(['is_late', 'Date_TO'], axis=1)
y_train = df_features[train_mask]['is_late']
X_valid = df_features[valid_mask].drop(['is_late', 'Date_TO'], axis=1)
y_valid = df_features[valid_mask]['is_late']

print(f"   - Обучающая выборка: {len(X_train)} примеров, валидационная: {len(X_valid)} примеров.")

# Определяем категориальные признаки
cat_features = ['Type_TO', 'Resilt', 'Mark', 'Group', 'Operation']
# Проверяем, что все они есть в X_train
cat_features = [col for col in cat_features if col in X_train.columns]

if len(X_train) == 0:
    raise ValueError("Нет обучающих примеров. Уменьшите порог года валидации или проверьте данные.")

# Создаём модель
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.03,
    depth=6,
    cat_features=cat_features,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    verbose=50
)

# Обучаем
model.fit(
    X_train, y_train,
    eval_set=(X_valid, y_valid) if len(X_valid) > 0 else None,
    early_stopping_rounds=50,
    verbose=50
)

# Оценка качества (если есть валидация)
if len(X_valid) > 0:
    train_pred = model.predict_proba(X_train)[:,1]
    valid_pred = model.predict_proba(X_valid)[:,1]
    print(f"   - ROC AUC на обучении: {roc_auc_score(y_train, train_pred):.4f}")
    print(f"   - ROC AUC на валидации: {roc_auc_score(y_valid, valid_pred):.4f}")
    print(f"   - LogLoss на валидации: {log_loss(y_valid, valid_pred):.4f}")
else:
    print("   - Нет валидационных данных, пропускаем метрики.")

# ---------------------------- ШАГ 6: ПРОГНОЗ НА МАЙ / ИЮНЬ 2026 ----------------------------
print("6. Формирование прогноза для машин, обязанных пройти ТО в июне 2026...")

# Определяем машины, у которых последний известный осмотр имеет Date_TO_next в июне 2026
# Для этого для каждой машины возьмём самую позднюю запись из to_active
last_inspections = to_active.sort_values('Date_TO').groupby('GRZ').last().reset_index()
# Сохраняем нужные столбцы, включая Kod_insp (последний осмотр)
last_inspections = last_inspections[['GRZ', 'Owner', 'Date_TO', 'Date_TO_next', 'Type_TO', 'Resilt', 'Mark', 'Group', 'Operation', 'Kod_insp']]
# Фильтруем: месяц и год ожидаемой даты
last_inspections = last_inspections.dropna(subset=['Date_TO_next'])
mask_june2026 = (last_inspections['Date_TO_next'].dt.year == 2026) & (last_inspections['Date_TO_next'].dt.month == 6)
machines_to_predict = last_inspections[mask_june2026].copy()
print(f"   - Найдено {len(machines_to_predict)} машин, обязанных пройти ТО в июне 2026.")

if len(machines_to_predict) == 0:
    print("   - Нет машин для прогноза. Выход.")
    exit(0)

# Для этих машин нужно построить признаки, аналогичные обучающим, используя всю историю до последнего осмотра
# Пересчитаем статистики по владельцам и машинам на основе всех исторических осмотров (без учёта будущего)
# Пройдём по всем осмотрам в хронологическом порядке, но остановимся перед последним осмотром (чтобы не использовать его результат)
# Нам нужны агрегаты для каждого владельца/машины на момент ДО последнего осмотра.

# Сбросим статистики
owner_stats_pred = {}
machine_stats_pred = {}

# Все осмотры (включая те, что у машин, не попавших в прогноз) нужны для корректной статистики.
# Отсортируем to_active по дате
to_sorted = to_active.sort_values('Date_TO').reset_index(drop=True)

# Словарь для хранения признаков для каждой машины, которую будем предсказывать
pred_features = []

# Для быстрого поиска, является ли машина прогнозируемой
to_predict_grz = set(machines_to_predict['GRZ'])

# Для каждого владельца сохраним его Kod_insp (наиболее часто встречающийся среди всех осмотров владельца)
# Позже добавим в результат.
# Пока просто сохраним словарь owner_kod_insp (последний или наиболее частый)
owner_kod_insp = {}

# Пройдём по всем осмотрам, обновляя статистики, и когда встречаем последний осмотр машины,
# которая входит в machines_to_predict, то в этот момент (перед обновлением статистики этим осмотром)
# мы сформируем признаки для прогноза.
# Нужно заранее знать индексы последних осмотров для каждой машины
last_idx_per_grz = to_sorted.groupby('GRZ').apply(lambda x: x.index[-1]).to_dict()

for idx, row in to_sorted.iterrows():
    grz = row['GRZ']
    owner = row['Owner']
    
    # Сохраняем Kod_insp для владельца (пока просто запомним все, потом выберем моду)
    if owner not in owner_kod_insp:
        owner_kod_insp[owner] = []
    if pd.notna(row.get('Kod_insp')):
        owner_kod_insp[owner].append(row['Kod_insp'])
    
    # Если эта машина относится к прогнозируемым и это её последний осмотр
    if grz in to_predict_grz and idx == last_idx_per_grz.get(grz):
        # Формируем признаки на основе текущих статистик (до этого осмотра)
        # Статистика по владельцу
        if owner in owner_stats_pred:
            owner_feats = {
                'owner_total_inspections': owner_stats_pred[owner]['total_inspections'],
                'owner_total_late': owner_stats_pred[owner]['total_late'],
                'owner_late_rate': owner_stats_pred[owner]['total_late'] / max(1, owner_stats_pred[owner]['total_inspections'])
            }
        else:
            owner_feats = {'owner_total_inspections': 0, 'owner_total_late': 0, 'owner_late_rate': 0.0}
        
        # Статистика по машине (все предыдущие осмотры этой машины)
        if grz in machine_stats_pred:
            machine_feats = {
                'machine_inspections': machine_stats_pred[grz]['inspections'],
                'machine_late_count': machine_stats_pred[grz]['total_late'],
                'machine_late_rate': machine_stats_pred[grz]['total_late'] / max(1, machine_stats_pred[grz]['inspections']),
                'machine_avg_days_late': machine_stats_pred[grz]['total_days_late'] / max(1, machine_stats_pred[grz]['inspections'])
            }
        else:
            machine_feats = {
                'machine_inspections': 0,
                'machine_late_count': 0,
                'machine_late_rate': 0.0,
                'machine_avg_days_late': 0.0
            }
        
        # Доп. признаки: количество дней между предыдущим осмотром и ожидаемой датой.
        # Найдём предыдущий осмотр этой машины (если есть)
        prev_rows = to_sorted[(to_sorted['GRZ'] == grz) & (to_sorted.index < idx)]
        if len(prev_rows) > 0:
            prev_row = prev_rows.iloc[-1]  # последний перед текущим
            days_between = (row['Date_TO_next'] - prev_row['Date_TO']).days if pd.notna(prev_row['Date_TO']) and pd.notna(row['Date_TO_next']) else 0
        else:
            days_between = 0
        
        # Категориальные признаки из текущей строки (последнего осмотра)
        cat_vals = {
            'Type_TO': row.get('Type_TO', 'unknown'),
            'Resilt': row.get('Resilt', 'unknown'),
            'Mark': row.get('Mark', 'unknown'),
            'Group': row.get('Group', 'unknown'),
            'Operation': row.get('Operation', 'unknown')
        }
        
        # Признаки ожидаемой даты
        expected = row['Date_TO_next']
        if pd.isna(expected):
            continue  # пропускаем, если нет ожидаемой даты
        
        feat_row = {
            'month_expected': expected.month,
            'dayofweek_expected': expected.dayofweek,
            'day_expected': expected.day,
            'owner_total_inspections': owner_feats['owner_total_inspections'],
            'owner_total_late': owner_feats['owner_total_late'],
            'owner_late_rate': owner_feats['owner_late_rate'],
            'machine_inspections': machine_feats['machine_inspections'],
            'machine_late_count': machine_feats['machine_late_count'],
            'machine_late_rate': machine_feats['machine_late_rate'],
            'machine_avg_days_late': machine_feats['machine_avg_days_late'],
            'days_since_prev_TO': days_between,
            'Type_TO': cat_vals['Type_TO'],
            'Resilt': cat_vals['Resilt'],
            'Mark': cat_vals['Mark'],
            'Group': cat_vals['Group'],
            'Operation': cat_vals['Operation'],
            'GRZ': grz,
            'Owner': owner,
            'expected_date': expected,
            'Kod_insp_this': row.get('Kod_insp', 'unknown')  # Kod_insp из последнего осмотра (запасной)
        }
        pred_features.append(feat_row)
    
    # После обработки текущего осмотра (независимо от того, прогнозный он или нет) обновляем статистики для будущих
    # Чтобы обновить, нужно вычислить опоздание для этого осмотра относительно предыдущего.
    # Найдём предыдущий осмотр этой машины (по данным до текущего)
    prev_insp_rows = to_sorted[(to_sorted['GRZ'] == grz) & (to_sorted.index < idx)]
    is_late = 0
    days_late = 0
    if len(prev_insp_rows) > 0:
        prev_row = prev_insp_rows.iloc[-1]
        if pd.notna(prev_row['Date_TO_next']) and pd.notna(row['Date_TO']):
            expected_prev = prev_row['Date_TO_next']
            if row['Date_TO'] > expected_prev:
                is_late = 1
                days_late = (row['Date_TO'] - expected_prev).days
    
    # Обновляем статистики
    if grz not in machine_stats_pred:
        machine_stats_pred[grz] = {'inspections': 0, 'total_late': 0, 'total_days_late': 0.0}
    machine_stats_pred[grz]['inspections'] += 1
    machine_stats_pred[grz]['total_late'] += is_late
    machine_stats_pred[grz]['total_days_late'] += days_late
    
    if owner not in owner_stats_pred:
        owner_stats_pred[owner] = {'total_inspections': 0, 'total_late': 0}
    owner_stats_pred[owner]['total_inspections'] += 1
    owner_stats_pred[owner]['total_late'] += is_late

df_pred = pd.DataFrame(pred_features)
print(f"   - Сформировано {len(df_pred)} записей для прогноза (по количеству машин, обязанных в июне).")

if len(df_pred) == 0:
    print("   - Не удалось сформировать признаки для прогноза. Выход.")
    exit(0)

# Определяем для каждого владельца его Kod_insp (наиболее часто встречающийся)
# Если у владельца нет ни одной записи с Kod_insp, ставим 'unknown'
owner_final_kod = {}
for owner, kod_list in owner_kod_insp.items():
    if len(kod_list) == 0:
        owner_final_kod[owner] = 'unknown'
    else:
        # Находим моду (самое частое значение)
        from collections import Counter
        counter = Counter(kod_list)
        most_common = counter.most_common(1)[0][0]
        owner_final_kod[owner] = most_common

# Добавляем Kod_insp в df_pred
df_pred['Kod_insp'] = df_pred['Owner'].map(owner_final_kod)

# Предсказание вероятностей опоздания
X_pred = df_pred.drop(['GRZ', 'Owner', 'expected_date', 'Kod_insp_this', 'Kod_insp'], axis=1)
probs = model.predict_proba(X_pred)[:, 1]
df_pred['prob_late'] = probs

# ---------------------------- ШАГ 7: АГРЕГАЦИЯ ПО ВЛАДЕЛЬЦАМ И ФОРМИРОВАНИЕ ОТЧЁТА ----------------------------
print("7. Агрегация результатов по владельцам...")

# Для каждого владельца, имеющего машины в прогнозе
owners = df_pred['Owner'].unique()

# Также нам нужны данные о машинах, уже прошедших осмотр в мае / июне 2026
to_june2026 = to_active[(to_active['Date_TO'].dt.year == 2026) & (to_active['Date_TO'].dt.month == 6)]
machines_passed_june = to_june2026.groupby('Owner')['GRZ'].nunique().to_dict()

# Общее количество машин у владельца (из tractors)
owner_total_machines = tractors.groupby('Owner')['GRZ'].nunique().to_dict()

# Подсчёт исторических осмотров вовремя/с опозданием по всем машинам владельца
# Используем df_features, где есть is_late для каждого события, и присоединим Owner
owner_hist_stats = df_features.merge(df_hist[['GRZ', 'Owner']], left_index=True, right_index=True)
owner_ontime = owner_hist_stats[owner_hist_stats['is_late']==0].groupby('Owner').size().to_dict()
owner_late = owner_hist_stats[owner_hist_stats['is_late']==1].groupby('Owner').size().to_dict()

result_rows = []
for owner in owners:
    # Машины этого владельца, обязанные в мае
    owner_machines = df_pred[df_pred['Owner'] == owner]
    n_machines_june = len(owner_machines)
    # Риск: максимальная вероятность опоздания среди этих машин
    risk = owner_machines['prob_late'].max() if n_machines_june > 0 else 0.0
    
    # Количество машин, уже прошедших осмотр в мае (у этого владельца)
    n_passed_june = machines_passed_june.get(owner, 0)
    
    # Общее количество машин у владельца
    total_machines = owner_total_machines.get(owner, 0)
    
    # Количество осмотров вовремя и с опозданием
    n_ontime = owner_ontime.get(owner, 0)
    n_late = owner_late.get(owner, 0)
    
    # Kod_insp владельца (берём из словаря)
    kod = owner_final_kod.get(owner, 'unknown')
    
    result_rows.append({
        'Kod_insp': kod,
        'Owner': owner,
        'Общее количество машин у владельца': total_machines,
        'Общее количество машин для ТО в мае 2026': n_machines_june,
        'Количество машин, прошедших осмотр в мае': n_passed_june,
        'Количество ТО вовремя за все время': n_ontime,
        'Количество ТО с опозданием за все время': n_late,
        'Условный риск нарушения (макс. вероятность)': round(risk, 4)
    })

df_result = pd.DataFrame(result_rows)
print(f"   - Сформирован отчёт по {len(df_result)} владельцам.")

# ---------------------------- ШАГ 8: СОХРАНЕНИЕ РЕЗУЛЬТАТА ----------------------------
output_file = 'risk_june2026.xlsx'
df_result.to_excel(output_file, index=False)
print(f"8. Результат сохранён в файл: {output_file}")
print("Работа завершена.")