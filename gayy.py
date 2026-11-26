import re
import time
import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz

start_time = time.time()

# ==========================================
# 1. ФИЛЬТРАЦИЯ ПО 3 ГОРОДАМ И НОРМАЛИЗАЦИЯ
# ==========================================
TARGET_CITIES = {'мирный', 'удачный', 'айхал'}

def detect_city(text):
    if pd.isna(text):
        return None
    s = str(text).lower()
    # Ищем точное вхождение названия города (чтобы "мирнинский" не сработал как "мирный")
    if re.search(r'\bмирный\b', s):
        return 'мирный'
    if re.search(r'\bудачный\b', s):
        return 'удачный'
    if re.search(r'\bайхал\b', s):
        return 'айхал'
    return None

def normalize_address(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # Убираем пометки в скобках, например (ЗАКРЛС)
    text = re.sub(r'\([^)]*\)', '', text)
    # Убираем сокращения и административные единицы
    text = re.sub(r'\b(г|ул|д|кв|респ|у|тер|пл|пер|пр-кт|город|село|с|п|пгт|саха|якутия|мирнинский)\b\.?', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    # Убираем ведущие нули в номерах квартир/домов ("014" -> "14")
    text = re.sub(r'\b0+(\d+)\b', r'\1', text)
    return re.sub(r'\s+', ' ', text).strip()

# Оставляем только Мирный, Удачный, Айхал во всех трех таблицах
df_ik = df_ik.copy()
df_utv = df_utv.copy()
df_fias = df_fias.copy()

df_ik['city'] = df_ik['Наименование'].apply(detect_city)
df_utv['city'] = df_utv['Наименование'].apply(detect_city)
df_fias['city'] = df_fias['Полный адрес'].apply(detect_city)

df_ik = df_ik[df_ik['city'].isin(TARGET_CITIES)].reset_index(drop=True)
df_utv = df_utv[df_utv['city'].isin(TARGET_CITIES)].reset_index(drop=True)
df_fias = df_fias[df_fias['city'].isin(TARGET_CITIES)].reset_index(drop=True)

df_ik['clean'] = df_ik['Наименование'].apply(normalize_address)
df_utv['clean'] = df_utv['Наименование'].apply(normalize_address)
df_fias['clean'] = df_fias['Полный адрес'].apply(normalize_address)

# ==========================================
# 2. БЫСТРОЕ НЕЧЕТКОЕ СРАВНЕНИЕ ПО ГОРОДАМ
# ==========================================
THRESHOLD = 85

def fast_fuzzy_match(df_left, df_right, right_col_orig, scorer, threshold=85):
    n = len(df_left)
    best_idx = np.full(n, -1, dtype=int)
    best_scores = np.zeros(n, dtype=float)
    best_vals = [None] * n

    right_orig_arr = df_right[right_col_orig].values

    for city in TARGET_CITIES:
        left_group = df_left[df_left['city'] == city]
        right_group = df_right[df_right['city'] == city]
        if left_group.empty or right_group.empty:
            continue

        l_indices = left_group.index.values
        r_indices = right_group.index.values

        mat = process.cdist(
            left_group['clean'].tolist(),
            right_group['clean'].tolist(),
            scorer=scorer,
            workers=-1
        )

        max_pos = mat.argmax(axis=1)
        max_val = mat.max(axis=1)

        for i, (pos, score) in enumerate(zip(max_pos, max_val)):
            orig_l_idx = l_indices[i]
            best_scores[orig_l_idx] = score
            if score >= threshold:
                orig_r_idx = r_indices[pos]
                best_idx[orig_l_idx] = orig_r_idx
                best_vals[orig_l_idx] = right_orig_arr[orig_r_idx]

    return best_idx, best_vals, best_scores

# --- 1) Сверяем ИК с УТВ ---
idx_utv, addr_utv, score_utv = fast_fuzzy_match(df_ik, df_utv, 'Наименование', fuzz.token_sort_ratio, THRESHOLD)
df_ik['_utv_idx'] = [i if i != -1 else None for i in idx_utv]
df_ik['Адрес_в_УТВ'] = addr_utv
df_ik['Схожесть_УТВ_%'] = score_utv
df_ik['УТВ'] = ['Да' if s >= THRESHOLD else 'Нет' for s in score_utv]

# --- 2) Сверяем ИК с ФИАС ---
idx_fias_ik, addr_fias_ik, score_fias_ik = fast_fuzzy_match(df_ik, df_fias, 'Полный адрес', fuzz.token_set_ratio, THRESHOLD)
df_ik['_fias_idx'] = [i if i != -1 else None for i in idx_fias_ik]
df_ik['Адрес_в_ФИАС'] = addr_fias_ik
df_ik['Схожесть_ФИАС_%'] = score_fias_ik
df_ik['ФИАС'] = ['Да' if s >= THRESHOLD else 'Нет' for s in score_fias_ik]
df_ik['ИК'] = 'Да'

# --- 3) Сверяем УТВ с ФИАС ---
idx_fias_utv, addr_fias_utv, score_fias_utv = fast_fuzzy_match(df_utv, df_fias, 'Полный адрес', fuzz.token_set_ratio, THRESHOLD)
df_utv['_fias_idx'] = [i if i != -1 else None for i in idx_fias_utv]
df_utv['Адрес_в_ФИАС'] = addr_fias_utv
df_utv['Схожесть_ФИАС_%'] = score_fias_utv
df_utv['ФИАС'] = ['Да' if s >= THRESHOLD else 'Нет' for s in score_fias_utv]

def status_ik(row):
    if row['УТВ'] == 'Да' and row['ФИАС'] == 'Да':
        return 'Совпало везде (ИК + УТВ + ФИАС)'
    elif row['УТВ'] == 'Да':
        return 'Есть в ИК и УТВ, но НЕТ в ФИАС'
    elif row['ФИАС'] == 'Да':
        return 'Есть в ИК и ФИАС, но НЕТ в УТВ'
    else:
        return 'Есть только в ИК (НЕТ в УТВ и ФИАС)'

df_ik['Что_совпало'] = df_ik.apply(status_ik, axis=1)

# ==========================================
# 3. ВЫДЕЛЕНИЕ РАСХОЖДЕНИЙ И ОСТАТКОВ
# ==========================================
# 1) Что НЕ сходится между ИК и УТВ
ik_not_in_utv = df_ik[df_ik['УТВ'] == 'Нет'].copy()

matched_utv_indices = set(df_ik['_utv_idx'].dropna().astype(int))
utv_not_in_ik = df_utv[~df_utv.index.isin(matched_utv_indices)].copy()
utv_not_in_ik['ИК'] = 'Нет'
utv_not_in_ik['УТВ'] = 'Да'
utv_not_in_ik['Что_совпало'] = utv_not_in_ik['ФИАС'].apply(
    lambda x: 'Есть в УТВ и ФИАС, но НЕТ в ИК' if x == 'Да' else 'Есть только в УТВ (НЕТ в ИК и ФИАС)'
)

df_diff_ik_utv = pd.concat([ik_not_in_utv, utv_not_in_ik], ignore_index=True)

# 2) Адреса, которых НЕТ ни в УТВ, ни в ИК, но ЕСТЬ в ФИАС
fias_in_ik = set(df_ik['_fias_idx'].dropna().astype(int))
fias_in_utv = set(df_utv['_fias_idx'].dropna().astype(int))
all_fias_found = fias_in_ik | fias_in_utv

df_only_fias = df_fias[~df_fias.index.isin(all_fias_found)].copy()
df_only_fias['ФИАС'] = 'Да'
df_only_fias['УТВ'] = 'Нет'
df_only_fias['ИК'] = 'Нет'
df_only_fias['Что_совпало'] = 'Есть только в ФИАС (НЕТ ни в ИК, ни в УТВ)'

# 3) Полный объединенный реестр
df_full = pd.concat([df_ik, utv_not_in_ik, df_only_fias], ignore_index=True)

# Форматирование колонок (ФИАС, УТВ, ИК — в самом начале)
df_full['Населенный_пункт'] = df_full['city'].str.capitalize()
df_ik['Населенный_пункт'] = df_ik['city'].str.capitalize()
df_diff_ik_utv['Населенный_пункт'] = df_diff_ik_utv['city'].str.capitalize()
df_only_fias['Населенный_пункт'] = df_only_fias['city'].str.capitalize()

service_cols = ['clean', 'city', '_utv_idx', '_fias_idx']
first_cols = ['ФИАС', 'УТВ', 'ИК', 'Населенный_пункт', 'Что_совпало', 'Код', 'Наименование', 'Адрес_в_УТВ', 'Схожесть_УТВ_%', 'Адрес_в_ФИАС', 'Схожесть_ФИАС_%']

def format_df(df):
    d = df.drop(columns=[c for c in service_cols if c in df.columns])
    head = [c for c in first_cols if c in d.columns]
    tail = [c for c in d.columns if c not in head]
    return d[head + tail]

df_full_clean = format_df(df_full)
df_ik_clean = format_df(df_ik)
df_diff_clean = format_df(df_diff_ik_utv)
df_only_fias_clean = format_df(df_only_fias)

# ==========================================
# 4. ПОДРОБНАЯ ПРОЦЕНТНАЯ АНАЛИТИКА (ОБЩАЯ + ПО ГОРОДАМ)
# ==========================================
def build_stats(sub_ik, sub_utv, sub_fias, sub_utv_not_ik, sub_only_fias, label):
    n_ik, n_utv, n_fias = len(sub_ik), len(sub_utv), len(sub_fias)
    c_all_3 = (sub_ik['Что_совпало'] == 'Совпало везде (ИК + УТВ + ФИАС)').sum()
    c_ik_in_utv = (sub_ik['УТВ'] == 'Да').sum()
    c_ik_not_utv = (sub_ik['УТВ'] == 'Нет').sum()
    c_utv_not_ik = len(sub_utv_not_ik)
    c_ik_in_fias = (sub_ik['ФИАС'] == 'Да').sum()
    c_utv_in_fias = (sub_utv['ФИАС'] == 'Да').sum()
    c_only_fias = len(sub_only_fias)
    c_fias_in_ours = n_fias - c_only_fias

    pct = lambda a, b: f'{a/b:.2%}' if b > 0 else '0.00%'

    return [
        {'Территория': label, 'Показатель': 'Всего записей в ИК', 'Кол-во': n_ik, '% от ИК': '100.00%', '% от УТВ': '-', '% от ФИАС': '-'},
        {'Территория': label, 'Показатель': 'Всего записей в УТВ', 'Кол-во': n_utv, '% от ИК': '-', '% от УТВ': '100.00%', '% от ФИАС': '-'},
        {'Территория': label, 'Показатель': 'Всего записей в ФИАС', 'Кол-во': n_fias, '% от ИК': '-', '% от УТВ': '-', '% от ФИАС': '100.00%'},
        {'Территория': label, 'Показатель': 'Совпало между ИК и УТВ', 'Кол-во': c_ik_in_utv, '% от ИК': pct(c_ik_in_utv, n_ik), '% от УТВ': pct(c_ik_in_utv, n_utv), '% от ФИАС': '-'},
        {'Территория': label, 'Показатель': 'Не сошлось: есть в ИК, но НЕТ в УТВ', 'Кол-во': c_ik_not_utv, '% от ИК': pct(c_ik_not_utv, n_ik), '% от УТВ': '-', '% от ФИАС': '-'},
        {'Территория': label, 'Показатель': 'Не сошлось: есть в УТВ, но НЕТ в ИК', 'Кол-во': c_utv_not_ik, '% от ИК': '-', '% от УТВ': pct(c_utv_not_ik, n_utv), '% от ФИАС': '-'},
        {'Территория': label, 'Показатель': 'Совпало везде (ИК + УТВ + ФИАС)', 'Кол-во': c_all_3, '% от ИК': pct(c_all_3, n_ik), '% от УТВ': pct(c_all_3, n_utv), '% от ФИАС': pct(c_all_3, n_fias)},
        {'Территория': label, 'Показатель': 'Адресов ИК найдено в ФИАС', 'Кол-во': c_ik_in_fias, '% от ИК': pct(c_ik_in_fias, n_ik), '% от УТВ': '-', '% от ФИАС': pct(c_ik_in_fias, n_fias)},
        {'Территория': label, 'Показатель': 'Адресов УТВ найдено в ФИАС', 'Кол-во': c_utv_in_fias, '% от ИК': '-', '% от УТВ': pct(c_utv_in_fias, n_utv), '% от ФИАС': pct(c_utv_in_fias, n_fias)},
        {'Территория': label, 'Показатель': 'Адресов ФИАС, которые есть у нас (в ИК или УТВ)', 'Кол-во': c_fias_in_ours, '% от ИК': '-', '% от УТВ': '-', '% от ФИАС': pct(c_fias_in_ours, n_fias)},
        {'Территория': label, 'Показатель': 'Адресов ФИАС, КОТОРЫХ НЕТ НИ В ИК, НИ В УТВ', 'Кол-во': c_only_fias, '% от ИК': '-', '% от УТВ': '-', '% от ФИАС': pct(c_only_fias, n_fias)},
    ]

all_stats = build_stats(df_ik, df_utv, df_fias, utv_not_in_ik, df_only_fias, 'ВСЕГО (Мирный + Удачный + Айхал)')
for c in ['мирный', 'удачный', 'айхал']:
    all_stats.extend(build_stats(
        df_ik[df_ik['city'] == c],
        df_utv[df_utv['city'] == c],
        df_fias[df_fias['city'] == c],
        utv_not_in_ik[utv_not_in_ik['city'] == c],
        df_only_fias[df_only_fias['city'] == c],
        c.capitalize()
    ))

df_summary = pd.DataFrame(all_stats)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

print("\n" + "=" * 110)
print("ПОДРОБНАЯ АНАЛИТИКА СВЕРКИ (МИРНЫЙ, УДАЧНЫЙ, АЙХАЛ)")
print("=" * 110)
print(df_summary[df_summary['Территория'] == 'ВСЕГО (Мирный + Удачный + Айхал)'].to_string(index=False))
print("=" * 110 + "\n")

# ==========================================
# 5. СОХРАНЕНИЕ В EXCEL ПО ЛИСТАМ
# ==========================================
output_file = 'Сверка_Мирный_Удачный_Айхал.xlsx'

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    df_summary.to_excel(writer, sheet_name='1_Аналитика_и_города', index=False)
    df_full_clean.to_excel(writer, sheet_name='2_Полный_объединенный', index=False)
    df_ik_clean.to_excel(writer, sheet_name='3_Проверка_ИК', index=False)
    df_diff_clean.to_excel(writer, sheet_name='4_Не_сходится_ИК_и_УТВ', index=False)
    df_only_fias_clean.to_excel(writer, sheet_name='5_Есть_в_ФИАС_нет_у_нас', index=False)

    for sheet_name in writer.sheets:
        ws = writer.sheets[sheet_name]
        for col in ws.columns:
            max_len = max((len(str(cell.value or '')) for cell in col[:100]), default=10)
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 55)

print(f"Готово за {time.time() - start_time:.1f} сек! Файл сохранен: {output_file}")
