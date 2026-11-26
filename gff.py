import re
import time
import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz

start_time = time.time()
TARGET_CITIES = {'мирный', 'удачный', 'айхал'}
THRESHOLD = 85

def detect_city(text):
    if pd.isna(text):
        return None
    s = str(text).lower()
    if re.search(r'\bмирный\b', s): return 'мирный'
    if re.search(r'\bудачный\b', s): return 'удачный'
    if re.search(r'\bайхал\b', s): return 'айхал'
    return None

def normalize_address(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # Убираем пометки в скобках, например (ЗАКРЛС)
    text = re.sub(r'\([^)]*\)', '', text)
    # Убираем административные слова, типы объектов и сокращения (в т.ч. уд, улус, тер, онт, мкр, кв-л)
    stop_words = (
        r'г|ул|д|кв|респ|у|уд|улус|тер|онт|снт|пл|пер|пр-кт|проезд|ш|б-р|'
        r'город|село|с|п|пгт|саха|якутия|мирнинский|мр-н|с/п|поселок|посёлок|'
        r'мкр|микрорайон|кв-л|квартал|корп|корпус|стр|строение|лит|литера|дом|квартира'
    )
    text = re.sub(rf'\b({stop_words})\b\.?', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    # Убираем ведущие нули в номерах ("014" -> "14")
    text = re.sub(r'\b0+(\d+)\b', r'\1', text)
    return re.sub(r'\s+', ' ', text).strip()

def make_block_key(row):
    c = row['city']
    # Убираем название города и цифры, берем первые 2 буквы первого текстового токена (улицы/территории)
    street_part = re.sub(rf'\b{c}\b', '', row['clean']).strip()
    words = [w for w in street_part.split() if not w.isdigit()]
    prefix = words[0][:2] if words else 'num'
    return f"{c}_{prefix}"

# ==========================================
# 1. ФИЛЬТРАЦИЯ ПО 3 ГОРОДАМ И ПОИСК ДУБЛЕЙ
# ==========================================
df_ik_raw = df_ik.copy()
df_utv_raw = df_utv.copy()
df_fias_raw = df_fias.copy()

df_ik_raw['city'] = df_ik_raw['Наименование'].apply(detect_city)
df_utv_raw['city'] = df_utv_raw['Наименование'].apply(detect_city)
df_fias_raw['city'] = df_fias_raw['Полный адрес'].apply(detect_city)

df_ik_raw = df_ik_raw[df_ik_raw['city'].isin(TARGET_CITIES)].reset_index(drop=True)
df_utv_raw = df_utv_raw[df_utv_raw['city'].isin(TARGET_CITIES)].reset_index(drop=True)
df_fias_raw = df_fias_raw[df_fias_raw['city'].isin(TARGET_CITIES)].reset_index(drop=True)

df_ik_raw['clean'] = df_ik_raw['Наименование'].apply(normalize_address)
df_utv_raw['clean'] = df_utv_raw['Наименование'].apply(normalize_address)
df_fias_raw['clean'] = df_fias_raw['Полный адрес'].apply(normalize_address)

# Выделяем дубли внутри каждой базы (по нормализованному адресу)
def extract_duplicates(df, addr_col, source_name):
    df = df.copy()
    df['Источник'] = source_name
    df['Точный_дубль_строки'] = df.duplicated(subset=[addr_col], keep='first')
    df['Дубль_после_очистки'] = df.duplicated(subset=['city', 'clean'], keep='first')
    dups = df[df['Дубль_после_очистки']].copy()
    unique_df = df[~df['Дубль_после_очистки']].reset_index(drop=True)
    return unique_df, dups, len(df), df['Точный_дубль_строки'].sum(), len(dups)

df_ik, dups_ik, raw_n_ik, exact_dup_ik, norm_dup_ik = extract_duplicates(df_ik_raw, 'Наименование', 'ИК')
df_utv, dups_utv, raw_n_utv, exact_dup_utv, norm_dup_utv = extract_duplicates(df_utv_raw, 'Наименование', 'УТВ')
df_fias, dups_fias, raw_n_fias, exact_dup_fias, norm_dup_fias = extract_duplicates(df_fias_raw, 'Полный адрес', 'ФИАС')

df_all_dups = pd.concat([
    dups_ik[['Источник', 'city', 'Наименование', 'clean', 'Точный_дубль_строки']],
    dups_utv[['Источник', 'city', 'Наименование', 'clean', 'Точный_дубль_строки']],
    dups_fias[['Источник', 'city', 'Полный адрес', 'clean', 'Точный_дубль_строки']].rename(columns={'Полный адрес': 'Наименование'})
], ignore_index=True)

df_ik['block'] = df_ik.apply(make_block_key, axis=1)
df_utv['block'] = df_utv.apply(make_block_key, axis=1)
df_fias['block'] = df_fias.apply(make_block_key, axis=1)

print(f"Подготовка и очистка дублей завершены за {time.time() - start_time:.1f} сек. Начинаю сверку...")

# ==========================================
# 2. БЫСТРОЕ НЕЧЕТКОЕ СРАВНЕНИЕ (БЕЗ ДУБЛЕЙ)
# ==========================================
def ultra_fast_match(df_left, df_right, right_col_orig, scorer, threshold=85):
    n = len(df_left)
    best_idx = np.full(n, -1, dtype=int)
    best_scores = np.zeros(n, dtype=float)
    best_vals = [None] * n
    right_orig_arr = df_right[right_col_orig].values

    right_groups = {k: v for k, v in df_right.groupby('block')}

    for block_key, left_group in df_left.groupby('block'):
        right_group = right_groups.get(block_key)
        if right_group is None or right_group.empty:
            continue

        l_indices = left_group.index.values
        r_indices = right_group.index.values

        mat = process.cdist(
            left_group['clean'].tolist(),
            right_group['clean'].tolist(),
            scorer=scorer,
            score_cutoff=threshold,
            workers=-1
        )

        max_pos = mat.argmax(axis=1)
        max_val = mat.max(axis=1)

        for i, (pos, score) in enumerate(zip(max_pos, max_val)):
            if score >= threshold:
                orig_l_idx = l_indices[i]
                orig_r_idx = r_indices[pos]
                best_idx[orig_l_idx] = orig_r_idx
                best_scores[orig_l_idx] = score
                best_vals[orig_l_idx] = right_orig_arr[orig_r_idx]

    return best_idx, best_vals, best_scores

# 1) ИК -> УТВ
idx_utv, addr_utv, score_utv = ultra_fast_match(df_ik, df_utv, 'Наименование', fuzz.token_sort_ratio, THRESHOLD)
df_ik['_utv_idx'] = [i if i != -1 else None for i in idx_utv]
df_ik['Адрес_в_УТВ'] = addr_utv
df_ik['Схожесть_УТВ_%'] = score_utv
df_ik['УТВ'] = ['Да' if s >= THRESHOLD else 'Нет' for s in score_utv]

# 2) ИК -> ФИАС
idx_fias_ik, addr_fias_ik, score_fias_ik = ultra_fast_match(df_ik, df_fias, 'Полный адрес', fuzz.token_set_ratio, THRESHOLD)
df_ik['_fias_idx'] = [i if i != -1 else None for i in idx_fias_ik]
df_ik['Адрес_в_ФИАС'] = addr_fias_ik
df_ik['Схожесть_ФИАС_%'] = score_fias_ik
df_ik['ФИАС'] = ['Да' if s >= THRESHOLD else 'Нет' for s in score_fias_ik]
df_ik['ИК'] = 'Да'

# 3) УТВ -> ФИАС
idx_fias_utv, addr_fias_utv, score_fias_utv = ultra_fast_match(df_utv, df_fias, 'Полный адрес', fuzz.token_set_ratio, THRESHOLD)
df_utv['_fias_idx'] = [i if i != -1 else None for i in idx_fias_utv]
df_utv['Адрес_в_ФИАС'] = addr_fias_utv
df_utv['Схожесть_ФИАС_%'] = score_fias_utv
df_utv['ФИАС'] = ['Да' if s >= THRESHOLD else 'Нет' for s in score_fias_utv]

def status_ik(row):
    if row['УТВ'] == 'Да' and row['ФИАС'] == 'Да': return 'Совпало везде (ИК + УТВ + ФИАС)'
    elif row['УТВ'] == 'Да': return 'Есть в ИК и УТВ, но НЕТ в ФИАС'
    elif row['ФИАС'] == 'Да': return 'Есть в ИК и ФИАС, но НЕТ в УТВ'
    else: return 'Есть только в ИК (НЕТ в УТВ и ФИАС)'

df_ik['Что_совпало'] = df_ik.apply(status_ik, axis=1)

# ==========================================
# 3. ВЫДЕЛЕНИЕ РАСХОЖДЕНИЙ И ОСТАТКОВ
# ==========================================
ik_not_in_utv = df_ik[df_ik['УТВ'] == 'Нет'].copy()
matched_utv_indices = set(df_ik['_utv_idx'].dropna().astype(int))
utv_not_in_ik = df_utv[~df_utv.index.isin(matched_utv_indices)].copy()
utv_not_in_ik['ИК'] = 'Нет'
utv_not_in_ik['УТВ'] = 'Да'
utv_not_in_ik['Что_совпало'] = utv_not_in_ik['ФИАС'].apply(
    lambda x: 'Есть в УТВ и ФИАС, но НЕТ в ИК' if x == 'Да' else 'Есть только в УТВ (НЕТ в ИК и ФИАС)'
)

df_diff_ik_utv = pd.concat([ik_not_in_utv, utv_not_in_ik], ignore_index=True)

all_fias_found = set(df_ik['_fias_idx'].dropna().astype(int)) | set(df_utv['_fias_idx'].dropna().astype(int))
df_only_fias = df_fias[~df_fias.index.isin(all_fias_found)].copy()
df_only_fias['ФИАС'] = 'Да'
df_only_fias['УТВ'] = 'Нет'
df_only_fias['ИК'] = 'Нет'
df_only_fias['Что_совпало'] = 'Есть только в ФИАС (НЕТ ни в ИК, ни в УТВ)'

df_full = pd.concat([df_ik, utv_not_in_ik, df_only_fias], ignore_index=True)

for d in [df_full, df_ik, df_diff_ik_utv, df_only_fias]:
    d['Населенный_пункт'] = d['city'].str.capitalize()

service_cols = ['clean', 'city', 'block', '_utv_idx', '_fias_idx', 'Источник', 'Точный_дубль_строки', 'Дубль_после_очистки']
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
# 4. СТАТИСТИКА ДУБЛЕЙ + АНАЛИТИКА ПО УНИКАЛЬНЫМ
# ==========================================
dups_summary = pd.DataFrame([
    {'База': 'ИК', 'Исходно строк': raw_n_ik, 'Точных дублей': exact_dup_ik, 'Дублей после очистки (в т.ч. точные)': norm_dup_ik, 'Уникальных адресов': len(df_ik), '% дублей': f'{norm_dup_ik/raw_n_ik:.2%}'},
    {'База': 'УТВ', 'Исходно строк': raw_n_utv, 'Точных дублей': exact_dup_utv, 'Дублей после очистки (в т.ч. точные)': norm_dup_utv, 'Уникальных адресов': len(df_utv), '% дублей': f'{norm_dup_utv/raw_n_utv:.2%}'},
    {'База': 'ФИАС', 'Исходно строк': raw_n_fias, 'Точных дублей': exact_dup_fias, 'Дублей после очистки (в т.ч. точные)': norm_dup_fias, 'Уникальных адресов': len(df_fias), '% дублей': f'{norm_dup_fias/raw_n_fias:.2%}'},
])

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
        {'Территория': label, 'Показатель': 'Уникальных адресов в ИК', 'Кол-во': n_ik, '% от ИК': '100.00%', '% от УТВ': '-', '% от ФИАС': '-'},
        {'Территория': label, 'Показатель': 'Уникальных адресов в УТВ', 'Кол-во': n_utv, '% от ИК': '-', '% от УТВ': '100.00%', '% от ФИАС': '-'},
        {'Территория': label, 'Показатель': 'Уникальных адресов в ФИАС', 'Кол-во': n_fias, '% от ИК': '-', '% от УТВ': '-', '% от ФИАС': '100.00%'},
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
        df_ik[df_ik['city'] == c], df_utv[df_utv['city'] == c], df_fias[df_fias['city'] == c],
        utv_not_in_ik[utv_not_in_ik['city'] == c], df_only_fias[df_only_fias['city'] == c], c.capitalize()
    ))

df_summary = pd.DataFrame(all_stats)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("\n" + "=" * 110)
print("1. СТАТИСТИКА ДУБЛЕЙ ВНУТРИ БАЗ")
print("=" * 110)
print(dups_summary.to_string(index=False))
print("\n" + "=" * 110)
print("2. АНАЛИТИКА СВЕРКИ ПО УНИКАЛЬНЫМ АДРЕСАМ (МИРНЫЙ, УДАЧНЫЙ, АЙХАЛ)")
print("=" * 110)
print(df_summary[df_summary['Территория'] == 'ВСЕГО (Мирный + Удачный + Айхал)'].to_string(index=False))
print("=" * 110 + "\n")

# ==========================================
# 5. СОХРАНЕНИЕ В EXCEL ПО ЛИСТАМ
# ==========================================
output_file = 'Сверка_Мирный_Удачный_Айхал_без_дублей.xlsx'
try:
    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
except ImportError:
    writer = pd.ExcelWriter(output_file, engine='openpyxl')

with writer:
    df_summary.to_excel(writer, sheet_name='1_Аналитика_и_города', index=False)
    dups_summary.to_excel(writer, sheet_name='1_Аналитика_и_города', startrow=len(df_summary) + 3, index=False)
    df_full_clean.to_excel(writer, sheet_name='2_Полный_объединенный', index=False)
    df_ik_clean.to_excel(writer, sheet_name='3_Проверка_ИК', index=False)
    df_diff_clean.to_excel(writer, sheet_name='4_Не_сходится_ИК_и_УТВ', index=False)
    df_only_fias_clean.to_excel(writer, sheet_name='5_Есть_в_ФИАС_нет_у_нас', index=False)
    df_all_dups.to_excel(writer, sheet_name='6_Дубли_внутри_баз', index=False)

print(f"Готово за {time.time() - start_time:.1f} сек! Файл сохранен: {output_file}")
