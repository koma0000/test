import re
import time
import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz

start_time = time.time()
TARGET_CITIES = {'мирный', 'удачный', 'айхал'}
THRESHOLD = 88  # Чуть строже порог, чтобы не путать соседние номера квартир

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
    # Убираем административные слова и сокращения
    stop_words = (
        r'г|ул|д|кв|респ|у|уд|улус|тер|онт|снт|пл|пер|пр-кт|проезд|ш|б-р|'
        r'город|село|с|п|пгт|саха|якутия|мирнинский|мр-н|с/п|поселок|посёлок|'
        r'мкр|микрорайон|кв-л|квартал|корп|корпус|стр|строение|лит|литера|дом|квартира'
    )
    text = re.sub(rf'\b({stop_words})\b\.?', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    # Убираем ведущие нули ("014" -> "14")
    text = re.sub(r'\b0+(\d+)\b', r'\1', text)
    return re.sub(r'\s+', ' ', text).strip()

def make_block_key(row):
    c = row['city']
    street_part = re.sub(rf'\b{c}\b', '', row['clean']).strip()
    words = [w for w in street_part.split() if not w.isdigit()]
    prefix = words[0][:2] if words else 'num'
    return f"{c}_{prefix}"

# ==========================================
# 1. ФИЛЬТРАЦИЯ ПО 3 ГОРОДАМ И УДАЛЕНИЕ ДУБЛЕЙ
# ==========================================
df_ik_raw = df_ik[df_ik['Наименование'].apply(detect_city).isin(TARGET_CITIES)].copy()
df_utv_raw = df_utv[df_utv['Наименование'].apply(detect_city).isin(TARGET_CITIES)].copy()
df_fias_raw = df_fias[df_fias['Полный адрес'].apply(detect_city).isin(TARGET_CITIES)].copy()

for df_tmp, col, name in [(df_ik_raw, 'Наименование', 'ИК'), (df_utv_raw, 'Наименование', 'УТВ'), (df_fias_raw, 'Полный адрес', 'ФИАС')]:
    df_tmp['city'] = df_tmp[col].apply(detect_city)
    df_tmp['clean'] = df_tmp[col].apply(normalize_address)
    df_tmp['Источник'] = name
    df_tmp['Точный_дубль'] = df_tmp.duplicated(subset=[col], keep='first')
    df_tmp['Дубль_очищенный'] = df_tmp.duplicated(subset=['city', 'clean'], keep='first')

dups_ik = df_ik_raw[df_ik_raw['Дубль_очищенный']].copy()
dups_utv = df_utv_raw[df_utv_raw['Дубль_очищенный']].copy()
dups_fias = df_fias_raw[df_fias_raw['Дубль_очищенный']].copy()

df_ik = df_ik_raw[~df_ik_raw['Дубль_очищенный']].reset_index(drop=True)
df_utv = df_utv_raw[~df_utv_raw['Дубль_очищенный']].reset_index(drop=True)
df_fias = df_fias_raw[~df_fias_raw['Дубль_очищенный']].reset_index(drop=True)

df_all_dups = pd.concat([
    dups_ik[['Источник', 'city', 'Наименование', 'clean', 'Точный_дубль']],
    dups_utv[['Источник', 'city', 'Наименование', 'clean', 'Точный_дубль']],
    dups_fias[['Источник', 'city', 'Полный адрес', 'clean', 'Точный_дубль']].rename(columns={'Полный адрес': 'Наименование'})
], ignore_index=True)

df_ik['block'] = df_ik.apply(make_block_key, axis=1)
df_utv['block'] = df_utv.apply(make_block_key, axis=1)
df_fias['block'] = df_fias.apply(make_block_key, axis=1)

print(f"Подготовка завершена за {time.time() - start_time:.1f} сек. Запускаю честную сверку 1-к-1...")

# ==========================================
# 2. ЧЕСТНАЯ СВЕРКА 1-К-1 (БЕЗ СКЛЕЙКИ НЕСКОЛЬКИХ СТРОК В ОДНУ)
# ==========================================
def match_one_to_one(df_left, df_right, right_col_orig, threshold=88):
    n_left = len(df_left)
    matched_r_idx = np.full(n_left, -1, dtype=int)
    matched_scores = np.zeros(n_left, dtype=float)
    matched_vals = [None] * n_left

    right_orig_arr = df_right[right_col_orig].values
    right_groups = {k: v for k, v in df_right.groupby('block')}

    for block_key, left_group in df_left.groupby('block'):
        right_group = right_groups.get(block_key)
        if right_group is None or right_group.empty:
            continue

        l_indices = left_group.index.values
        r_indices = right_group.index.values

        # Используем token_sort_ratio: дом и квартира больше не склеятся!
        mat = process.cdist(
            left_group['clean'].tolist(),
            right_group['clean'].tolist(),
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold,
            workers=-1
        )

        # Находим все пары >= threshold и сортируем от самых точных к менее точным
        rows, cols = np.where(mat >= threshold)
        if len(rows) == 0:
            continue
        scores = mat[rows, cols]
        order = np.argsort(-scores)

        used_l = set()
        used_r = set()
        for idx in order:
            r_pos = rows[idx]
            c_pos = cols[idx]
            if r_pos not in used_l and c_pos not in used_r:
                used_l.add(r_pos)
                used_r.add(c_pos)
                orig_l = l_indices[r_pos]
                orig_r = r_indices[c_pos]
                matched_r_idx[orig_l] = orig_r
                matched_scores[orig_l] = scores[idx]
                matched_vals[orig_l] = right_orig_arr[orig_r]

    return matched_r_idx, matched_vals, matched_scores

# 1) ИК <-> УТВ (строго 1-к-1)
idx_utv, addr_utv, score_utv = match_one_to_one(df_ik, df_utv, 'Наименование', THRESHOLD)
df_ik['_utv_idx'] = [i if i != -1 else None for i in idx_utv]
df_ik['Адрес_в_УТВ'] = addr_utv
df_ik['Схожесть_УТВ_%'] = score_utv
df_ik['УТВ'] = ['Да' if i != -1 else 'Нет' for i in idx_utv]

# 2) ИК <-> ФИАС (строго 1-к-1)
idx_fias_ik, addr_fias_ik, score_fias_ik = match_one_to_one(df_ik, df_fias, 'Полный адрес', THRESHOLD)
df_ik['_fias_idx'] = [i if i != -1 else None for i in idx_fias_ik]
df_ik['Адрес_в_ФИАС'] = addr_fias_ik
df_ik['Схожесть_ФИАС_%'] = score_fias_ik
df_ik['ФИАС'] = ['Да' if i != -1 else 'Нет' for i in idx_fias_ik]
df_ik['ИК'] = 'Да'

# 3) УТВ <-> ФИАС (строго 1-к-1)
idx_fias_utv, addr_fias_utv, score_fias_utv = match_one_to_one(df_utv, df_fias, 'Полный адрес', THRESHOLD)
df_utv['_fias_idx'] = [i if i != -1 else None for i in idx_fias_utv]
df_utv['Адрес_в_ФИАС'] = addr_fias_utv
df_utv['Схожесть_ФИАС_%'] = score_fias_utv
df_utv['ФИАС'] = ['Да' if i != -1 else 'Нет' for i in idx_fias_utv]

def status_ik(row):
    if row['УТВ'] == 'Да' and row['ФИАС'] == 'Да': return 'Совпало везде (ИК + УТВ + ФИАС)'
    elif row['УТВ'] == 'Да': return 'Есть в ИК и УТВ, но НЕТ в ФИАС'
    elif row['ФИАС'] == 'Да': return 'Есть в ИК и ФИАС, но НЕТ в УТВ'
    else: return 'Есть только в ИК (НЕТ в УТВ и ФИАС)'

df_ik['Что_совпало'] = df_ik.apply(status_ik, axis=1)

# ==========================================
# 3. ВЫДЕЛЕНИЕ РАСХОЖДЕНИЙ И ОСТАТКА ФИАС
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

# Все индексы ФИАС, которые сопоставились с ИК или с УТВ
fias_found_indices = set(df_ik['_fias_idx'].dropna().astype(int)) | set(df_utv['_fias_idx'].dropna().astype(int))
df_only_fias = df_fias[~df_fias.index.isin(fias_found_indices)].copy()
df_only_fias['ФИАС'] = 'Да'
df_only_fias['УТВ'] = 'Нет'
df_only_fias['ИК'] = 'Нет'
df_only_fias['Что_совпало'] = 'Есть только в ФИАС (НЕТ ни в ИК, ни в УТВ)'

df_full = pd.concat([df_ik, utv_not_in_ik, df_only_fias], ignore_index=True)

for d in [df_full, df_ik, df_diff_ik_utv, df_only_fias]:
    d['Населенный_пункт'] = d['city'].str.capitalize()

service_cols = ['clean', 'city', 'block', '_utv_idx', '_fias_idx', 'Источник', 'Точный_дубль', 'Дубль_очищенный']
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
# 4. СТАТИСТИКА ДУБЛЕЙ + ЧЕСТНАЯ АНАЛИТИКА
# ==========================================
dups_summary = pd.DataFrame([
    {'База': 'ИК', 'Исходно строк': len(df_ik_raw), 'Точных дублей': int(df_ik_raw['Точный_дубль'].sum()), 'Дублей после очистки': len(dups_ik), 'Уникальных адресов': len(df_ik), '% дублей': f'{len(dups_ik)/len(df_ik_raw):.2%}'},
    {'База': 'УТВ', 'Исходно строк': len(df_utv_raw), 'Точных дублей': int(df_utv_raw['Точный_дубль'].sum()), 'Дублей после очистки': len(dups_utv), 'Уникальных адресов': len(df_utv), '% дублей': f'{len(dups_utv)/len(df_utv_raw):.2%}'},
    {'База': 'ФИАС', 'Исходно строк': len(df_fias_raw), 'Точных дублей': int(df_fias_raw['Точный_дубль'].sum()), 'Дублей после очистки': len(dups_fias), 'Уникальных адресов': len(df_fias), '% дублей': f'{len(dups_fias)/len(df_fias_raw):.2%}'},
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
        {'Территория': label, 'Показатель': 'Совпало 1-к-1 между ИК и УТВ', 'Кол-во': c_ik_in_utv, '% от ИК': pct(c_ik_in_utv, n_ik), '% от УТВ': pct(c_ik_in_utv, n_utv), '% от ФИАС': '-'},
        {'Территория': label, 'Показатель': 'Не сошлось: есть в ИК, но НЕТ в УТВ', 'Кол-во': c_ik_not_utv, '% от ИК': pct(c_ik_not_utv, n_ik), '% от УТВ': '-', '% от ФИАС': '-'},
        {'Территория': label, 'Показатель': 'Не сошлось: есть в УТВ, но НЕТ в ИК', 'Кол-во': c_utv_not_ik, '% от ИК': '-', '% от УТВ': pct(c_utv_not_ik, n_utv), '% от ФИАС': '-'},
        {'Территория': label, 'Показатель': 'Совпало везде (ИК + УТВ + ФИАС)', 'Кол-во': c_all_3, '% от ИК': pct(c_all_3, n_ik), '% от УТВ': pct(c_all_3, n_utv), '% от ФИАС': pct(c_all_3, n_fias)},
        {'Территория': label, 'Показатель': 'Адресов ИК найдено в ФИАС', 'Кол-во': c_ik_in_fias, '% от ИК': pct(c_ik_in_fias, n_ik), '% от УТВ': '-', '% от ФИАС': pct(c_ik_in_fias, n_fias)},
        {'Территория': label, 'Показатель': 'Адресов УТВ найдено в ФИАС', 'Кол-во': c_utv_in_fias, '% от ИК': '-', '% от УТВ': pct(c_utv_in_fias, n_utv), '% от ФИАС': pct(c_utv_in_fias, n_fias)},
        {'Территория': label, 'Показатель': 'Адресов ФИАС, которые ЕСТЬ у нас (в ИК или УТВ)', 'Кол-во': c_fias_in_ours, '% от ИК': '-', '% от УТВ': '-', '% от ФИАС': pct(c_fias_in_ours, n_fias)},
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
print("2. ЧЕСТНАЯ АНАЛИТИКА 1-К-1 (МИРНЫЙ, УДАЧНЫЙ, АЙХАЛ)")
print("=" * 110)
print(df_summary[df_summary['Территория'] == 'ВСЕГО (Мирный + Удачный + Айхал)'].to_string(index=False))
print("=" * 110 + "\n")

# ==========================================
# 5. СОХРАНЕНИЕ В EXCEL ПО ЛИСТАМ
# ==========================================
output_file = 'Сверка_Мирный_Удачный_Айхал_1к1.xlsx'
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
