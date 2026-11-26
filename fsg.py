import re
import time
import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz

start_time = time.time()
TARGET_CITIES = {'мирный', 'удачный', 'айхал'}
THRESHOLD = 88

# ==========================================
# 1. ФУНКЦИИ ОПРЕДЕЛЕНИЯ ГОРОДА И ЗДАНИЯ (С КОРПУСАМИ, БЕЗ КВАРТИР И УЧАСТКОВ)
# ==========================================
def detect_city(text):
    if pd.isna(text):
        return None
    s = str(text).lower()
    if re.search(r'\bмирный\b', s): return 'мирный'
    if re.search(r'\bудачный\b', s): return 'удачный'
    if re.search(r'\bайхал\b', s): return 'айхал'
    return None

def extract_building_only(text):
    if pd.isna(text):
        return ""
    t = str(text).lower()
    # 1. Убираем пометки в скобках, например (ЗАКРЛС)
    t = re.sub(r'\([^)]*\)', '', t)
    
    # 2. Стандартизируем корпуса, строения и литеры ("10к1", "10 корп. 1", "10 к. 1" -> "10 корп 1")
    t = re.sub(r'(\d+)\s*(?:корп\.?|корпус|к\.?)\s*(\d+[а-яa-z]?)', r'\1 корп \2', t)
    t = re.sub(r'(\d+)\s*(?:стр\.?|строение)\s*(\d+[а-яa-z]?)', r'\1 стр \2', t)
    t = re.sub(r'(\d+)\s*(?:лит\.?|литера)\s*([а-яa-z0-9]+)', r'\1 лит \2', t)
    
    # 3. Отрезаем квартиры, комнаты, помещения, подъезды и блоки внутри МКД
    cut_pattern = r'(?:,|\s|^)(?:кв|квартира|комн|комната|пом|помещение|оф|офис|бокс|гараж|машино-?место|подъезд|под|блок|секция|эт|этаж)\b.*$'
    t = re.sub(cut_pattern, '', t)
    
    # Если после номера дома или корпуса через запятую осталась голая цифра квартиры ("д. 10 корп 1, 45") — отрезаем её
    t = re.sub(r'(\b(?:д|дом|корп|стр|лит)\.?\s*[0-9а-яa-z/-]+)\s*,\s*\d+.*$', r'\1', t)

    # 4. Убираем административные и муниципальные сокращения (вкл. г.п., м.р-н, улус и т.д.)
    t = re.sub(r'\b(г\.?\s*п\.?|м\.?\s*р-н|с\.?\s*п\.?)\b', ' ', t)
    stop_words = (
        r'г|ул|д|респ|у|уд|улус|тер|онт|снт|пл|пер|пр-кт|проезд|ш|б-р|'
        r'город|село|с|п|пгт|саха|якутия|мирнинский|мр-н|поселок|посёлок|'
        r'мкр|микрорайон|кв-л|квартал|дом'
    )
    t = re.sub(rf'\b({stop_words})\b\.?', ' ', t)
    t = re.sub(r'[^\w\s/-]', ' ', t)
    
    # Убираем ведущие нули в номерах ("05" -> "5")
    t = re.sub(r'\b0+(\d+)\b', r'\1', t)
    return re.sub(r'\s+', ' ', t).strip()

def make_block_key(row):
    c = row['city']
    street_part = re.sub(rf'\b{c}\b', '', row['building_clean']).strip()
    words = [w for w in street_part.split() if not w.isdigit()]
    prefix = words[0][:2] if words else 'num'
    return f"{c}_{prefix}"

# ==========================================
# 2. ФИЛЬТРАЦИЯ ФИАС (УБИРАЕМ УЧАСТКИ И НЕЗДАНИЯ) И СХЛОПЫВАНИЕ ДО ЗДАНИЙ
# ==========================================
df_fias_filtered = df_fias.copy()
df_fias_filtered.columns = df_fias_filtered.columns.str.strip()

# Оставляем только объекты с заполненным номером дома
if 'Дом' in df_fias_filtered.columns:
    df_fias_filtered = df_fias_filtered[
        df_fias_filtered['Дом'].notna() &
        (df_fias_filtered['Дом'].astype(str).str.strip() != '') &
        (df_fias_filtered['Дом'].astype(str).str.lower() != 'nan')
    ]

# Исключаем земельные участки
if 'Участок' in df_fias_filtered.columns:
    df_fias_filtered = df_fias_filtered[
        df_fias_filtered['Участок'].isna() |
        (df_fias_filtered['Участок'].astype(str).str.strip() == '') |
        (df_fias_filtered['Участок'].astype(str).str.lower() == 'nan')
    ]

# Исключаем некапитальные/неподходящие типы объектов, если колонка заполнена
if 'Тип объекта' in df_fias_filtered.columns:
    df_fias_filtered = df_fias_filtered[
        ~df_fias_filtered['Тип объекта'].astype(str).str.lower().str.contains('участок|машино-место|сооружение', na=False)
    ]

# Фильтруем по 3 городам
df_ik_raw = df_ik[df_ik['Наименование'].apply(detect_city).isin(TARGET_CITIES)].copy()
df_utv_raw = df_utv[df_utv['Наименование'].apply(detect_city).isin(TARGET_CITIES)].copy()
df_fias_raw = df_fias_filtered[df_fias_filtered['Полный адрес'].apply(detect_city).isin(TARGET_CITIES)].copy()

for df_tmp, col in [(df_ik_raw, 'Наименование'), (df_utv_raw, 'Наименование'), (df_fias_raw, 'Полный адрес')]:
    df_tmp['city'] = df_tmp[col].apply(detect_city)
    df_tmp['building_clean'] = df_tmp[col].apply(extract_building_only)

def collapse_to_buildings(df_raw):
    counts = df_raw.groupby(['city', 'building_clean']).size().rename('Кол_во_записей_в_здании')
    unique_bld = df_raw.drop_duplicates(subset=['city', 'building_clean']).copy()
    unique_bld = unique_bld.merge(counts, on=['city', 'building_clean'], how='left')
    unique_bld = unique_bld[unique_bld['building_clean'] != ''].reset_index(drop=True)
    return unique_bld

df_ik = collapse_to_buildings(df_ik_raw)
df_utv = collapse_to_buildings(df_utv_raw)
df_fias = collapse_to_buildings(df_fias_raw)

df_ik['block'] = df_ik.apply(make_block_key, axis=1)
df_utv['block'] = df_utv.apply(make_block_key, axis=1)
df_fias['block'] = df_fias.apply(make_block_key, axis=1)

print(f"Здания выделены за {time.time() - start_time:.1f} сек: ИК={len(df_ik)}, УТВ={len(df_utv)}, ФИАС={len(df_fias)}. Начинаю сверку...")

# ==========================================
# 3. ЧЕСТНАЯ СВЕРКА ЗДАНИЙ 1-К-1
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

        mat = process.cdist(
            left_group['building_clean'].tolist(),
            right_group['building_clean'].tolist(),
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold,
            workers=-1
        )

        rows, cols = np.where(mat >= threshold)
        if len(rows) == 0:
            continue
        scores = mat[rows, cols]
        order = np.argsort(-scores)

        used_l, used_r = set(), set()
        for idx in order:
            r_pos, c_pos = rows[idx], cols[idx]
            if r_pos not in used_l and c_pos not in used_r:
                used_l.add(r_pos)
                used_r.add(c_pos)
                orig_l, orig_r = l_indices[r_pos], r_indices[c_pos]
                matched_r_idx[orig_l] = orig_r
                matched_scores[orig_l] = scores[idx]
                matched_vals[orig_l] = right_orig_arr[orig_r]

    return matched_r_idx, matched_vals, matched_scores

# 1) Здания ИК <-> Здания УТВ
idx_utv, addr_utv, score_utv = match_one_to_one(df_ik, df_utv, 'building_clean', THRESHOLD)
df_ik['_utv_idx'] = [i if i != -1 else None for i in idx_utv]
df_ik['Здание_в_УТВ'] = addr_utv
df_ik['Схожесть_УТВ_%'] = score_utv
df_ik['УТВ'] = ['Да' if i != -1 else 'Нет' for i in idx_utv]

# 2) Здания ИК <-> Здания ФИАС
idx_fias_ik, addr_fias_ik, score_fias_ik = match_one_to_one(df_ik, df_fias, 'building_clean', THRESHOLD)
df_ik['_fias_idx'] = [i if i != -1 else None for i in idx_fias_ik]
df_ik['Здание_в_ФИАС'] = addr_fias_ik
df_ik['Схожесть_ФИАС_%'] = score_fias_ik
df_ik['ФИАС'] = ['Да' if i != -1 else 'Нет' for i in idx_fias_ik]
df_ik['ИК'] = 'Да'

# 3) Здания УТВ <-> Здания ФИАС
idx_fias_utv, addr_fias_utv, score_fias_utv = match_one_to_one(df_utv, df_fias, 'building_clean', THRESHOLD)
df_utv['_fias_idx'] = [i if i != -1 else None for i in idx_fias_utv]
df_utv['Здание_в_ФИАС'] = addr_fias_utv
df_utv['Схожесть_ФИАС_%'] = score_fias_utv
df_utv['ФИАС'] = ['Да' if i != -1 else 'Нет' for i in idx_fias_utv]

def status_ik(row):
    if row['УТВ'] == 'Да' and row['ФИАС'] == 'Да': return 'Совпало везде (ИК + УТВ + ФИАС)'
    elif row['УТВ'] == 'Да': return 'Есть в ИК и УТВ, но НЕТ в ФИАС'
    elif row['ФИАС'] == 'Да': return 'Есть в ИК и ФИАС, но НЕТ в УТВ'
    else: return 'Есть только в ИК (НЕТ в УТВ и ФИАС)'

df_ik['Что_совпало'] = df_ik.apply(status_ik, axis=1)

# ==========================================
# 4. ВЫДЕЛЕНИЕ РАСХОЖДЕНИЙ И ОСТАТКА ФИАС
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

fias_found_indices = set(df_ik['_fias_idx'].dropna().astype(int)) | set(df_utv['_fias_idx'].dropna().astype(int))
df_only_fias = df_fias[~df_fias.index.isin(fias_found_indices)].copy()
df_only_fias['ФИАС'] = 'Да'
df_only_fias['УТВ'] = 'Нет'
df_only_fias['ИК'] = 'Нет'
df_only_fias['Что_совпало'] = 'Есть только в ФИАС (НЕТ ни в ИК, ни в УТВ)'

df_full = pd.concat([df_ik, utv_not_in_ik, df_only_fias], ignore_index=True)

for d in [df_full, df_ik, df_diff_ik_utv, df_only_fias]:
    d['Населенный_пункт'] = d['city'].str.capitalize()

service_cols = ['city', 'block', '_utv_idx', '_fias_idx']
first_cols = [
    'ФИАС', 'УТВ', 'ИК', 'Населенный_пункт', 'Что_совпало',
    'building_clean', 'Кол_во_записей_в_здании',
    'Здание_в_УТВ', 'Схожесть_УТВ_%', 'Здание_в_ФИАС', 'Схожесть_ФИАС_%',
    'Наименование', 'Полный адрес'
]

def format_df(df):
    d = df.drop(columns=[c for c in service_cols if c in df.columns])
    d = d.rename(columns={'building_clean': 'Здание_нормализованное'})
    head = [c if c != 'building_clean' else 'Здание_нормализованное' for c in first_cols if (c in d.columns or (c == 'building_clean' and 'Здание_нормализованное' in d.columns))]
    tail = [c for c in d.columns if c not in head]
    return d[head + tail]

df_full_clean = format_df(df_full)
df_ik_clean = format_df(df_ik)
df_diff_clean = format_df(df_diff_ik_utv)
df_only_fias_clean = format_df(df_only_fias)

# ==========================================
# 5. ПОДРОБНАЯ АНАЛИТИКА ПО ЗДАНИЯМ В ПРОЦЕНТАХ
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
        {'Территория': label, 'Показатель': 'Уникальных зданий (вкл. корпуса) в ИК', 'Кол-во': n_ik, '% от ИК': '100.00%', '% от УТВ': '-', '% от ФИАС': '-'},
        {'Территория': label, 'Показатель': 'Уникальных зданий (вкл. корпуса) в УТВ', 'Кол-во': n_utv, '% от ИК': '-', '% от УТВ': '100.00%', '% от ФИАС': '-'},
        {'Территория': label, 'Показатель': 'Уникальных зданий (вкл. корпуса) в ФИАС', 'Кол-во': n_fias, '% от ИК': '-', '% от УТВ': '-', '% от ФИАС': '100.00%'},
        {'Территория': label, 'Показатель': 'Совпало зданий между ИК и УТВ', 'Кол-во': c_ik_in_utv, '% от ИК': pct(c_ik_in_utv, n_ik), '% от УТВ': pct(c_ik_in_utv, n_utv), '% от ФИАС': '-'},
        {'Территория': label, 'Показатель': 'Не сошлось: здание есть в ИК, но НЕТ в УТВ', 'Кол-во': c_ik_not_utv, '% от ИК': pct(c_ik_not_utv, n_ik), '% от УТВ': '-', '% от ФИАС': '-'},
        {'Территория': label, 'Показатель': 'Не сошлось: здание есть в УТВ, но НЕТ в ИК', 'Кол-во': c_utv_not_ik, '% от ИК': '-', '% от УТВ': pct(c_utv_not_ik, n_utv), '% от ФИАС': '-'},
        {'Территория': label, 'Показатель': 'Совпало везде (ИК + УТВ + ФИАС)', 'Кол-во': c_all_3, '% от ИК': pct(c_all_3, n_ik), '% от УТВ': pct(c_all_3, n_utv), '% от ФИАС': pct(c_all_3, n_fias)},
        {'Территория': label, 'Показатель': 'Зданий ИК найдено в ФИАС', 'Кол-во': c_ik_in_fias, '% от ИК': pct(c_ik_in_fias, n_ik), '% от УТВ': '-', '% от ФИАС': pct(c_ik_in_fias, n_fias)},
        {'Территория': label, 'Показатель': 'Зданий УТВ найдено в ФИАС', 'Кол-во': c_utv_in_fias, '% от ИК': '-', '% от УТВ': pct(c_utv_in_fias, n_utv), '% от ФИАС': pct(c_utv_in_fias, n_fias)},
        {'Территория': label, 'Показатель': 'Зданий ФИАС, которые ЕСТЬ у нас (в ИК или УТВ)', 'Кол-во': c_fias_in_ours, '% от ИК': '-', '% от УТВ': '-', '% от ФИАС': pct(c_fias_in_ours, n_fias)},
        {'Территория': label, 'Показатель': 'Зданий ФИАС, КОТОРЫХ НЕТ НИ В ИК, НИ В УТВ', 'Кол-во': c_only_fias, '% от ИК': '-', '% от УТВ': '-', '% от ФИАС': pct(c_only_fias, n_fias)},
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

print("\n" + "=" * 115)
print("АНАЛИТИКА СВЕРКИ ПО КАПИТАЛЬНЫМ ЗДАНИЯМ И КОРПУСАМ (МИРНЫЙ, УДАЧНЫЙ, АЙХАЛ)")
print("=" * 115)
print(df_summary[df_summary['Территория'] == 'ВСЕГО (Мирный + Удачный + Айхал)'].to_string(index=False))
print("=" * 115 + "\n")

# ==========================================
# 6. СОХРАНЕНИЕ В EXCEL ПО ЛИСТАМ
# ==========================================
output_file = 'Сверка_КАП_ЗДАНИЙ_Мирный_Удачный_Айхал.xlsx'
try:
    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
except ImportError:
    writer = pd.ExcelWriter(output_file, engine='openpyxl')

with writer:
    df_summary.to_excel(writer, sheet_name='1_Аналитика_по_зданиям', index=False)
    df_full_clean.to_excel(writer, sheet_name='2_Полный_реестр_зданий', index=False)
    df_ik_clean.to_excel(writer, sheet_name='3_Проверка_зданий_ИК', index=False)
    df_diff_clean.to_excel(writer, sheet_name='4_Расхождения_ИК_и_УТВ', index=False)
    df_only_fias_clean.to_excel(writer, sheet_name='5_Здания_ФИАС_нет_у_нас', index=False)

print(f"Готово за {time.time() - start_time:.1f} сек! Файл сохранен: {output_file}")
