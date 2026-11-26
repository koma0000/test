import re
import pandas as pd
from rapidfuzz import process, fuzz

# ==========================================
# 1. НОРМАЛИЗАЦИЯ АДРЕСОВ
# ==========================================
def normalize_address(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # Убираем пометки в скобках, например (ЗАКРЛС)
    text = re.sub(r'\([^)]*\)', '', text)
    # Убираем сокращения (г, ул, д, кв, респ, у, тер и т.д.) и знаки препинания
    text = re.sub(r'\b(г|ул|д|кв|респ|у|тер|пл|пер|пр-кт|город|село|с|п)\b\.?', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    # Убираем ведущие нули в номерах квартир/домов ("014" -> "14")
    text = re.sub(r'\b0+(\d+)\b', r'\1', text)
    return re.sub(r'\s+', ' ', text).strip()

# ==========================================
# 2. ПОДГОТОВКА И НЕЧЕТКОЕ СРАВНЕНИЕ
# ==========================================
THRESHOLD = 85  # Порог схожести (85%)

df_ik = df_ik.copy().reset_index(drop=True)
df_utv = df_utv.copy().reset_index(drop=True)
df_fias = df_fias.copy().reset_index(drop=True)

df_ik['clean'] = df_ik['Наименование'].apply(normalize_address)
df_utv['clean'] = df_utv['Наименование'].apply(normalize_address)
df_fias['clean'] = df_fias['Полный адрес'].apply(normalize_address)

# --- 1) Сверяем ИК с УТВ ---
utv_clean = df_utv['clean'].tolist()
utv_orig = df_utv['Наименование'].tolist()

mat_ik_utv = process.cdist(df_ik['clean'].tolist(), utv_clean, scorer=fuzz.token_sort_ratio, workers=-1)
best_utv_idx = mat_ik_utv.argmax(axis=1)
best_utv_score = mat_ik_utv.max(axis=1)

df_ik['_utv_idx'] = [i if s >= THRESHOLD else None for i, s in zip(best_utv_idx, best_utv_score)]
df_ik['Адрес_в_УТВ'] = [utv_orig[i] if s >= THRESHOLD else None for i, s in zip(best_utv_idx, best_utv_score)]
df_ik['Схожесть_УТВ_%'] = best_utv_score
df_ik['УТВ'] = ['Да' if s >= THRESHOLD else 'Нет' for s in best_utv_score]

# --- 2) Сверяем ИК с ФИАС ---
fias_clean = df_fias['clean'].tolist()
fias_orig = df_fias['Полный адрес'].tolist()

res_ik_fias = [process.extractOne(a, fias_clean, scorer=fuzz.token_set_ratio) for a in df_ik['clean']]
df_ik['_fias_idx'] = [r[2] if (r and r[1] >= THRESHOLD) else None for r in res_ik_fias]
df_ik['Адрес_в_ФИАС'] = [fias_orig[r[2]] if (r and r[1] >= THRESHOLD) else None for r in res_ik_fias]
df_ik['Схожесть_ФИАС_%'] = [r[1] if r else 0 for r in res_ik_fias]
df_ik['ФИАС'] = ['Да' if (r and r[1] >= THRESHOLD) else 'Нет' for r in res_ik_fias]
df_ik['ИК'] = 'Да'

# --- 3) Сверяем УТВ с ФИАС (чтобы точно знать, каких адресов ФИАС нет ни в ИК, ни в УТВ) ---
res_utv_fias = [process.extractOne(a, fias_clean, scorer=fuzz.token_set_ratio) for a in df_utv['clean']]
df_utv['_fias_idx'] = [r[2] if (r and r[1] >= THRESHOLD) else None for r in res_utv_fias]
df_utv['Адрес_в_ФИАС'] = [fias_orig[r[2]] if (r and r[1] >= THRESHOLD) else None for r in res_utv_fias]
df_utv['Схожесть_ФИАС_%'] = [r[1] if r else 0 for r in res_utv_fias]
df_utv['ФИАС'] = ['Да' if (r and r[1] >= THRESHOLD) else 'Нет' for r in res_utv_fias]

# Статус по каждой строке ИК
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
# 1) Что НЕ сходится между ИК и УТВ:
# - есть в ИК, но нет в УТВ
ik_not_in_utv = df_ik[df_ik['УТВ'] == 'Нет'].copy()
ik_not_in_utv['Причина_расхождения'] = 'Есть в ИК, но отсутствует в УТВ'

# - есть в УТВ, но нет в ИК
matched_utv_indices = set(df_ik['_utv_idx'].dropna().astype(int))
utv_not_in_ik = df_utv[~df_utv.index.isin(matched_utv_indices)].copy()
utv_not_in_ik['ИК'] = 'Нет'
utv_not_in_ik['УТВ'] = 'Да'
utv_not_in_ik['Что_совпало'] = utv_not_in_ik['ФИАС'].apply(
    lambda x: 'Есть в УТВ и ФИАС, но НЕТ в ИК' if x == 'Да' else 'Есть только в УТВ (НЕТ в ИК и ФИАС)'
)
utv_not_in_ik['Причина_расхождения'] = 'Есть в УТВ, но отсутствует в ИК'

# Общая таблица всех расхождений между ИК и УТВ
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

# 3) Полный объединенный лист по всем трем базам
df_full = pd.concat([df_ik, utv_not_in_ik, df_only_fias], ignore_index=True)

# Чистим служебные колонки и ставим 3 главных поля (ФИАС, УТВ, ИК) в начало
service_cols = ['clean', '_utv_idx', '_fias_idx', 'Причина_расхождения']
first_cols = ['ФИАС', 'УТВ', 'ИК', 'Что_совпало', 'Код', 'Наименование', 'Адрес_в_УТВ', 'Схожесть_УТВ_%', 'Адрес_в_ФИАС', 'Схожесть_ФИАС_%']

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
# 4. ПОДРОБНАЯ АНАЛИТИКА В ПРОЦЕНТАХ
# ==========================================
n_ik = len(df_ik)
n_utv = len(df_utv)
n_fias = len(df_fias)

c_all_3 = (df_ik['Что_совпало'] == 'Совпало везде (ИК + УТВ + ФИАС)').sum()
c_ik_in_utv = (df_ik['УТВ'] == 'Да').sum()
c_ik_not_utv = (df_ik['УТВ'] == 'Нет').sum()
c_utv_not_ik = len(utv_not_in_ik)
c_ik_in_fias = (df_ik['ФИАС'] == 'Да').sum()
c_utv_in_fias = (df_utv['ФИАС'] == 'Да').sum()
c_fias_in_ours = len(all_fias_found)
c_only_fias = len(df_only_fias)

summary_rows = [
    {'Раздел': 'Общие объемы', 'Показатель': 'Всего записей в ИК', 'Количество': n_ik, '% от ИК': '100.0%', '% от УТВ': '-', '% от ФИАС': '-'},
    {'Раздел': 'Общие объемы', 'Показатель': 'Всего записей в УТВ', 'Количество': n_utv, '% от ИК': '-', '% от УТВ': '100.0%', '% от ФИАС': '-'},
    {'Раздел': 'Общие объемы', 'Показатель': 'Всего записей в ФИАС', 'Количество': n_fias, '% от ИК': '-', '% от УТВ': '-', '% от ФИАС': '100.0%'},

    {'Раздел': 'Сверка ИК и УТВ', 'Показатель': 'Совпало между ИК и УТВ', 'Количество': c_ik_in_utv, '% от ИК': f'{c_ik_in_utv/n_ik:.2%}', '% от УТВ': f'{c_ik_in_utv/n_utv:.2%}', '% от ФИАС': '-'},
    {'Раздел': 'Сверка ИК и УТВ', 'Показатель': 'Не сошлось: есть в ИК, но НЕТ в УТВ', 'Количество': c_ik_not_utv, '% от ИК': f'{c_ik_not_utv/n_ik:.2%}', '% от УТВ': '-', '% от ФИАС': '-'},
    {'Раздел': 'Сверка ИК и УТВ', 'Показатель': 'Не сошлось: есть в УТВ, но НЕТ в ИК', 'Количество': c_utv_not_ik, '% от ИК': '-', '% от УТВ': f'{c_utv_not_ik/n_utv:.2%}', '% от ФИАС': '-'},

    {'Раздел': 'Сверка с ФИАС', 'Показатель': 'Совпало во всех трех базах (ИК + УТВ + ФИАС)', 'Количество': c_all_3, '% от ИК': f'{c_all_3/n_ik:.2%}', '% от УТВ': f'{c_all_3/n_utv:.2%}', '% от ФИАС': f'{c_all_3/n_fias:.2%}'},
    {'Раздел': 'Сверка с ФИАС', 'Показатель': 'Адресов ИК найдено в ФИАС', 'Количество': c_ik_in_fias, '% от ИК': f'{c_ik_in_fias/n_ik:.2%}', '% от УТВ': '-', '% от ФИАС': f'{c_ik_in_fias/n_fias:.2%}'},
    {'Раздел': 'Сверка с ФИАС', 'Показатель': 'Адресов УТВ найдено в ФИАС', 'Количество': c_utv_in_fias, '% от ИК': '-', '% от УТВ': f'{c_utv_in_fias/n_utv:.2%}', '% от ФИАС': f'{c_utv_in_fias/n_fias:.2%}'},
    {'Раздел': 'Сверка с ФИАС', 'Показатель': 'Адресов ФИАС, которые есть у нас (в ИК или УТВ)', 'Количество': c_fias_in_ours, '% от ИК': '-', '% от УТВ': '-', '% от ФИАС': f'{c_fias_in_ours/n_fias:.2%}'},
    {'Раздел': 'Сверка с ФИАС', 'Показатель': 'Адресов ФИАС, КОТОРЫХ НЕТ НИ В ИК, НИ В УТВ', 'Количество': c_only_fias, '% от ИК': '-', '% от УТВ': '-', '% от ФИАС': f'{c_only_fias/n_fias:.2%}'},
]

df_summary = pd.DataFrame(summary_rows)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("\n" + "=" * 105)
print("АНАЛИТИКА СВЕРКИ АДРЕСОВ (ИК / УТВ / ФИАС)")
print("=" * 105)
print(df_summary.to_string(index=False))
print("=" * 105 + "\n")

# ==========================================
# 5. СОХРАНЕНИЕ В EXCEL ПО ЛИСТАМ
# ==========================================
output_file = 'Сверка_адресов_ИК_УТВ_ФИАС.xlsx'

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    df_summary.to_excel(writer, sheet_name='1_Аналитика', index=False)
    df_full_clean.to_excel(writer, sheet_name='2_Полный_объединенный', index=False)
    df_ik_clean.to_excel(writer, sheet_name='3_Проверка_ИК', index=False)
    df_diff_clean.to_excel(writer, sheet_name='4_Не_сходится_ИК_и_УТВ', index=False)
    df_only_fias_clean.to_excel(writer, sheet_name='5_Есть_в_ФИАС_нет_у_нас', index=False)

    for sheet_name in writer.sheets:
        ws = writer.sheets[sheet_name]
        for col in ws.columns:
            max_len = max((len(str(cell.value or '')) for cell in col[:100]), default=10)
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 55)

print(f"Готово! Файл сохранен: {output_file}")
