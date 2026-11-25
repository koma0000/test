import re
import pandas as pd
from rapidfuzz import process, fuzz

# ==========================================
# 1. ФУНКЦИЯ НОРМАЛИЗАЦИИ АДРЕСОВ
# ==========================================
def normalize_address(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # Удаляем спецпометки в скобках, например (ЗАКРЛС)
    text = re.sub(r'\([^)]*\)', '', text)
    # Убираем сокращения типов адресных объектов и знаки препинания
    text = re.sub(r'\b(г|ул|д|кв|респ|у|тер|пл|пер|пр-кт|город|село|с|п)\b\.?', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    # Убираем ведущие нули в номерах (например, "014" -> "14")
    text = re.sub(r'\b0+(\d+)\b', r'\1', text)
    # Схлопываем пробелы
    return re.sub(r'\s+', ' ', text).strip()

# ==========================================
# 2. ПОДГОТОВКА И НЕЧЕТКИЙ ПОИСК (RAPIDFUZZ)
# ==========================================
THRESHOLD = 85  # Порог схожести в процентах

df_ik = df_ik.copy()
df_utv = df_utv.copy()
df_fias = df_fias.copy()

df_ik['clean_addr'] = df_ik['Наименование'].apply(normalize_address)
df_utv['clean_addr'] = df_utv['Наименование'].apply(normalize_address)
df_fias['clean_addr'] = df_fias['Полный адрес'].apply(normalize_address)

# --- 2.1 Поиск совпадений ИК -> УТВ ---
utv_choices = df_utv['clean_addr'].tolist()
utv_orig = df_utv['Наименование'].tolist()
utv_codes = df_utv['Код'].tolist()

matches_utv = process.cdist(
    df_ik['clean_addr'].tolist(),
    utv_choices,
    scorer=fuzz.token_sort_ratio,
    workers=-1
)

best_utv_idx = matches_utv.argmax(axis=1)
best_utv_scores = matches_utv.max(axis=1)

df_ik['УТВ_совпадение_адрес'] = [utv_orig[i] if s >= THRESHOLD else None for i, s in zip(best_utv_idx, best_utv_scores)]
df_ik['УТВ_код'] = [utv_codes[i] if s >= THRESHOLD else None for i, s in zip(best_utv_idx, best_utv_scores)]
df_ik['УТВ_процент'] = best_utv_scores
df_ik['УТВ'] = ['Да' if s >= THRESHOLD else 'Нет' for s in best_utv_scores]

# --- 2.2 Поиск совпадений ИК -> ФИАС ---
fias_choices = df_fias['clean_addr'].tolist()
fias_orig = df_fias['Полный адрес'].tolist()
fias_guids = df_fias['ГАР GUID ФИАС'].tolist()

fias_results_ik = [
    process.extractOne(addr, fias_choices, scorer=fuzz.token_set_ratio)
    for addr in df_ik['clean_addr']
]

df_ik['ФИАС_совпадение_адрес'] = [fias_orig[r[2]] if (r and r[1] >= THRESHOLD) else None for r in fias_results_ik]
df_ik['ГАР_GUID_ФИАС'] = [fias_guids[r[2]] if (r and r[1] >= THRESHOLD) else None for r in fias_results_ik]
df_ik['ФИАС_процент'] = [r[1] if r else 0 for r in fias_results_ik]
df_ik['ФИАС'] = ['Да' if (r and r[1] >= THRESHOLD) else 'Нет' for r in fias_results_ik]
df_ik['ИК'] = 'Да'

# --- 2.3 Поиск совпадений УТВ -> ФИАС (чтобы точно выделить остаток ФИАС, которого нет нигде у нас) ---
fias_results_utv = [
    process.extractOne(addr, fias_choices, scorer=fuzz.token_set_ratio)
    for addr in df_utv['clean_addr']
]
df_utv['ГАР_GUID_ФИАС'] = [fias_guids[r[2]] if (r and r[1] >= THRESHOLD) else None for r in fias_results_utv]
df_utv['ФИАС_совпадение_адрес'] = [fias_orig[r[2]] if (r and r[1] >= THRESHOLD) else None for r in fias_results_utv]
df_utv['ФИАС_процент'] = [r[1] if r else 0 for r in fias_results_utv]
df_utv['ФИАС'] = ['Да' if (r and r[1] >= THRESHOLD) else 'Нет' for r in fias_results_utv]

# Статус для ИК
def get_ik_status(row):
    in_utv = row['УТВ'] == 'Да'
    in_fias = row['ФИАС'] == 'Да'
    if in_utv and in_fias:
        return 'Совпало везде (ИК + УТВ + ФИАС)'
    elif in_utv:
        return 'Совпало ИК + УТВ (нет в ФИАС)'
    elif in_fias:
        return 'Совпало ИК + ФИАС (нет в УТВ)'
    else:
        return 'Только в ИК (нет ни в УТВ, ни в ФИАС)'

df_ik['Статус_совпадения'] = df_ik.apply(get_ik_status, axis=1)

# ==========================================
# 3. ПОЛНОЕ ОБЪЕДИНЕНИЕ ВСЕХ БАЗ И ОСТАТОК ФИАС
# ==========================================
# Все GUID из ФИАС, которые нашлись в ИК или в УТВ
guids_in_ik = set(df_ik['ГАР_GUID_ФИАС'].dropna())
guids_in_utv = set(df_utv['ГАР_GUID_ФИАС'].dropna())
all_our_fias_guids = guids_in_ik | guids_in_utv

# Остаток УТВ (чего нет в ИК)
matched_utv_codes = set(df_ik['УТВ_код'].dropna())
unmatched_utv = df_utv[~df_utv['Код'].isin(matched_utv_codes)].copy()
unmatched_utv['ИК'] = 'Нет'
unmatched_utv['УТВ'] = 'Да'
unmatched_utv['Статус_совпадения'] = unmatched_utv['ФИАС'].apply(
    lambda x: 'Совпало УТВ + ФИАС (нет в ИК)' if x == 'Да' else 'Только в УТВ (нет ни в ИК, ни в ФИАС)'
)

# Остаток ФИАС (ТО, ЧЕГО НЕТ У НАС НИ В ИК, НИ В УТВ)
unmatched_fias = df_fias[~df_fias['ГАР GUID ФИАС'].isin(all_our_fias_guids)].copy()
unmatched_fias['ИК'] = 'Нет'
unmatched_fias['УТВ'] = 'Нет'
unmatched_fias['ФИАС'] = 'Да'
unmatched_fias['Статус_совпадения'] = 'Только в ФИАС (отсутствует у нас в ИК и УТВ)'

# Объединяем всё в единый реестр
df_all_combined = pd.concat([df_ik, unmatched_utv, unmatched_fias], ignore_index=True)

priority_cols = [
    'ФИАС', 'УТВ', 'ИК', 'Статус_совпадения',
    'Код', 'Наименование',
    'УТВ_код', 'УТВ_совпадение_адрес', 'УТВ_процент',
    'ГАР_GUID_ФИАС', 'ФИАС_совпадение_адрес', 'ФИАС_процент'
]
other_cols = [c for c in df_all_combined.columns if c not in priority_cols and c != 'clean_addr']
df_all_combined = df_all_combined[priority_cols + other_cols]

# ==========================================
# 4. РАСЧЕТ ПОДРОБНОЙ ПРОЦЕНТНОЙ АНАЛИТИКИ
# ==========================================
n_ik = len(df_ik)
n_utv = len(df_utv)
n_fias = len(df_fias)
n_total = len(df_all_combined)

# Пересечения по ИК
c_all_three = (df_ik['Статус_совпадения'] == 'Совпало везде (ИК + УТВ + ФИАС)').sum()
c_ik_utv_all = (df_ik['УТВ'] == 'Да').sum()
c_ik_fias_all = (df_ik['ФИАС'] == 'Да').sum()
c_ik_utv_only = (df_ik['Статус_совпадения'] == 'Совпало ИК + УТВ (нет в ФИАС)').sum()
c_ik_fias_only = (df_ik['Статус_совпадения'] == 'Совпало ИК + ФИАС (нет в УТВ)').sum()
c_ik_only = (df_ik['Статус_совпадения'] == 'Только в ИК (нет ни в УТВ, ни в ФИАС)').sum()

# По УТВ
c_utv_fias_no_ik = (unmatched_utv['ФИАС'] == 'Да').sum()
c_utv_only = (unmatched_utv['ФИАС'] == 'Нет').sum()

# По ФИАС
c_fias_in_ours = len(all_our_fias_guids)  # уникальные записи ФИАС, которые есть у нас (в ИК или УТВ)
c_fias_not_in_ours = len(unmatched_fias)  # записи ФИАС, которых нет ни в ИК, ни в УТВ

analytics_rows = [
    # Блок 1: Общие объемы
    {'Блок': '1. Общие объемы баз', 'Показатель': 'Всего записей в базе ИК', 'Кол-во': n_ik, '% от ИК': '100.00%', '% от УТВ': '-', '% от ФИАС': '-', '% от общего реестра': f'{n_ik/n_total:.2%}'},
    {'Блок': '1. Общие объемы баз', 'Показатель': 'Всего записей в базе УТВ', 'Кол-во': n_utv, '% от ИК': '-', '% от УТВ': '100.00%', '% от ФИАС': '-', '% от общего реестра': f'{n_utv/n_total:.2%}'},
    {'Блок': '1. Общие объемы баз', 'Показатель': 'Всего записей в справочнике ФИАС', 'Кол-во': n_fias, '% от ИК': '-', '% от УТВ': '-', '% от ФИАС': '100.00%', '% от общего реестра': f'{n_fias/n_total:.2%}'},
    {'Блок': '1. Общие объемы баз', 'Показатель': 'Всего строк в объединенном реестре', 'Кол-во': n_total, '% от ИК': '-', '% от УТВ': '-', '% от ФИАС': '-', '% от общего реестра': '100.00%'},

    # Блок 2: Сверка базы ИК
    {'Блок': '2. Детализация по ИК', 'Показатель': 'Совпало везде (ИК + УТВ + ФИАС)', 'Кол-во': c_all_three, '% от ИК': f'{c_all_three/n_ik:.2%}', '% от УТВ': f'{c_all_three/n_utv:.2%}', '% от ФИАС': f'{c_all_three/n_fias:.2%}', '% от общего реестра': f'{c_all_three/n_total:.2%}'},
    {'Блок': '2. Детализация по ИК', 'Показатель': 'ИК найдено в УТВ (всего)', 'Кол-во': c_ik_utv_all, '% от ИК': f'{c_ik_utv_all/n_ik:.2%}', '% от УТВ': f'{c_ik_utv_all/n_utv:.2%}', '% от ФИАС': '-', '% от общего реестра': f'{c_ik_utv_all/n_total:.2%}'},
    {'Блок': '2. Детализация по ИК', 'Показатель': 'ИК найдено в ФИАС (всего)', 'Кол-во': c_ik_fias_all, '% от ИК': f'{c_ik_fias_all/n_ik:.2%}', '% от УТВ': '-', '% от ФИАС': f'{c_ik_fias_all/n_fias:.2%}', '% от общего реестра': f'{c_ik_fias_all/n_total:.2%}'},
    {'Блок': '2. Детализация по ИК', 'Показатель': 'Только ИК + УТВ (отсутствует в ФИАС)', 'Кол-во': c_ik_utv_only, '% от ИК': f'{c_ik_utv_only/n_ik:.2%}', '% от УТВ': f'{c_ik_utv_only/n_utv:.2%}', '% от ФИАС': '-', '% от общего реестра': f'{c_ik_utv_only/n_total:.2%}'},
    {'Блок': '2. Детализация по ИК', 'Показатель': 'Только ИК + ФИАС (отсутствует в УТВ)', 'Кол-во': c_ik_fias_only, '% от ИК': f'{c_ik_fias_only/n_ik:.2%}', '% от УТВ': '-', '% от ФИАС': f'{c_ik_fias_only/n_fias:.2%}', '% от общего реестра': f'{c_ik_fias_only/n_total:.2%}'},
    {'Блок': '2. Детализация по ИК', 'Показатель': 'Только в ИК (нет ни в УТВ, ни в ФИАС)', 'Кол-во': c_ik_only, '% от ИК': f'{c_ik_only/n_ik:.2%}', '% от УТВ': '-', '% от ФИАС': '-', '% от общего реестра': f'{c_ik_only/n_total:.2%}'},

    # Блок 3: Остаток УТВ (чего нет в ИК)
    {'Блок': '3. Остаток УТВ (нет в ИК)', 'Показатель': 'УТВ + ФИАС (отсутствует в ИК)', 'Кол-во': c_utv_fias_no_ik, '% от ИК': '-', '% от УТВ': f'{c_utv_fias_no_ik/n_utv:.2%}', '% от ФИАС': f'{c_utv_fias_no_ik/n_fias:.2%}', '% от общего реестра': f'{c_utv_fias_no_ik/n_total:.2%}'},
    {'Блок': '3. Остаток УТВ (нет в ИК)', 'Показатель': 'Только в УТВ (нет ни в ИК, ни в ФИАС)', 'Кол-во': c_utv_only, '% от ИК': '-', '% от УТВ': f'{c_utv_only/n_utv:.2%}', '% от ФИАС': '-', '% от общего реестра': f'{c_utv_only/n_total:.2%}'},

    # Блок 4: Анализ базы ФИАС (покрытие и то, чего нет у нас)
    {'Блок': '4. Анализ базы ФИАС', 'Показатель': 'Адресов ФИАС, которые ЕСТЬ у нас (в ИК или УТВ)', 'Кол-во': c_fias_in_ours, '% от ИК': '-', '% от УТВ': '-', '% от ФИАС': f'{c_fias_in_ours/n_fias:.2%}', '% от общего реестра': f'{c_fias_in_ours/n_total:.2%}'},
    {'Блок': '4. Анализ базы ФИАС', 'Показатель': 'Адресов в ФИАС, КОТОРЫХ НЕТ У НАС (ни в ИК, ни в УТВ)', 'Кол-во': c_fias_not_in_ours, '% от ИК': '-', '% от УТВ': '-', '% от ФИАС': f'{c_fias_not_in_ours/n_fias:.2%}', '% от общего реестра': f'{c_fias_not_in_ours/n_total:.2%}'},
]

df_summary = pd.DataFrame(analytics_rows)

# ==========================================
# 5. ВЫВОД В КОНСОЛЬ
# ==========================================
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 60)

print("\n" + "=" * 115)
print("ПОЛНАЯ ПРОЦЕНТНАЯ АНАЛИТИКА СОПОСТАВЛЕНИЯ АДРЕСОВ (ИК / УТВ / ФИАС)")
print("=" * 115)
print(df_summary.to_string(index=False))
print("=" * 115)
print(f"ИТОГО В ФИАС АДРЕСОВ, КОТОРЫХ НЕТ У НАС: {c_fias_not_in_ours} из {n_fias} ({c_fias_not_in_ours/n_fias:.2%} от базы ФИАС)")
print("=" * 115 + "\n")

# ==========================================
# 6. СОХРАНЕНИЕ В EXCEL ПО ЛИСТАМ
# ==========================================
output_file = 'Аналитика_адресов_полная.xlsx'

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # Лист 1: Полная сводная процентная аналитика
    df_summary.to_excel(writer, sheet_name='1_Сводная_аналитика', index=False)
    
    # Лист 2: Полный объединенный реестр с полями ФИАС, УТВ, ИК и статусами
    df_all_combined.to_excel(writer, sheet_name='2_Полный_объединенный', index=False)
    
    # Лист 3: Детальный разбор по ИК
    df_ik_clean = df_ik.drop(columns=['clean_addr'])
    df_ik_clean.to_excel(writer, sheet_name='3_Сверка_ИК_детально', index=False)
    
    # Лист 4: Расхождения по ИК (где не сошлись все 3 базы)
    df_problems = df_ik_clean[df_ik_clean['Статус_совпадения'] != 'Совпало везде (ИК + УТВ + ФИАС)']
    df_problems.to_excel(writer, sheet_name='4_Расхождения_ИК', index=False)
    
    # Лист 5: Остаток из УТВ (чего нет в ИК)
    unmatched_utv.drop(columns=['clean_addr'], errors='ignore').to_excel(writer, sheet_name='5_Только_в_УТВ', index=False)

    # Лист 6: Адреса из ФИАС, которых нет у нас (ни в ИК, ни в УТВ)
    unmatched_fias.drop(columns=['clean_addr'], errors='ignore').to_excel(writer, sheet_name='6_Только_в_ФИАС_нет_у_нас', index=False)

    # Автоподбор ширины колонок
    for sheet_name in writer.sheets:
        worksheet = writer.sheets[sheet_name]
        for col in worksheet.columns:
            max_len = 0
            col_letter = col[0].column_letter
            for cell in col[:100]:
                val = str(cell.value or '')
                if len(val) > max_len:
                    max_len = len(val)
            worksheet.column_dimensions[col_letter].width = min(max_len + 4, 55)

print(f"Файл успешно сохранен: {output_file}")
