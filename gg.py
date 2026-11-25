import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

# --- КОНФИГУРАЦИЯ ---
FIG_W, FIG_H, DPI = 14.0, 10.6, 300
H = 0.44
K = 31 / 426

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "figure.dpi": 130,
    "savefig.dpi": DPI,
    "hatch.linewidth": 1.1,
})

# Палитра
C_ROS      = "#1B4F72"   # Глубокий корпоративный синий (Роснефть - Москва)
C_ROS_SPB  = "#2E86C1"   # Средне-синий (Роснефть - СПб)
C_ROS_KZN  = "#5DADE2"   # Голубой (Роснефть - Казань)
C_ROS_NA   = "#AED6F1"   # Светло-голубой (Роснефть - не указан)
C_ROS_TXT  = "#154360"

C_UV       = "#B03A2E"   # Насыщенный терракотово-красный (Уволенные - Москва)
C_UV_SPB   = "#E6B0AA"   # Светло-коралловый (Уволенные - СПб)
C_UV_TXT   = "#922B21"

C_OTH      = "#D5DBDB"   # Спокойный светло-серый (Прочие)
C_OTH_TXT  = "#424949"

C_GHOST    = "#F2F4F4"   # Фон предыдущего уровня
C_BG       = "#FFFFFF"
C_CARD     = "#F8F9F9"
C_LINE     = "#E5E8E8"
C_DARK     = "#1C2833"
C_MUTED    = "#566573"

# Добавлены контрастные подложки (bbox) для цифр поверх штриховки
CITY_STYLE = {
    ("ros", "M"): dict(fc=C_ROS,     ec="white",   hatch=None,   tcol="white",   bbox=None),
    ("ros", "S"): dict(fc=C_ROS_SPB, ec="#85C1E9", hatch="///",  tcol="white",   bbox=dict(boxstyle="round,pad=0.14", fc=C_ROS, ec="none", alpha=0.88)),
    ("ros", "K"): dict(fc=C_ROS_KZN, ec="#D6EAF8", hatch="\\\\\\", tcol=C_DARK,  bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.85)),
    ("ros", "N"): dict(fc=C_ROS_NA,  ec="white",   hatch=None,   tcol=C_DARK,    bbox=None),
    ("uv",  "M"): dict(fc=C_UV,      ec="white",   hatch=None,   tcol="white",   bbox=None),
    ("uv",  "S"): dict(fc=C_UV_SPB,  ec=C_UV,      hatch="///",  tcol=C_UV_TXT,  bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85)),
}

CITY_NAMES = {
    "M": "Москва",
    "S": "Санкт-Петербург",
    "K": "Казань",
    "N": "город не указан"
}
ORDER = "MSKN"

ROWS = [
    dict(
        step="УРОВЕНЬ 1",
        lab="Всего ГСК\nв базе СПАРК",
        tot=426,
        sub="100 % выборки",
        parent=None,
        ros=dict(v=226, pct=53, city=dict(M=150, S=28, K=23, N=25)),
        uv=dict(v=25,   pct=11, city=dict(M=19, S=6)),
        oth=dict(v=175, pct=41, city=None),
    ),
    dict(
        step="УРОВЕНЬ 2",
        lab="Действующие или\nликвидированные\nс 01.01.2025",
        tot=343,
        sub="81 % от 426",
        parent=426,
        ros=dict(v=186, pct=54, city=dict(M=132, S=16, K=20, N=18)),
        uv=dict(v=18,   pct=10, city=dict(M=16, S=2)),
        oth=dict(v=139, pct=41, city=None),
    ),
    dict(
        step="УРОВЕНЬ 3 · ФИНАЛ",
        lab="Есть в учёте\nООО «Теплоснаб»",
        tot=31,
        sub="9 % от 343  ·  7 % от 426",
        parent=343,
        ros=dict(v=9,  pct=29, city=dict(M=8, N=1)),
        uv=dict(v=1,   pct=11, city=dict(M=1)),
        oth=dict(v=21, pct=68, city=None),
    ),
]

def draw_segment_bar(ax, y, x, key, grp, min_seg):
    if not grp["city"]:
        ax.barh(y, grp["v"], left=x, height=H, color=C_OTH, edgecolor="white", linewidth=1.5, zorder=3)
        if grp["v"] >= min_seg:
            ax.text(x + grp["v"] / 2, y, f"{grp['v']}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=C_OTH_TXT, zorder=6)
        return x + grp["v"]

    for ck in ORDER:
        v = grp["city"].get(ck, 0)
        if not v:
            continue
        st = CITY_STYLE[(key, ck)]
        ax.barh(y, v, left=x, height=H, color=st["fc"], edgecolor=st["ec"],
                hatch=st["hatch"], linewidth=1.0, zorder=3)
        if v >= min_seg:
            ax.text(x + v / 2, y, f"{v}", ha="center", va="center",
                    fontsize=9.2, fontweight="bold", color=st["tcol"],
                    bbox=st["bbox"], zorder=7)
        x += v
    return x

def draw_top_labels(ax, y, x0, x1, key, grp, is_bottom=False):
    cx = (x0 + x1) / 2
    col = {"ros": C_ROS_TXT, "uv": C_UV_TXT, "oth": C_OTH_TXT}[key]
    name = {"ros": "Сотрудники Роснефти", "uv": "Уволенные", "oth": "Прочие / не указан"}[key]

    if key == "uv":
        offset_x = 1.6 if is_bottom else 0.0
        ha_align = "left" if is_bottom else "center"
        ax.annotate(
            f"{name}: {grp['v']} ({grp['pct']} %)",
            xy=(cx, y + H / 2 + 0.02),
            xytext=(cx + offset_x, y + H / 2 + 0.36),
            ha=ha_align, va="bottom",
            fontsize=9.2, fontweight="bold", color=col, zorder=8,
            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.1, shrinkA=0, shrinkB=2)
        )
    else:
        ax.text(cx, y + H / 2 + 0.08, f"{name}\n{grp['v']}  ({grp['pct']} %)",
                ha="center", va="bottom", fontsize=9.8, fontweight="bold",
                color=col, linespacing=1.25, zorder=8)

def draw_city_pills(ax, y, row):
    y_pos = y - H / 2 - 0.11
    parts_ros = [f"{CITY_NAMES[k]}: {row['ros']['city'][k]}" for k in ORDER if row["ros"]["city"].get(k)]
    parts_uv  = [f"{CITY_NAMES[k]}: {row['uv']['city'][k]}"  for k in ORDER if row["uv"]["city"].get(k)]

    line1 = "Роснефть:  " + "  •  ".join(parts_ros)
    line2 = "Уволенные:  " + "  •  ".join(parts_uv)

    ax.text(0, y_pos, line1, fontsize=8.4, color=C_ROS_TXT, va="top", ha="left")
    ax.text(0, y_pos - 0.21, line2, fontsize=8.4, color=C_UV_TXT, va="top", ha="left")

def render_level(ax, y, row, min_seg, tot_x, ghost=True, is_bottom=False):
    if ghost and row["parent"]:
        ax.barh(y, row["parent"], height=H, color=C_GHOST, edgecolor="#D5DBDB", linewidth=0.8, zorder=1)
        loss = row["parent"] - row["tot"]
        ax.text((row["tot"] + row["parent"]) / 2, y,
                f"выбыло: {loss} (−{loss / row['parent'] * 100:.0f} %)",
                ha="center", va="center", fontsize=8.8, style="italic", color=C_MUTED, zorder=2)

    x, boundaries = 0, []
    for key in ("ros", "uv", "oth"):
        x0 = x
        x = draw_segment_bar(ax, y, x, key, row[key], min_seg)
        boundaries.append(x0)
        draw_top_labels(ax, y, x0, x, key, row[key], is_bottom=is_bottom)

    for xb in boundaries[1:]:
        ax.plot([xb, xb], [y - H / 2 - 0.01, y + H / 2 + 0.01], color="white", lw=2.5, zorder=5)

    draw_city_pills(ax, y, row)

    ax.text(tot_x, y + 0.03, f"{row['tot']}", fontsize=18, fontweight="bold",
            color=C_DARK, va="bottom", ha="left")
    ax.text(tot_x, y - 0.06, row["sub"], fontsize=9.2, fontweight="bold",
            color=C_MUTED, va="top", ha="left")


# --- СОЗДАНИЕ ФИГУРЫ ---
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=C_BG)

header_line = FancyBboxPatch((0.04, 0.965), 0.92, 0.006, boxstyle="round,pad=0,rounding_size=0.002",
                             fc=C_ROS, ec="none", transform=fig.transFigure)
fig.patches.append(header_line)

fig.text(0.04, 0.948, "Воронка ГСК: структура по председателям и географии",
         fontsize=18.5, fontweight="bold", color=C_DARK, va="top")

fig.text(0.04, 0.910,
         "На первых двух этапах доля сотрудников Роснефти стабильна (53–54 %). При переходе в учёт ООО «Теплоснаб»\n"
         "доля «прочих» вырастает до 68 % (21 из 31), тогда как конверсия из базы СПАРК у Роснефти составляет 4,0 % (9 из 226).",
         fontsize=10.0, color=C_MUTED, va="top", linespacing=1.45)

zoom_card = FancyBboxPatch((0.035, 0.215), 0.93, 0.225, boxstyle="round,pad=0.01,rounding_size=0.015",
                           fc=C_CARD, ec=C_LINE, linewidth=1.2, transform=fig.transFigure, zorder=0)
fig.patches.append(zoom_card)

fig.text(0.05, 0.425, "МАСШТАБ 100 %  ·  ДЕТАЛИЗАЦИЯ УРОВНЯ 3 (31 ГСК = 9 % от действующих, 7 % от базы СПАРК)",
         fontsize=8.8, fontweight="bold", color=C_ROS, va="top", zorder=2)

axA = fig.add_axes([0.22, 0.48, 0.73, 0.34], zorder=1)
axB = fig.add_axes([0.22, 0.245, 0.73, 0.135], zorder=1)
axB.set_facecolor(C_CARD)

render_level(axA, 1.75, ROWS[0], min_seg=14, tot_x=440, ghost=True, is_bottom=False)
render_level(axA, 0.00, ROWS[1], min_seg=14, tot_x=440, ghost=True, is_bottom=False)
render_level(axB, 0.00, ROWS[2], min_seg=1.5, tot_x=440 * K, ghost=False, is_bottom=True)

axA.set_xlim(-4, 535)
axA.set_ylim(-0.62, 2.55)
axB.set_xlim(-4 * K, 535 * K)
axB.set_ylim(-0.60, 0.98)

for ax_curr, y_coords, rows_subset in [(axA, [1.75, 0.00], [ROWS[0], ROWS[1]]), (axB, [0.00], [ROWS[2]])]:
    ax_curr.set_yticks(y_coords)
    ax_curr.set_yticklabels([""] * len(y_coords))
    for yc, r in zip(y_coords, rows_subset):
        ax_curr.text(-14 if ax_curr == axA else -14 * K, yc + 0.16, r["step"],
                     ha="right", va="bottom", fontsize=7.8, fontweight="bold",
                     color=C_ROS if "ФИНАЛ" in r["step"] else C_MUTED)
        ax_curr.text(-14 if ax_curr == axA else -14 * K, yc + 0.08, r["lab"],
                     ha="right", va="top", fontsize=10.5, fontweight="bold",
                     color=C_DARK, linespacing=1.35)

for a in (axA, axB):
    a.set_xticks([])
    for s in ("top", "right", "bottom", "left"):
        a.spines[s].set_visible(False)
    a.tick_params(axis="y", length=0)

legend_box = FancyBboxPatch((0.035, 0.075), 0.93, 0.115, boxstyle="round,pad=0.008,rounding_size=0.012",
                            fc="#FBFBFB", ec=C_LINE, linewidth=1.0, transform=fig.transFigure, zorder=0)
fig.patches.append(legend_box)

fig.legend(
    handles=[
        Patch(fc=C_ROS,     ec="white",   label="Сотрудник Роснефти (Москва)"),
        Patch(fc=C_ROS_SPB, ec="#85C1E9", hatch="///", label="Санкт-Петербург"),
        Patch(fc=C_ROS_KZN, ec="#D6EAF8", hatch="\\\\\\", label="Казань"),
        Patch(fc=C_ROS_NA,  ec="white",   label="Город не указан"),
    ],
    loc="upper left", bbox_to_anchor=(0.045, 0.185), ncol=4, frameon=False,
    fontsize=8.6, columnspacing=1.6, handlelength=1.8,
    title="Категория «Сотрудник Роснефти» и география:",
    title_fontsize=8.8, alignment="left"
)

fig.legend(
    handles=[
        Patch(fc=C_UV,     ec="white", label="Уволенный (Москва)"),
        Patch(fc=C_UV_SPB, ec=C_UV,    hatch="///", label="Уволенный (СПб)"),
        Patch(fc=C_OTH,    ec="white", label="Прочие / не указан (без разбивки по городам)"),
        Patch(fc=C_GHOST,  ec="#D5DBDB", label="Отсев с предыдущего уровня"),
    ],
    loc="upper left", bbox_to_anchor=(0.045, 0.128), ncol=4, frameon=False,
    fontsize=8.6, columnspacing=1.6, handlelength=1.8,
    title="Прочие категории:",
    title_fontsize=8.8, alignment="left"
)

fig.text(0.04, 0.052,
         "Методология: доли «Сотрудников Роснефти» и «Прочих» рассчитаны от итога строки; доля «Уволенных» — от числа сотрудников Роснефти (25/226 = 11 %, 18/186 = 10 %, 1/9 = 11 %).",
         fontsize=8.0, color=C_MUTED, va="top")

plt.savefig("gsk_funnel_executive_v3.png", format="png", dpi=DPI)
plt.show()
