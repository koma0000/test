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

C_ROS      = "#1B4F72"   # Глубокий корпоративный синий (Роснефть - Москва)
C_ROS_SPB  = "#2E86C1"   # Средне-синий (Роснефть - Питер)
C_ROS_KZN  = "#5DADE2"   # Голубой (Роснефть - Казань)
C_ROS_TXT  = "#154360"

C_UV       = "#B03A2E"   # Насыщенный терракотово-красный (Уволенные - Москва)
C_UV_SPB   = "#E6B0AA"   # Светло-коралловый (Уволенные - Питер)
C_UV_TXT   = "#922B21"

C_OTH      = "#D5DBDB"   # Спокойный светло-серый (Прочие)
C_OTH_TXT  = "#424949"

C_BG       = "#FFFFFF"
C_CARD     = "#F8F9F9"
C_LINE     = "#E5E8E8"
C_DARK     = "#1C2833"
C_MUTED    = "#566573"

# Только 3 города: Москва (M), Питер (S), Казань (K)
CITY_STYLE = {
    ("ros", "M"): dict(fc=C_ROS,     ec="white",   hatch=None,   tcol="white",   bbox=None),
    ("ros", "S"): dict(fc=C_ROS_SPB, ec="#85C1E9", hatch="///",  tcol="white",   bbox=dict(boxstyle="round,pad=0.14", fc=C_ROS, ec="none", alpha=0.88)),
    ("ros", "K"): dict(fc=C_ROS_KZN, ec="#D6EAF8", hatch="\\\\\\", tcol=C_DARK,  bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.85)),
    ("uv",  "M"): dict(fc=C_UV,      ec="white",   hatch=None,   tcol="white",   bbox=None),
    ("uv",  "S"): dict(fc=C_UV_SPB,  ec=C_UV,      hatch="///",  tcol=C_UV_TXT,  bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.90)),
}

CITY_NAMES = {
    "M": "Москва",
    "S": "Питер",
    "K": "Казань",
}
ORDER = "MSK"

ROWS = [
    dict(
        step="УРОВЕНЬ 1",
        lab="Всего ГСК\n(в СПАРК)",
        tot=426,
        sub="100 %  (Мск: 337 · Питер: 53 · Каз: 36)",
        ros=dict(v=201, pct=47, city=dict(M=150, S=28, K=23)),
        uv=dict(v=25,   pct=11, city=dict(M=19, S=6)),
        oth=dict(v=200, pct=47, city=None),
    ),
    dict(
        step="УРОВЕНЬ 2",
        lab="Действующие или\nликвидированные\nс 01.01.2025",
        tot=343,
        sub="81 % от 426  (Мск: 283 · Питер: 31 · Каз: 29)",
        ros=dict(v=168, pct=49, city=dict(M=132, S=16, K=20)),
        uv=dict(v=18,   pct=10, city=dict(M=16, S=2)),
        oth=dict(v=157, pct=46, city=None),
    ),
    dict(
        step="УРОВЕНЬ 3 · ФИНАЛ",
        lab="Есть в учёте\nООО «Теплоснаб»",
        tot=31,
        sub="9 % от 343  (Мск: 31)",
        ros=dict(v=8,  pct=26, city=dict(M=8)),
        uv=dict(v=1,   pct=11, city=dict(M=1)),
        oth=dict(v=22, pct=71, city=None),
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
        # Показываем цифру, если ширина >= min_seg ИЛИ если это 6 уволенных в Питере
        if v >= min_seg or (key == "uv" and ck == "S" and v == 6):
            fsize = 8.0 if (key == "uv" and ck == "S") else 9.2
            ax.text(x + v / 2, y, f"{v}", ha="center", va="center",
                    fontsize=fsize, fontweight="bold", color=st["tcol"],
                    bbox=st["bbox"], zorder=7)
        x += v
    return x

def draw_top_labels(ax, y, x0, x1, key, grp, is_bottom=False):
    cx = (x0 + x1) / 2
    col = {"ros": C_ROS_TXT, "uv": C_UV_TXT, "oth": C_OTH_TXT}[key]
    name = {"ros": "Сотрудник Роснефть", "uv": "Уволенные сотрудники", "oth": "Прочие"}[key]

    if key == "uv":
        offset_x = 0.4 if is_bottom else 0.0
        offset_y = 0.46 if is_bottom else 0.36
        ax.annotate(
            f"{name}: {grp['v']} ({grp['pct']} %)",
            xy=(cx, y + H / 2 + 0.02),
            xytext=(cx + offset_x, y + H / 2 + offset_y),
            ha="center", va="bottom",
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

    line1 = "Сотрудник Роснефть:  " + "  •  ".join(parts_ros)
    line2 = "Уволенные сотрудники:  " + "  •  ".join(parts_uv)

    ax.text(0, y_pos, line1, fontsize=8.4, color=C_ROS_TXT, va="top", ha="left")
    ax.text(0, y_pos - 0.21, line2, fontsize=8.4, color=C_UV_TXT, va="top", ha="left")

def render_level(ax, y, row, min_seg, tot_x, is_bottom=False):
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
    ax.text(tot_x, y - 0.06, row["sub"], fontsize=8.8, fontweight="bold",
            color=C_MUTED, va="top", ha="left")


# --- СОЗДАНИЕ ФИГУРЫ ---
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=C_BG)

header_line = FancyBboxPatch((0.04, 0.965), 0.92, 0.006, boxstyle="round,pad=0,rounding_size=0.002",
                             fc=C_ROS, ec="none", transform=fig.transFigure)
fig.patches.append(header_line)

fig.text(0.04, 0.948, "Воронка ГСК: структура по председателям и городам",
         fontsize=18.5, fontweight="bold", color=C_DARK, va="top")

fig.text(0.04, 0.910,
         "Всего ГСК в СПАРК: 426 шт. (Москва — 337, Питер — 53, Казань — 36). Сотрудники Роснефти по 3 городам составляют\n"
         "47–49 % на первых двух уровнях и 26 % (8 чел.) в учёте ООО «Теплоснаб» (31 ГСК, все в Москве).",
         fontsize=10.0, color=C_MUTED, va="top", linespacing=1.45)

zoom_card = FancyBboxPatch((0.035, 0.215), 0.93, 0.230, boxstyle="round,pad=0.01,rounding_size=0.015",
                           fc=C_CARD, ec=C_LINE, linewidth=1.2, transform=fig.transFigure, zorder=0)
fig.patches.append(zoom_card)

fig.text(0.05, 0.432, "МАСШТАБ 100 %  ·  ДЕТАЛИЗАЦИЯ УРОВНЯ 3 (31 ГСК = 9 % от 343 действующих)",
         fontsize=8.8, fontweight="bold", color=C_ROS, va="top", zorder=2)

axA = fig.add_axes([0.22, 0.485, 0.73, 0.34], zorder=1)
axB = fig.add_axes([0.22, 0.245, 0.73, 0.140], zorder=1)
axB.set_facecolor(C_CARD)

render_level(axA, 1.75, ROWS[0], min_seg=14, tot_x=426 + 14, is_bottom=False)
render_level(axA, 0.00, ROWS[1], min_seg=14, tot_x=343 + 14, is_bottom=False)
render_level(axB, 0.00, ROWS[2], min_seg=1.5, tot_x=(426 + 14) * K, is_bottom=True)

axA.set_xlim(-4, 585)
axA.set_ylim(-0.62, 2.55)
axB.set_xlim(-4 * K, 585 * K)
axB.set_ylim(-0.60, 1.08)

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
        Patch(fc=C_ROS,     ec="white",   label="Сотрудник Роснефть (Москва)"),
        Patch(fc=C_ROS_SPB, ec="#85C1E9", hatch="///", label="Питер"),
        Patch(fc=C_ROS_KZN, ec="#D6EAF8", hatch="\\\\\\", label="Казань"),
    ],
    loc="upper left", bbox_to_anchor=(0.045, 0.185), ncol=3, frameon=False,
    fontsize=8.6, columnspacing=2.0, handlelength=1.8,
    title="Категория «Сотрудник Роснефть» и города:",
    title_fontsize=8.8, alignment="left"
)

fig.legend(
    handles=[
        Patch(fc=C_UV,     ec="white", label="Уволенные сотрудники (Москва)"),
        Patch(fc=C_UV_SPB, ec=C_UV,    hatch="///", label="Уволенные сотрудники (Питер)"),
        Patch(fc=C_OTH,    ec="white", label="Прочие председатели ГСК"),
    ],
    loc="upper left", bbox_to_anchor=(0.045, 0.128), ncol=3, frameon=False,
    fontsize=8.6, columnspacing=2.0, handlelength=1.8,
    title="Прочие категории:",
    title_fontsize=8.8, alignment="left"
)

fig.text(0.04, 0.052,
         "Примечание: проценты по категории «Сотрудник Роснефть» (47 %, 49 %, 26 %) и «Прочие» (47 %, 46 %, 71 %) рассчитаны от общего числа ГСК уровня (426 / 343 / 31);\n"
         "доля «Уволенных сотрудников» (11 %, 10 %, 11 %) — от общего числа сотрудников Роснефти по таблице (25/226, 18/186, 1/9).",
         fontsize=8.0, color=C_MUTED, va="top", linespacing=1.35)

plt.savefig("gsk_funnel_uv_spb6.png", format="png", dpi=DPI)
plt.show()
