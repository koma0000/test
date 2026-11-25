import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

FIG_W, FIG_H, DPI = 12, 8.2, 300
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": "#cfd8dc", "axes.linewidth": 0.8,
    "figure.dpi": 110, "savefig.dpi": DPI,
})

# Палитра председателей
C_ROS, C_UVOL, C_OTH = "#0071bb", "#c62828", "#dfe4e8"
# Палитра городов
C_MSK, C_SPB, C_KZN = "#00897b", "#f57c00", "#5c6bc0"
C_TXT, C_MUT = "#1a1a1a", "#5a6b78"

# Формат: (подпись, [роснефть, уволенные, прочие], [москва, питер, казань], итого)
ROWS = [
    ("Всего ГСК (в СПАРК)", [226, 25, 175], [337, 53, 36], 426),
    ("Действующие или ликвидированные\nв период проверки (с 01.01.2025)", [186, 18, 139], [283, 31, 29], 343),
    ("Есть в учете\nООО «Теплоснаб»", [9, 1, 21], [31, 0, 0], 31),
]

XMAX, H = 426, 0.34
Y = [2, 1, 0]

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

for k, (y, (name, preds, cities, tot)) in enumerate(zip(Y, ROWS)):
    # 1. Верхняя полоса — по председателям
    y_top = y + H / 2 + 0.03
    x0, small_top = 0, []
    for val, col, dark in zip(preds, (C_ROS, C_UVOL, C_OTH), (C_ROS, "#8e1d1d", C_TXT)):
        if val == 0:
            continue
        ax.barh(y_top, val, left=x0, height=H, color=col, alpha=0.75,
                edgecolor="white", linewidth=1.2, zorder=3)
        pct = val / tot * 100
        if val >= 25:
            ax.text(x0 + val / 2, y_top, f"{val} ({pct:.0f}%)", ha="center", va="center",
                    fontsize=8.5, fontweight="bold",
                    color="white" if col != C_OTH else C_TXT, zorder=5)
        else:
            small_top.append((x0 + val / 2, val, pct, dark))
        x0 += val

    for i, (cx, val, pct, dark) in enumerate(small_top):
        ax.annotate(f"{val} ({pct:.0f}%)", xy=(cx, y_top + H / 2),
                    xytext=(cx + 8 + 65 * i, y_top + H / 2 + 0.14), fontsize=9, color=dark,
                    ha="left", va="center", zorder=6,
                    arrowprops=dict(arrowstyle="-", color=dark, lw=0.8, shrinkA=0, shrinkB=2))

    # 2. Нижняя полоса — по городам
    y_bot = y - H / 2 - 0.03
    x0, small_bot = 0, []
    for val, col, dark in zip(cities, (C_MSK, C_SPB, C_KZN), (C_MSK, "#e65100", "#3949ab")):
        if val == 0:
            continue
        ax.barh(y_bot, val, left=x0, height=H, color=col, alpha=0.8,
                edgecolor="white", linewidth=1.2, zorder=3)
        pct = val / tot * 100
        if val >= 25:
            ax.text(x0 + val / 2, y_bot, f"{val} ({pct:.0f}%)", ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color="white", zorder=5)
        else:
            small_bot.append((x0 + val / 2, val, pct, dark))
        x0 += val

    for i, (cx, val, pct, dark) in enumerate(small_bot):
        ax.annotate(f"{val} ({pct:.0f}%)", xy=(cx, y_bot - H / 2),
                    xytext=(cx + 8 + 65 * i, y_bot - H / 2 - 0.14), fontsize=9, color=dark,
                    ha="left", va="center", zorder=6,
                    arrowprops=dict(arrowstyle="-", color=dark, lw=0.8, shrinkA=0, shrinkB=2))

    # Итоги справа по центру пары полос
    sub = ["100 %", f"{tot/ROWS[0][3]*100:.0f}% от 426",
           f"{tot/ROWS[1][3]*100:.0f}% от 343"][k]
    ax.text(XMAX + 14, y + 0.04, f"{tot}", va="bottom", ha="left",
            fontsize=15, fontweight="bold", color=C_TXT)
    ax.text(XMAX + 14, y - 0.08, sub, va="top", ha="left",
            fontsize=9.5, color=C_MUT)

ax.set_yticks(Y)
ax.set_yticklabels([r[0] for r in ROWS], fontsize=11.5, color=C_TXT)
ax.set_xlim(-4, 500)
ax.set_ylim(-0.75, 2.75)
ax.set_xticks([])
for s in ("top", "right", "bottom", "left"):
    ax.spines[s].set_visible(False)
ax.grid(axis="x", ls=":", color="#cfd8dc", lw=0.8, zorder=0)

ax.set_title("Разбивка ГСК по председателям и городам",
             fontsize=15, fontweight="bold", color=C_TXT, loc="left", pad=16)

# Двухрядная легенда: сверху председатели, снизу города
handles = [
    Patch(fc=C_ROS, alpha=0.75, label="сотрудник Роснефть"),
    Patch(fc=C_UVOL, alpha=0.75, label="уволенный сотрудник"),
    Patch(fc=C_OTH, alpha=0.75, label="не указан / прочие"),
    Patch(fc=C_MSK, alpha=0.8, label="Москва"),
    Patch(fc=C_SPB, alpha=0.8, label="Питер"),
    Patch(fc=C_KZN, alpha=0.8, label="Казань"),
]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.01),
          ncol=3, frameon=False, fontsize=10)

plt.tight_layout()
plt.savefig("gsk_combined.jpg", format="jpg", dpi=DPI)
plt.show()
