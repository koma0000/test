import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

FIG_W, FIG_H, DPI = 12, 7, 300
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": "#cfd8dc", "axes.linewidth": 0.8,
    "figure.dpi": 110, "savefig.dpi": DPI,
})

# Другая палитра для городов (бирюзовый, оранжевый, индиго)
C_MSK, C_SPB, C_KZN = "#00897b", "#f57c00", "#5c6bc0"
C_TXT, C_MUT = "#1a1a1a", "#5a6b78"

ROWS = [  # подпись, Москва, Питер, Казань, итого
    ("Всего ГСК (в СПАРК)", 337, 53, 36, 426),
    ("Действующие или ликвидированные\nв период проверки (с 01.01.2025)", 283, 31, 29, 343),
    ("Есть в учете\nООО «Теплоснаб»", 31, 0, 0, 31),
]
XMAX, H, Y = 426, 0.46, [2, 1, 0]

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

for k, (y, (name, msk, spb, kzn, tot)) in enumerate(zip(Y, ROWS)):
    x0, small = 0, []
    for val, col, dark in ((msk, C_MSK, C_MSK),
                           (spb, C_SPB, "#e65100"),
                           (kzn, C_KZN, "#3949ab")):
        if val == 0:
            continue
        ax.barh(y, val, left=x0, height=H, color=col, alpha=0.8,
                edgecolor="white", linewidth=1.4, zorder=3)
        pct = val / tot * 100
        if val >= 25:  # подпись внутри ячейки
            ax.text(x0 + val / 2, y, f"{val}\n({pct:.0f}%)", ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white", zorder=5)
        else:
            small.append((x0 + val / 2, val, pct, dark))
        x0 += val

    for i, (cx, val, pct, dark) in enumerate(small):  # тонкие - выноской
        ax.annotate(f"{val} ({pct:.0f}%)", xy=(cx, y - H / 2),
                    xytext=(60 + 80 * i, y - 0.52), fontsize=10, color=dark,
                    ha="left", va="center", zorder=6,
                    arrowprops=dict(arrowstyle="-", color=dark, lw=0.8,
                                    shrinkA=0, shrinkB=2))

    sub = ["100 %", f"{tot/ROWS[0][4]*100:.0f}% от 426",
           f"{tot/ROWS[1][4]*100:.0f}% от 343"][k]
    ax.text(XMAX + 14, y + 0.05, f"{tot}", va="bottom", ha="left",
            fontsize=15, fontweight="bold", color=C_TXT)
    ax.text(XMAX + 14, y - 0.13, sub, va="top", ha="left",
            fontsize=9.5, color=C_MUT)

ax.set_yticks(Y)
ax.set_yticklabels([r[0] for r in ROWS], fontsize=11.5, color=C_TXT)
ax.set_xlim(-4, 500)
ax.set_ylim(-1.3, 2.7)
ax.set_xticks([])
for s in ("top", "right", "bottom", "left"):
    ax.spines[s].set_visible(False)
ax.grid(axis="x", ls=":", color="#cfd8dc", lw=0.8, zorder=0)

ax.set_title("Разбивка ГСК по городам",
             fontsize=15, fontweight="bold", color=C_TXT, loc="left", pad=18)

ax.legend(handles=[Patch(fc=C_MSK, alpha=0.8, label="Москва"),
                   Patch(fc=C_SPB, alpha=0.8, label="Питер"),
                   Patch(fc=C_KZN, alpha=0.8, label="Казань")],
          loc="upper center", bbox_to_anchor=(0.45, -0.02),
          ncol=3, frameon=False, fontsize=10.5)

plt.tight_layout()
plt.savefig("gsk_cities_v1.jpg", format="jpg", dpi=DPI)
plt.show()
