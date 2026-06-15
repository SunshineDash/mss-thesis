"""
format_macs_table.py — форматирование результатов MACs-профилирования
в таблицу для диссертации.

Читает macs_results.csv, добавляет колонки reduction_pct и criterion_5gmacs,
сохраняет macs_table_final.csv и выводит Markdown-таблицу.

Запуск:
    python format_macs_table.py
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "macs_results.csv"
DST = ROOT / "macs_table_final.csv"

# ---------------------------------------------------------------------------
# 1. Чтение
# ---------------------------------------------------------------------------
with open(SRC, newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# ---------------------------------------------------------------------------
# 2. Преобразование и вычисление новых колонок
# ---------------------------------------------------------------------------
baseline = None
for r in rows:
    r["params_M"] = float(r["params_M"])
    r["macs_fvcore_GMACs"] = float(r["macs_fvcore_GMACs"])
    if r["config"] == "L=0":
        baseline = r

assert baseline is not None, "L=0 не найден в CSV"

macs_0 = baseline["macs_fvcore_GMACs"]

for r in rows:
    if r["config"] == "L=0":
        r["reduction_pct"] = "—"
    else:
        pct = (1.0 - r["macs_fvcore_GMACs"] / macs_0) * 100.0
        r["reduction_pct"] = f"{pct:.1f}"

    r["criterion_5gmacs"] = r["macs_fvcore_GMACs"] <= 5.0

# ---------------------------------------------------------------------------
# 3. Сохранение CSV
# ---------------------------------------------------------------------------
fieldnames = [
    "config",
    "params_M",
    "macs_fvcore_GMACs",
    "reduction_pct",
    "criterion_5gmacs",
]

with open(DST, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(
            {
                "config": r["config"],
                "params_M": f"{r['params_M']:.2f}",
                "macs_fvcore_GMACs": f"{r['macs_fvcore_GMACs']:.3f}",
                "reduction_pct": r["reduction_pct"],
                "criterion_5gmacs": r["criterion_5gmacs"],
            }
        )

print(f"Сохранено: {DST}")

# ---------------------------------------------------------------------------
# 4. Вывод Markdown-таблицы
# ---------------------------------------------------------------------------
HEADERS = [
    "Конфигурация",
    "Параметры, M",
    "MACs, GMACs/с",
    "Снижение MACs, %",
    "≤ 5 GMACs",
]

# Название для L=0 — Baseline, для остальных — DSC-L(L)
config_labels = {
    "L=0": "Baseline (L=0)",
    "L=5": "DSC-5 (L=5)",
    "L=10": "DSC-10 (L=10)",
    "L=20": "DSC-20 (L=20)",
    "L=24": "DSC-24 (L=24)",
}

print("\n" + "| " + " | ".join(HEADERS) + " |")
print("|" + "|".join("---" for _ in HEADERS) + "|")

for r in rows:
    label = config_labels[r["config"]]
    params_str = f"{r['params_M']:.2f}"
    macs_str = f"{r['macs_fvcore_GMACs']:.3f}"
    reduction = r["reduction_pct"]
    criterion = str(r["criterion_5gmacs"])
    print(f"| {label} | {params_str} | {macs_str} | {reduction} | {criterion} |")