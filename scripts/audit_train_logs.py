#!/usr/bin/env python3
"""
audit_train_logs.py — Аудит CSV логов обучения на предмет:
  1. Дублированных прогонов (epoch сбрасывается к 1 в середине лога)
  2. best.pt сохранён не от лучшей эпохи (val_loss достигала лучшего значения,
     но чекпоинт не обновился)
  3. Монотонности lr (обрыв/перезапуск LR schedule)
  4. Расхождения val_si_sdr между прогонами

Использование:
  python audit_train_logs.py results/          # все CSV из папки
  python audit_train_logs.py results/dsc20_train_log.csv  # один файл

Колонки ожидаются: experiment, seed, epoch, train_loss, train_si_sdr,
                    val_loss, val_si_sdr, lr
"""

import sys
import os
import glob
import pandas as pd
import numpy as np

ANSI = {
    "red":    "\033[91m",
    "yellow": "\033[93m",
    "green":  "\033[92m",
    "bold":   "\033[1m",
    "reset":  "\033[0m",
}

def c(color, text):
    return f"{ANSI[color]}{text}{ANSI['reset']}"

def detect_runs(df):
    """Разбивает DataFrame на отдельные прогоны по сбросу epoch."""
    runs = []
    run_start = 0
    for i in range(1, len(df)):
        if df["epoch"].iloc[i] <= df["epoch"].iloc[i - 1]:
            runs.append(df.iloc[run_start:i].reset_index(drop=True))
            run_start = i
    runs.append(df.iloc[run_start:].reset_index(drop=True))
    return runs

def audit_file(path):
    print(c("bold", f"\n{'='*60}"))
    print(c("bold", f"  Файл: {os.path.basename(path)}"))
    print(c("bold", f"{'='*60}"))

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(c("red", f"  [ERROR] Не удалось прочитать файл: {e}"))
        return

    required = {"epoch", "val_loss", "val_si_sdr", "lr"}
    missing = required - set(df.columns)
    if missing:
        print(c("yellow", f"  [WARN] Отсутствуют колонки: {missing}. Пропуск."))
        return

    runs = detect_runs(df)

    # --- 1. Количество прогонов ---
    if len(runs) > 1:
        print(c("red", f"  [!] ОБНАРУЖЕНО {len(runs)} ПРОГОНА (epoch сбрасывается {len(runs)-1} раз)"))
        for i, r in enumerate(runs):
            print(f"      Прогон {i+1}: epoch {r['epoch'].min()}–{r['epoch'].max()}  "
                  f"({len(r)} строк)")
    else:
        print(c("green", f"  [OK] Один прогон, epoch {df['epoch'].min()}–{df['epoch'].max()} "
                         f"({len(df)} строк)"))

    # --- 2. Анализ каждого прогона ---
    issues_found = False
    for i, run in enumerate(runs):
        prefix = f"  Прогон {i+1}" if len(runs) > 1 else "  "

        best_idx = run["val_loss"].idxmin()
        best_epoch = run.loc[best_idx, "epoch"]
        best_val_loss = run.loc[best_idx, "val_loss"]
        last_epoch = run["epoch"].max()
        last_val_loss = run.loc[run["epoch"].idxmax(), "val_loss"]
        final_best_epoch = run.loc[run["val_loss"].idxmin(), "epoch"]

        print(f"\n{prefix} Статистика val_loss:")
        print(f"    Лучший: epoch {best_epoch:3d}  val_loss = {best_val_loss:.4f}")
        print(f"    Последний: epoch {last_epoch:3d}  val_loss = {last_val_loss:.4f}")

        # Проверяем: было ли улучшение val_loss ПОСЛЕ best_epoch в этом прогоне?
        after_best = run[run["epoch"] > best_epoch]
        if not after_best.empty:
            min_after = after_best["val_loss"].min()
            if min_after < best_val_loss:
                better_epoch = after_best.loc[after_best["val_loss"].idxmin(), "epoch"]
                print(c("red",
                    f"    [!] ОШИБКА best.pt: epoch {better_epoch} даёт val_loss={min_after:.4f} "
                    f"< best epoch {best_epoch} val_loss={best_val_loss:.4f}"))
                print(c("red",
                    f"        Чекпоинт НЕ обновился на epoch {better_epoch}!"))
                issues_found = True
            else:
                print(c("green", f"    [OK] best.pt корректен — val_loss не улучшался после epoch {best_epoch}"))
        else:
            print(c("green", f"    [OK] best.pt на последней эпохе прогона"))

        # --- val_si_sdr на лучшей эпохе ---
        best_sisdr = run.loc[best_idx, "val_si_sdr"]
        print(f"    val_si_sdr @ best epoch {best_epoch}: {best_sisdr:.3f}")

        # --- LR schedule ---
        lr_vals = run["lr"].values
        if len(lr_vals) > 1:
            diffs = np.diff(lr_vals)
            increases = np.sum(diffs > 1e-10)
            if increases > 0:
                print(c("yellow",
                    f"    [~] LR возрастало {increases} раз (warm-up или cosine restart?)"))
            else:
                print(c("green", f"    [OK] LR монотонно убывает"))

        # --- Переобучение: val_si_sdr ухудшается после best ---
        if not after_best.empty:
            sisdr_at_best = run.loc[best_idx, "val_si_sdr"]
            sisdr_after_min = after_best["val_si_sdr"].min()
            if sisdr_after_min < sisdr_at_best - 2.0:
                print(c("yellow",
                    f"    [~] val_si_sdr падает после best: {sisdr_at_best:.2f} → "
                    f"{sisdr_after_min:.2f} (возможное переобучение)"))

    if not issues_found and len(runs) == 1:
        print(c("green", "\n  [✓] Проблем с чекпоинтом не обнаружено"))
    elif not issues_found and len(runs) > 1:
        print(c("yellow", "\n  [~] Несколько прогонов, но best.pt корректен в каждом"))
    else:
        print(c("red", "\n  [✗] Обнаружены проблемы — см. выше"))

def main():
    if len(sys.argv) < 2:
        print("Использование: python audit_train_logs.py <путь_к_папке_или_csv>")
        sys.exit(1)

    target = sys.argv[1]

    if os.path.isdir(target):
        files = sorted(glob.glob(os.path.join(target, "*train_log*.csv")))
        if not files:
            # Попробуем все CSV
            files = sorted(glob.glob(os.path.join(target, "*.csv")))
        if not files:
            print(c("red", f"CSV файлы не найдены в {target}"))
            sys.exit(1)
        print(c("bold", f"Найдено {len(files)} файлов в {target}:"))
        for f in files:
            print(f"  {os.path.basename(f)}")
    elif os.path.isfile(target):
        files = [target]
    else:
        print(c("red", f"Путь не найден: {target}"))
        sys.exit(1)

    for f in files:
        audit_file(f)

    print(c("bold", f"\n{'='*60}"))
    print(c("bold", f"  Аудит завершён: {len(files)} файл(ов)"))
    print(c("bold", f"{'='*60}\n"))

if __name__ == "__main__":
    main()
