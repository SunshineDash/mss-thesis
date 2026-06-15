import pandas as pd, glob, os

for path in sorted(glob.glob("results/*train_log*.csv")):
    df = pd.read_csv(path)
    global_best = df["val_loss"].min()
    global_best_epoch = df.loc[df["val_loss"].idxmin(), "epoch"]
    
    # Последний прогон = строки после последнего сброса epoch
    resets = df[df["epoch"].diff() < 0].index.tolist()
    last_run_start = resets[-1] if resets else 0
    last_run = df.iloc[last_run_start:]
    last_best = last_run["val_loss"].min()
    
    gap = last_best - global_best
    flag = "🔴" if gap > 0.1 else "✅"
    print(f"{flag} {os.path.basename(path):<35} global_best={global_best:.4f} (ep.{global_best_epoch}) | last_run_best={last_best:.4f} | gap={gap:+.4f}")