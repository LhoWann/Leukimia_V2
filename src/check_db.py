import sqlite3
import optuna

con = sqlite3.connect("optuna_source_tiny.db")
cur = con.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("Tables:", [r[0] for r in cur.fetchall()])

cur.execute("SELECT * FROM trials LIMIT 3")
cols = [d[0] for d in cur.description]
print("Cols:", cols)
con.close()

study = optuna.load_study(
    study_name="leukemia_source_tiny",
    storage="sqlite:///optuna_source_tiny.db",
)
print(f"\nTotal trials: {len(study.trials)}")
done = [t for t in study.trials if t.state.name == "COMPLETE"]
pruned = [t for t in study.trials if t.state.name == "PRUNED"]
print(f"Complete: {len(done)} | Pruned: {len(pruned)}")

print("\nTop 5 completed:")
done_sorted = sorted(done, key=lambda t: t.value, reverse=True)
for t in done_sorted[:5]:
    print(f"  Trial {t.number:3d} | val_f1={t.value:.4f} | {t.params}")

best = study.best_trial
print(f"\nBEST: Trial {best.number} | val_f1={best.value:.4f}")
print("Params:")
for k, v in best.params.items():
    print(f"  {k}: {v}")
