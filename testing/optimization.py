import optuna
from corner_testing_main import test_detect_corners_black_box 

def objective(trial):
    params = {
        'threshold': trial.suggest_float('threshold', 0, 100),
        'L_weight': trial.suggest_float('L_weight', 0.0, 1.0),
        'RG_weight': trial.suggest_float('RG_weight', 0.0, 1.0),
        'BY_weight': trial.suggest_float('BY_weight', 0.0, 1.0),
    }
    return test_detect_corners_black_box(**params)  # Optuna maximizes by default

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
print(study.best_params)