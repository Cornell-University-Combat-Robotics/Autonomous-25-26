import optuna
from corner_testing_main import test_detect_corners 

#Settings

#This number represents the angle difference we assign to corner detection with no orientation output
#In other words: a high number represents high punishment for not identifying an orientation (and low => little punishment)
NO_ORIENTATION_SCORE = 40 # ~45
NUMBER_OF_TRIALS = 50    # ~100
# *** Change settings for dirrectory in corner_testing_main.py

def objective(trial):
    params = {
        # 'threshold': trial.suggest_float('threshold', 18, 20),
        'threshold': 19.0,
        'L_weight': trial.suggest_float('L_weight', 0, 1.0),
        'RG_weight': trial.suggest_float('RG_weight', 0, 1.0),
        'BY_weight': trial.suggest_float('BY_weight', 0, 1.0),
        'area_threshold': trial.suggest_int('area_threshold', 4, 16),
        'DISPLAY_IMAGES': False,
        'NO_ORIENTATION_SCORE': NO_ORIENTATION_SCORE
    }
    return test_detect_corners(**params)  

study = optuna.create_study(direction='minimize') # Optuna maximizes by default
study.optimize(objective, n_trials=NUMBER_OF_TRIALS)
print(study.best_params)