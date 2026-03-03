# import optuna
# from corner_testing_main import test_detect_corners_black_box 

# def objective(trial):
#     params = {


#         "threshold": trial.suggest_float()
#         "quantization_weights": [L_weight, RG_weight, BY_weight]

#         'color_threshold': trial.suggest_float('color_threshold', 0.0, 1.0),
#         'kernel_size': trial.suggest_int('kernel_size', 3, 50, step=2),
#         'blur_sigma': trial.suggest_float('blur_sigma', 0.1, 5.0),
#     }
#     return test_detect_corners_black_box(**params)  # Optuna maximizes by default

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=100)
# print(study.best_params)