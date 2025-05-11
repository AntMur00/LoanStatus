## Автоматизированный выбор и настройка алгоритма классификации
## задачи кредитного скоринга с использованием TPOT и H2O

### Logistic Regression Results

Accuracy: 0.9000
Precision: 0.0155
Recall: 0.0800
F1-Score: 0.0260
ROC_AUC: 0.4833

Confusion Matrix:
 [[1348  127]
 [  23    2]]

Classification Report:
               precision    recall  f1-score   support

           0       0.98      0.91      0.95      1475
           1       0.02      0.08      0.03        25

    accuracy                           0.90      1500
   macro avg       0.50      0.50      0.49      1500
weighted avg       0.97      0.90      0.93      1500


### TPOT Results

1 tpot model: generations=5, pop_size=50
Validation metrics:
Accuracy: 0.6283
Precision: 0.5748
Recall: 0.9947
F1-Score: 0.7286
ROC_AUC: 0.6291

Confusion Matrix:
 [[ 344  981]
 [   7 1326]]

Classification Report:
               precision    recall  f1-score   support

           0       0.98      0.26      0.41      1325
           1       0.57      0.99      0.73      1333

    accuracy                           0.63      2658
   macro avg       0.78      0.63      0.57      2658
weighted avg       0.78      0.63      0.57      2658


Best pipeline steps:
1. MaxAbsScaler()
2. Passthrough()
3. FeatureUnion(transformer_list=[('featureunion',
                                FeatureUnion(transformer_list=[('binarizer',
                                                                Binarizer(threshold=0.0126479301056)),
                                                               ('columnonehotencoder',
                                                                ColumnOneHotEncoder())])),
                               ('passthrough', Passthrough())])
4. FeatureUnion(transformer_list=[('skiptransformer', SkipTransformer()),
                               ('passthrough', Passthrough())])
5. GaussianNB()
Test metrics:
Accuracy: 0.2667
Precision: 0.0188
Recall: 0.8400
F1-Score: 0.0368
ROC_AUC: 0.5504

Confusion Matrix:
 [[ 379 1096]
 [   4   21]]

Classification Report:
               precision    recall  f1-score   support

           0       0.99      0.26      0.41      1475
           1       0.02      0.84      0.04        25

    accuracy                           0.27      1500
   macro avg       0.50      0.55      0.22      1500
weighted avg       0.97      0.27      0.40      1500


-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

2 tpot model: generations=5, pop_size=100
Validation metrics:
Accuracy: 0.8811
Precision: 0.8084
Recall: 1.0000
F1-Score: 0.8940
ROC_AUC: 0.9958

Confusion Matrix:
 [[1009  316]
 [   0 1333]]

Classification Report:
               precision    recall  f1-score   support

           0       1.00      0.76      0.86      1325
           1       0.81      1.00      0.89      1333

    accuracy                           0.88      2658
   macro avg       0.90      0.88      0.88      2658
weighted avg       0.90      0.88      0.88      2658


Best pipeline steps:
1. RobustScaler(quantile_range=(0.2475690452454, 0.9671736154307))
2. VarianceThreshold(threshold=0.1022792220919)
3. FeatureUnion(transformer_list=[('skiptransformer', SkipTransformer()),
                               ('passthrough', Passthrough())])
4. FeatureUnion(transformer_list=[('skiptransformer', SkipTransformer()),
                               ('passthrough', Passthrough())])
5. KNeighborsClassifier(n_jobs=1, n_neighbors=75, p=1, weights='distance')
Test metrics:
Accuracy: 0.7580
Precision: 0.0171
Recall: 0.2400
F1-Score: 0.0320
ROC_AUC: 0.5138

Confusion Matrix:
 [[1131  344]
 [  19    6]]

Classification Report:
               precision    recall  f1-score   support

           0       0.98      0.77      0.86      1475
           1       0.02      0.24      0.03        25

    accuracy                           0.76      1500
   macro avg       0.50      0.50      0.45      1500
weighted avg       0.97      0.76      0.85      1500


-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

3 tpot model: generations=10, pop_size=50
Validation metrics:
Accuracy: 0.6283
Precision: 0.5748
Recall: 0.9947
F1-Score: 0.7286
ROC_AUC: 0.6287

Confusion Matrix:
 [[ 344  981]
 [   7 1326]]

Classification Report:
               precision    recall  f1-score   support

           0       0.98      0.26      0.41      1325
           1       0.57      0.99      0.73      1333

    accuracy                           0.63      2658
   macro avg       0.78      0.63      0.57      2658
weighted avg       0.78      0.63      0.57      2658


Best pipeline steps:
1. StandardScaler()
2. VarianceThreshold(threshold=0.0020260724316)
3. FeatureUnion(transformer_list=[('featureunion',
                                FeatureUnion(transformer_list=[('binarizer',
                                                                Binarizer(threshold=0.0126479301056)),
                                                               ('columnonehotencoder',
                                                                ColumnOneHotEncoder())])),
                               ('passthrough', Passthrough())])
4. FeatureUnion(transformer_list=[('skiptransformer', SkipTransformer()),
                               ('passthrough', Passthrough())])
5. GaussianNB()
Test metrics:
Accuracy: 0.2653
Precision: 0.0188
Recall: 0.8400
F1-Score: 0.0367
ROC_AUC: 0.5503

Confusion Matrix:
 [[ 377 1098]
 [   4   21]]

Classification Report:
               precision    recall  f1-score   support

           0       0.99      0.26      0.41      1475
           1       0.02      0.84      0.04        25

    accuracy                           0.27      1500
   macro avg       0.50      0.55      0.22      1500
weighted avg       0.97      0.27      0.40      1500


-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

4 tpot model: generations=10, pop_size=100
Validation metrics:
Accuracy: 0.8811
Precision: 0.8084
Recall: 1.0000
F1-Score: 0.8940
ROC_AUC: 0.9958

Confusion Matrix:
 [[1009  316]
 [   0 1333]]

Classification Report:
               precision    recall  f1-score   support

           0       1.00      0.76      0.86      1325
           1       0.81      1.00      0.89      1333

    accuracy                           0.88      2658
   macro avg       0.90      0.88      0.88      2658
weighted avg       0.90      0.88      0.88      2658


Best pipeline steps:
1. RobustScaler(quantile_range=(0.2475690452454, 0.9671736154307))
2. VarianceThreshold(threshold=0.1022792220919)
3. FeatureUnion(transformer_list=[('skiptransformer', SkipTransformer()),
                               ('passthrough', Passthrough())])
4. FeatureUnion(transformer_list=[('skiptransformer', SkipTransformer()),
                               ('passthrough', Passthrough())])
5. KNeighborsClassifier(n_jobs=1, n_neighbors=75, p=1, weights='distance')
Test metrics:
Accuracy: 0.7580
Precision: 0.0171
Recall: 0.2400
F1-Score: 0.0320
ROC_AUC: 0.5138

Confusion Matrix:
 [[1131  344]
 [  19    6]]

Classification Report:
               precision    recall  f1-score   support

           0       0.98      0.77      0.86      1475
           1       0.02      0.24      0.03        25

    accuracy                           0.76      1500
   macro avg       0.50      0.50      0.45      1500
weighted avg       0.97      0.76      0.85      1500

## H2O Results

Model 1: gbm:GBM_grid_1_AutoML_1_20250508_25958_model_4

Model parameters:
{'model_id': {'default': None, 'actual': {'__meta': {'schema_version': 3, 'schema_name': 'ModelKeyV3', 'schema_type': 'Key<Model>'}, 'name': 'GBM_grid_1_AutoML_1_20250508_25958_model_4', 'type': 'Key<Model>', 'URL': '/3/Models/GBM_grid_1_AutoML_1_20250508_25958_model_4'}, 'input': None}, 'training_frame': {'default': None, 'actual': {'__meta': {'schema_version': 3, 'schema_name': 'FrameKeyV3', 'schema_type': 'Key<Frame>'}, 'name': 'AutoML_1_20250508_25958_training_Key_Frame__upload_a2a47f77cd9fe81cb60d781ea4074dc0.hex', 'type': 'Key<Frame>', 'URL': '/3/Frames/AutoML_1_20250508_25958_training_Key_Frame__upload_a2a47f77cd9fe81cb60d781ea4074dc0.hex'}, 'input': {'__meta': {'schema_version': 3, 'schema_name': 'FrameKeyV3', 'schema_type': 'Key<Frame>'}, 'name': 'AutoML_1_20250508_25958_training_Key_Frame__upload_a2a47f77cd9fe81cb60d781ea4074dc0.hex', 'type': 'Key<Frame>', 'URL': '/3/Frames/AutoML_1_20250508_25958_training_Key_Frame__upload_a2a47f77cd9fe81cb60d781ea4074dc0.hex'}}, 'validation_frame': {'default': None, 'actual': {'__meta': {'schema_version': 3, 'schema_name': 'FrameKeyV3', 'schema_type': 'Key<Frame>'}, 'name': 'Key_Frame__upload_a8718bf506dca95a475872883784b19.hex', 'type': 'Key<Frame>', 'URL': '/3/Frames/Key_Frame__upload_a8718bf506dca95a475872883784b19.hex'}, 'input': {'__meta': {'schema_version': 3, 'schema_name': 'FrameKeyV3', 'schema_type': 'Key<Frame>'}, 'name': 'Key_Frame__upload_a8718bf506dca95a475872883784b19.hex', 'type': 'Key<Frame>', 'URL': '/3/Frames/Key_Frame__upload_a8718bf506dca95a475872883784b19.hex'}}, 'nfolds': {'default': 0, 'actual': 5, 'input': 5}, 'keep_cross_validation_models': {'default': True, 'actual': False, 'input': False}, 'keep_cross_validation_predictions': {'default': False, 'actual': True, 'input': True}, 'keep_cross_validation_fold_assignment': {'default': False, 'actual': False, 'input': False}, 'score_each_iteration': {'default': False, 'actual': False, 'input': False}, 'score_tree_interval': {'default': 0, 'actual': 5, 'input': 5}, 'fold_assignment': {'default': 'AUTO', 'actual': 'Modulo', 'input': 'Modulo'}, 'fold_column': {'default': None, 'actual': None, 'input': None}, 'response_column': {'default': None, 'actual': {'__meta': {'schema_version': 3, 'schema_name': 'ColSpecifierV3', 'schema_type': 'VecSpecifier'}, 'column_name': 'flag', 'is_member_of_frames': None}, 'input': {'__meta': {'schema_version': 3, 'schema_name': 'ColSpecifierV3', 'schema_type': 'VecSpecifier'}, 'column_name': 'flag', 'is_member_of_frames': None}}, 'ignored_columns': {'default': None, 'actual': [], 'input': []}, 'ignore_const_cols': {'default': True, 'actual': True, 'input': True}, 'offset_column': {'default': None, 'actual': None, 'input': None}, 'weights_column': {'default': None, 'actual': None, 'input': None}, 'balance_classes': {'default': False, 'actual': False, 'input': False}, 'class_sampling_factors': {'default': None, 'actual': None, 'input': None}, 'max_after_balance_size': {'default': 5.0, 'actual': 5.0, 'input': 5.0}, 'max_confusion_matrix_size': {'default': 20, 'actual': 20, 'input': 20}, 'ntrees': {'default': 50, 'actual': 121, 'input': 10000}, 'max_depth': {'default': 5, 'actual': 17, 'input': 17}, 'min_rows': {'default': 10.0, 'actual': 10.0, 'input': 10.0}, 'nbins': {'default': 20, 'actual': 20, 'input': 20}, 'nbins_top_level': {'default': 1024, 'actual': 1024, 'input': 1024}, 'nbins_cats': {'default': 1024, 'actual': 1024, 'input': 1024}, 'r2_stopping': {'default': 1.7976931348623157e+308, 'actual': 1.7976931348623157e+308, 'input': 1.7976931348623157e+308}, 'stopping_rounds': {'default': 0, 'actual': 0, 'input': 3}, 'stopping_metric': {'default': 'AUTO', 'actual': 'logloss', 'input': 'logloss'}, 'stopping_tolerance': {'default': 0.001, 'actual': 0.01270001270001905, 'input': 0.01270001270001905}, 'max_runtime_secs': {'default': 0.0, 'actual': 0.0, 'input': 0.0}, 'seed': {'default': -1, 'actual': 45, 'input': 45}, 'build_tree_one_node': {'default': False, 'actual': False, 'input': False}, 'learn_rate': {'default': 0.1, 'actual': 0.1, 'input': 0.1}, 'learn_rate_annealing': {'default': 1.0, 'actual': 1.0, 'input': 1.0}, 'distribution': {'default': 'AUTO', 'actual': 'bernoulli', 'input': 'bernoulli'}, 'quantile_alpha': {'default': 0.5, 'actual': 0.5, 'input': 0.5}, 'tweedie_power': {'default': 1.5, 'actual': 1.5, 'input': 1.5}, 'huber_alpha': {'default': 0.9, 'actual': 0.9, 'input': 0.9}, 'checkpoint': {'default': None, 'actual': None, 'input': None}, 'sample_rate': {'default': 1.0, 'actual': 0.7, 'input': 0.7}, 'sample_rate_per_class': {'default': None, 'actual': None, 'input': None}, 'col_sample_rate': {'default': 1.0, 'actual': 0.4, 'input': 0.4}, 'col_sample_rate_change_per_level': {'default': 1.0, 'actual': 1.0, 'input': 1.0}, 'col_sample_rate_per_tree': {'default': 1.0, 'actual': 0.7, 'input': 0.7}, 'min_split_improvement': {'default': 1e-05, 'actual': 0.0001, 'input': 0.0001}, 'histogram_type': {'default': 'AUTO', 'actual': 'UniformAdaptive', 'input': 'AUTO'}, 'max_abs_leafnode_pred': {'default': 1.7976931348623157e+308, 'actual': 1.7976931348623157e+308, 'input': 1.7976931348623157e+308}, 'pred_noise_bandwidth': {'default': 0.0, 'actual': 0.0, 'input': 0.0}, 'categorical_encoding': {'default': 'AUTO', 'actual': 'Enum', 'input': 'AUTO'}, 'calibrate_model': {'default': False, 'actual': False, 'input': False}, 'calibration_frame': {'default': None, 'actual': None, 'input': None}, 'calibration_method': {'default': 'AUTO', 'actual': 'PlattScaling', 'input': 'AUTO'}, 'custom_metric_func': {'default': None, 'actual': None, 'input': None}, 'custom_distribution_func': {'default': None, 'actual': None, 'input': None}, 'export_checkpoints_dir': {'default': None, 'actual': None, 'input': None}, 'in_training_checkpoints_dir': {'default': None, 'actual': None, 'input': None}, 'in_training_checkpoints_tree_interval': {'default': 1, 'actual': 1, 'input': 1}, 'monotone_constraints': {'default': None, 'actual': None, 'input': None}, 'check_constant_response': {'default': True, 'actual': True, 'input': True}, 'gainslift_bins': {'default': -1, 'actual': -1, 'input': -1}, 'auc_type': {'default': 'AUTO', 'actual': 'AUTO', 'input': 'AUTO'}, 'interaction_constraints': {'default': None, 'actual': None, 'input': None}, 'auto_rebalance': {'default': True, 'actual': True, 'input': True}}

Predications:
predict            no          yes
no         0.999936    6.40933e-05
no         0.999676    0.000324404
no         0.97056     0.0294404
no         0.999978    2.16689e-05
no         0.999554    0.000446199
no         0.999803    0.00019703
no         0.999886    0.000113948
yes        0.00837027  0.99163
no         0.999318    0.000682256
no         0.999907    9.27314e-05
[1500 rows x 3 columns]

Confusion Matrix:
[[1465   10]
 [  23    2]]

Classification Report:
              precision    recall  f1-score   support

           0       0.98      0.99      0.99      1475
           1       0.17      0.08      0.11        25

    accuracy                           0.98      1500
   macro avg       0.58      0.54      0.55      1500
weighted avg       0.97      0.98      0.97      1500


-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

Model 2: gbm:GBM_grid_1_AutoML_1_20250508_25958_model_11

Model parameters:
{'model_id': {'default': None, 'actual': {'__meta': {'schema_version': 3, 'schema_name': 'ModelKeyV3', 'schema_type': 'Key<Model>'}, 'name': 'GBM_grid_1_AutoML_1_20250508_25958_model_11', 'type': 'Key<Model>', 'URL': '/3/Models/GBM_grid_1_AutoML_1_20250508_25958_model_11'}, 'input': None}, 'training_frame': {'default': None, 'actual': {'__meta': {'schema_version': 3, 'schema_name': 'FrameKeyV3', 'schema_type': 'Key<Frame>'}, 'name': 'AutoML_1_20250508_25958_training_Key_Frame__upload_a2a47f77cd9fe81cb60d781ea4074dc0.hex', 'type': 'Key<Frame>', 'URL': '/3/Frames/AutoML_1_20250508_25958_training_Key_Frame__upload_a2a47f77cd9fe81cb60d781ea4074dc0.hex'}, 'input': {'__meta': {'schema_version': 3, 'schema_name': 'FrameKeyV3', 'schema_type': 'Key<Frame>'}, 'name': 'AutoML_1_20250508_25958_training_Key_Frame__upload_a2a47f77cd9fe81cb60d781ea4074dc0.hex', 'type': 'Key<Frame>', 'URL': '/3/Frames/AutoML_1_20250508_25958_training_Key_Frame__upload_a2a47f77cd9fe81cb60d781ea4074dc0.hex'}}, 'validation_frame': {'default': None, 'actual': {'__meta': {'schema_version': 3, 'schema_name': 'FrameKeyV3', 'schema_type': 'Key<Frame>'}, 'name': 'Key_Frame__upload_a8718bf506dca95a475872883784b19.hex', 'type': 'Key<Frame>', 'URL': '/3/Frames/Key_Frame__upload_a8718bf506dca95a475872883784b19.hex'}, 'input': {'__meta': {'schema_version': 3, 'schema_name': 'FrameKeyV3', 'schema_type': 'Key<Frame>'}, 'name': 'Key_Frame__upload_a8718bf506dca95a475872883784b19.hex', 'type': 'Key<Frame>', 'URL': '/3/Frames/Key_Frame__upload_a8718bf506dca95a475872883784b19.hex'}}, 'nfolds': {'default': 0, 'actual': 5, 'input': 5}, 'keep_cross_validation_models': {'default': True, 'actual': False, 'input': False}, 'keep_cross_validation_predictions': {'default': False, 'actual': True, 'input': True}, 'keep_cross_validation_fold_assignment': {'default': False, 'actual': False, 'input': False}, 'score_each_iteration': {'default': False, 'actual': False, 'input': False}, 'score_tree_interval': {'default': 0, 'actual': 5, 'input': 5}, 'fold_assignment': {'default': 'AUTO', 'actual': 'Modulo', 'input': 'Modulo'}, 'fold_column': {'default': None, 'actual': None, 'input': None}, 'response_column': {'default': None, 'actual': {'__meta': {'schema_version': 3, 'schema_name': 'ColSpecifierV3', 'schema_type': 'VecSpecifier'}, 'column_name': 'flag', 'is_member_of_frames': None}, 'input': {'__meta': {'schema_version': 3, 'schema_name': 'ColSpecifierV3', 'schema_type': 'VecSpecifier'}, 'column_name': 'flag', 'is_member_of_frames': None}}, 'ignored_columns': {'default': None, 'actual': [], 'input': []}, 'ignore_const_cols': {'default': True, 'actual': True, 'input': True}, 'offset_column': {'default': None, 'actual': None, 'input': None}, 'weights_column': {'default': None, 'actual': None, 'input': None}, 'balance_classes': {'default': False, 'actual': False, 'input': False}, 'class_sampling_factors': {'default': None, 'actual': None, 'input': None}, 'max_after_balance_size': {'default': 5.0, 'actual': 5.0, 'input': 5.0}, 'max_confusion_matrix_size': {'default': 20, 'actual': 20, 'input': 20}, 'ntrees': {'default': 50, 'actual': 121, 'input': 10000}, 'max_depth': {'default': 5, 'actual': 17, 'input': 17}, 'min_rows': {'default': 10.0, 'actual': 10.0, 'input': 10.0}, 'nbins': {'default': 20, 'actual': 20, 'input': 20}, 'nbins_top_level': {'default': 1024, 'actual': 1024, 'input': 1024}, 'nbins_cats': {'default': 1024, 'actual': 1024, 'input': 1024}, 'r2_stopping': {'default': 1.7976931348623157e+308, 'actual': 1.7976931348623157e+308, 'input': 1.7976931348623157e+308}, 'stopping_rounds': {'default': 0, 'actual': 0, 'input': 3}, 'stopping_metric': {'default': 'AUTO', 'actual': 'logloss', 'input': 'logloss'}, 'stopping_tolerance': {'default': 0.001, 'actual': 0.01270001270001905, 'input': 0.01270001270001905}, 'max_runtime_secs': {'default': 0.0, 'actual': 0.0, 'input': 0.0}, 'seed': {'default': -1, 'actual': 52, 'input': 52}, 'build_tree_one_node': {'default': False, 'actual': False, 'input': False}, 'learn_rate': {'default': 0.1, 'actual': 0.1, 'input': 0.1}, 'learn_rate_annealing': {'default': 1.0, 'actual': 1.0, 'input': 1.0}, 'distribution': {'default': 'AUTO', 'actual': 'bernoulli', 'input': 'bernoulli'}, 'quantile_alpha': {'default': 0.5, 'actual': 0.5, 'input': 0.5}, 'tweedie_power': {'default': 1.5, 'actual': 1.5, 'input': 1.5}, 'huber_alpha': {'default': 0.9, 'actual': 0.9, 'input': 0.9}, 'checkpoint': {'default': None, 'actual': None, 'input': None}, 'sample_rate': {'default': 1.0, 'actual': 0.7, 'input': 0.7}, 'sample_rate_per_class': {'default': None, 'actual': None, 'input': None}, 'col_sample_rate': {'default': 1.0, 'actual': 1.0, 'input': 1.0}, 'col_sample_rate_change_per_level': {'default': 1.0, 'actual': 1.0, 'input': 1.0}, 'col_sample_rate_per_tree': {'default': 1.0, 'actual': 0.7, 'input': 0.7}, 'min_split_improvement': {'default': 1e-05, 'actual': 0.0001, 'input': 0.0001}, 'histogram_type': {'default': 'AUTO', 'actual': 'UniformAdaptive', 'input': 'AUTO'}, 'max_abs_leafnode_pred': {'default': 1.7976931348623157e+308, 'actual': 1.7976931348623157e+308, 'input': 1.7976931348623157e+308}, 'pred_noise_bandwidth': {'default': 0.0, 'actual': 0.0, 'input': 0.0}, 'categorical_encoding': {'default': 'AUTO', 'actual': 'Enum', 'input': 'AUTO'}, 'calibrate_model': {'default': False, 'actual': False, 'input': False}, 'calibration_frame': {'default': None, 'actual': None, 'input': None}, 'calibration_method': {'default': 'AUTO', 'actual': 'PlattScaling', 'input': 'AUTO'}, 'custom_metric_func': {'default': None, 'actual': None, 'input': None}, 'custom_distribution_func': {'default': None, 'actual': None, 'input': None}, 'export_checkpoints_dir': {'default': None, 'actual': None, 'input': None}, 'in_training_checkpoints_dir': {'default': None, 'actual': None, 'input': None}, 'in_training_checkpoints_tree_interval': {'default': 1, 'actual': 1, 'input': 1}, 'monotone_constraints': {'default': None, 'actual': None, 'input': None}, 'check_constant_response': {'default': True, 'actual': True, 'input': True}, 'gainslift_bins': {'default': -1, 'actual': -1, 'input': -1}, 'auc_type': {'default': 'AUTO', 'actual': 'AUTO', 'input': 'AUTO'}, 'interaction_constraints': {'default': None, 'actual': None, 'input': None}, 'auto_rebalance': {'default': True, 'actual': True, 'input': True}}

Predications:
predict            no          yes
no         0.999976    2.43828e-05
no         0.99976     0.000240217
no         0.988974    0.0110264
no         0.999991    9.39931e-06
no         0.99946     0.000539735
no         0.999847    0.000153469
no         0.999904    9.58356e-05
yes        0.00144968  0.99855
no         0.999268    0.000732196
no         0.999925    7.50068e-05
[1500 rows x 3 columns]

Confusion Matrix:
[[1462   13]
 [  23    2]]

Classification Report:
              precision    recall  f1-score   support

           0       0.98      0.99      0.99      1475
           1       0.13      0.08      0.10        25

    accuracy                           0.98      1500
   macro avg       0.56      0.54      0.54      1500
weighted avg       0.97      0.98      0.97      1500


-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

Model 3: gbm:GBM_grid_1_AutoML_1_20250508_25958_model_9

Model parameters:
{'model_id': {'default': None, 'actual': {'__meta': {'schema_version': 3, 'schema_name': 'ModelKeyV3', 'schema_type': 'Key<Model>'}, 'name': 'GBM_grid_1_AutoML_1_20250508_25958_model_9', 'type': 'Key<Model>', 'URL': '/3/Models/GBM_grid_1_AutoML_1_20250508_25958_model_9'}, 'input': None}, 'training_frame': {'default': None, 'actual': {'__meta': {'schema_version': 3, 'schema_name': 'FrameKeyV3', 'schema_type': 'Key<Frame>'}, 'name': 'AutoML_1_20250508_25958_training_Key_Frame__upload_a2a47f77cd9fe81cb60d781ea4074dc0.hex', 'type': 'Key<Frame>', 'URL': '/3/Frames/AutoML_1_20250508_25958_training_Key_Frame__upload_a2a47f77cd9fe81cb60d781ea4074dc0.hex'}, 'input': {'__meta': {'schema_version': 3, 'schema_name': 'FrameKeyV3', 'schema_type': 'Key<Frame>'}, 'name': 'AutoML_1_20250508_25958_training_Key_Frame__upload_a2a47f77cd9fe81cb60d781ea4074dc0.hex', 'type': 'Key<Frame>', 'URL': '/3/Frames/AutoML_1_20250508_25958_training_Key_Frame__upload_a2a47f77cd9fe81cb60d781ea4074dc0.hex'}}, 'validation_frame': {'default': None, 'actual': {'__meta': {'schema_version': 3, 'schema_name': 'FrameKeyV3', 'schema_type': 'Key<Frame>'}, 'name': 'Key_Frame__upload_a8718bf506dca95a475872883784b19.hex', 'type': 'Key<Frame>', 'URL': '/3/Frames/Key_Frame__upload_a8718bf506dca95a475872883784b19.hex'}, 'input': {'__meta': {'schema_version': 3, 'schema_name': 'FrameKeyV3', 'schema_type': 'Key<Frame>'}, 'name': 'Key_Frame__upload_a8718bf506dca95a475872883784b19.hex', 'type': 'Key<Frame>', 'URL': '/3/Frames/Key_Frame__upload_a8718bf506dca95a475872883784b19.hex'}}, 'nfolds': {'default': 0, 'actual': 5, 'input': 5}, 'keep_cross_validation_models': {'default': True, 'actual': False, 'input': False}, 'keep_cross_validation_predictions': {'default': False, 'actual': True, 'input': True}, 'keep_cross_validation_fold_assignment': {'default': False, 'actual': False, 'input': False}, 'score_each_iteration': {'default': False, 'actual': False, 'input': False}, 'score_tree_interval': {'default': 0, 'actual': 5, 'input': 5}, 'fold_assignment': {'default': 'AUTO', 'actual': 'Modulo', 'input': 'Modulo'}, 'fold_column': {'default': None, 'actual': None, 'input': None}, 'response_column': {'default': None, 'actual': {'__meta': {'schema_version': 3, 'schema_name': 'ColSpecifierV3', 'schema_type': 'VecSpecifier'}, 'column_name': 'flag', 'is_member_of_frames': None}, 'input': {'__meta': {'schema_version': 3, 'schema_name': 'ColSpecifierV3', 'schema_type': 'VecSpecifier'}, 'column_name': 'flag', 'is_member_of_frames': None}}, 'ignored_columns': {'default': None, 'actual': [], 'input': []}, 'ignore_const_cols': {'default': True, 'actual': True, 'input': True}, 'offset_column': {'default': None, 'actual': None, 'input': None}, 'weights_column': {'default': None, 'actual': None, 'input': None}, 'balance_classes': {'default': False, 'actual': False, 'input': False}, 'class_sampling_factors': {'default': None, 'actual': None, 'input': None}, 'max_after_balance_size': {'default': 5.0, 'actual': 5.0, 'input': 5.0}, 'max_confusion_matrix_size': {'default': 20, 'actual': 20, 'input': 20}, 'ntrees': {'default': 50, 'actual': 125, 'input': 10000}, 'max_depth': {'default': 5, 'actual': 16, 'input': 16}, 'min_rows': {'default': 10.0, 'actual': 10.0, 'input': 10.0}, 'nbins': {'default': 20, 'actual': 20, 'input': 20}, 'nbins_top_level': {'default': 1024, 'actual': 1024, 'input': 1024}, 'nbins_cats': {'default': 1024, 'actual': 1024, 'input': 1024}, 'r2_stopping': {'default': 1.7976931348623157e+308, 'actual': 1.7976931348623157e+308, 'input': 1.7976931348623157e+308}, 'stopping_rounds': {'default': 0, 'actual': 0, 'input': 3}, 'stopping_metric': {'default': 'AUTO', 'actual': 'logloss', 'input': 'logloss'}, 'stopping_tolerance': {'default': 0.001, 'actual': 0.01270001270001905, 'input': 0.01270001270001905}, 'max_runtime_secs': {'default': 0.0, 'actual': 0.0, 'input': 0.0}, 'seed': {'default': -1, 'actual': 50, 'input': 50}, 'build_tree_one_node': {'default': False, 'actual': False, 'input': False}, 'learn_rate': {'default': 0.1, 'actual': 0.1, 'input': 0.1}, 'learn_rate_annealing': {'default': 1.0, 'actual': 1.0, 'input': 1.0}, 'distribution': {'default': 'AUTO', 'actual': 'bernoulli', 'input': 'bernoulli'}, 'quantile_alpha': {'default': 0.5, 'actual': 0.5, 'input': 0.5}, 'tweedie_power': {'default': 1.5, 'actual': 1.5, 'input': 1.5}, 'huber_alpha': {'default': 0.9, 'actual': 0.9, 'input': 0.9}, 'checkpoint': {'default': None, 'actual': None, 'input': None}, 'sample_rate': {'default': 1.0, 'actual': 0.6, 'input': 0.6}, 'sample_rate_per_class': {'default': None, 'actual': None, 'input': None}, 'col_sample_rate': {'default': 1.0, 'actual': 0.7, 'input': 0.7}, 'col_sample_rate_change_per_level': {'default': 1.0, 'actual': 1.0, 'input': 1.0}, 'col_sample_rate_per_tree': {'default': 1.0, 'actual': 0.7, 'input': 0.7}, 'min_split_improvement': {'default': 1e-05, 'actual': 0.0001, 'input': 0.0001}, 'histogram_type': {'default': 'AUTO', 'actual': 'UniformAdaptive', 'input': 'AUTO'}, 'max_abs_leafnode_pred': {'default': 1.7976931348623157e+308, 'actual': 1.7976931348623157e+308, 'input': 1.7976931348623157e+308}, 'pred_noise_bandwidth': {'default': 0.0, 'actual': 0.0, 'input': 0.0}, 'categorical_encoding': {'default': 'AUTO', 'actual': 'Enum', 'input': 'AUTO'}, 'calibrate_model': {'default': False, 'actual': False, 'input': False}, 'calibration_frame': {'default': None, 'actual': None, 'input': None}, 'calibration_method': {'default': 'AUTO', 'actual': 'PlattScaling', 'input': 'AUTO'}, 'custom_metric_func': {'default': None, 'actual': None, 'input': None}, 'custom_distribution_func': {'default': None, 'actual': None, 'input': None}, 'export_checkpoints_dir': {'default': None, 'actual': None, 'input': None}, 'in_training_checkpoints_dir': {'default': None, 'actual': None, 'input': None}, 'in_training_checkpoints_tree_interval': {'default': 1, 'actual': 1, 'input': 1}, 'monotone_constraints': {'default': None, 'actual': None, 'input': None}, 'check_constant_response': {'default': True, 'actual': True, 'input': True}, 'gainslift_bins': {'default': -1, 'actual': -1, 'input': -1}, 'auc_type': {'default': 'AUTO', 'actual': 'AUTO', 'input': 'AUTO'}, 'interaction_constraints': {'default': None, 'actual': None, 'input': None}, 'auto_rebalance': {'default': True, 'actual': True, 'input': True}}

Predications:
predict             no          yes
no         0.999981     1.90475e-05
no         0.999233     0.000767084
no         0.988574     0.0114259
no         0.999994     5.95336e-06
no         0.999356     0.000644167
no         0.999599     0.000400872
no         0.999793     0.000207292
yes        0.000831293  0.999169
no         0.999674     0.000326453
no         0.999921     7.94935e-05
[1500 rows x 3 columns]

Confusion Matrix:
[[1465   10]
 [  23    2]]

Classification Report:
              precision    recall  f1-score   support

           0       0.98      0.99      0.99      1475
           1       0.17      0.08      0.11        25

    accuracy                           0.98      1500
   macro avg       0.58      0.54      0.55      1500
weighted avg       0.97      0.98      0.97      1500

Model Name	Algorithm	Accuracy	Precision	Recall	F1-score	ROC AUC
0	GBM_grid_1_AutoML_1_20250508_25958_model_4	gbm	0.978	0.166667	0.08	0.108108	0.490305
1	GBM_grid_1_AutoML_1_20250508_25958_model_11	gbm	0.976	0.133333	0.08	0.100000	0.481193
2	GBM_grid_1_AutoML_1_20250508_25958_model_9	gbm	0.978	0.166667	0.08	0.108108	0.499444