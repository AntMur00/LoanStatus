Автоматизированный выбор и настройка алгоритма классификации
задачи кредитного скоринга с использованием TPOT и H2O

#
# TPOT Results
# generatons=5; population_size=50; X_train=70; y_train=30
#

RMSE: 0.6583
MAE: 0.4333
R2: -0.81
Accuracy: 0.5667
Precision: 0.4667
Recall: 0.5833
F1-Score: 0.5185
ROC_AUC: 0.6435

Confusion Matrix:
 [[10  8]
 [ 5  7]]

Classification Report:
               precision    recall  f1-score   support

           0       0.67      0.56      0.61        18
           1       0.47      0.58      0.52        12

    accuracy                           0.57        30
   macro avg       0.57      0.57      0.56        30
weighted avg       0.59      0.57      0.57        30

#
# TPOT Results
# generatons=5; population_size=50; X_test=100
#

RMSE: 0.6708
MAE: 0.4500
R2: -0.86
Accuracy: 0.5500
Precision: 0.3571
Recall: 0.1220
F1-Score: 0.1818
ROC_AUC: 0.4841

Confusion Matrix:
 [[50  9]
 [36  5]]

Classification Report:
               precision    recall  f1-score   support

           0       0.58      0.85      0.69        59
           1       0.36      0.12      0.18        41

    accuracy                           0.55       100
   macro avg       0.47      0.48      0.44       100
weighted avg       0.49      0.55      0.48       100