import pandas as pd
import os

from emotion_detection_system.conf import main_folder

date_experiment = '280223'
file_name = 'undersampling_results.csv'

data_results = pd.read_csv(
    os.path.join(main_folder, 'emotion_detection_system', 'json_results', f'{date_experiment}',
                 file_name))

################
# MAX (ACC & B_ACC) VALUES IN GENERAL
max_acc_row = data_results.query('Accuracy == Accuracy.max()')
max_b_acc_row = data_results.query('Accuracy_Balanced == Accuracy_Balanced.max()')

################
# MULTIMODAL IN GENERAL (ACC & B_ACC)
best_va_acc_row = data_results.query("Scenario in ('va_late_fusion', 'va_early_fusion')").query(
    'Accuracy == Accuracy.max()')
best_va_balanced_acc_row = data_results.query("Scenario in ('va_late_fusion', 'va_early_fusion')").query(
    'Accuracy_Balanced == Accuracy_Balanced.max()')
avg_va_acc_models = data_results.query("Scenario in ('va_late_fusion', 'va_early_fusion')")[['Accuracy']].mean()
std_va_acc_models = data_results.query("Scenario in ('va_late_fusion', 'va_early_fusion')")[['Accuracy']].std()

avg_va_b_acc_models = data_results.query("Scenario in ('va_late_fusion', 'va_early_fusion')")[
    ['Accuracy_Balanced']].mean()
std_va_b_acc_models = data_results.query("Scenario in ('va_late_fusion', 'va_early_fusion')")[
    ['Accuracy_Balanced']].std()

avg_va_acc_models_parents = \
    data_results.query("Scenario in ('va_late_fusion', 'va_early_fusion') & Annotation_Type == 'parents'")[
        ['Accuracy']].mean()
std_va_acc_models_parents = \
    data_results.query("Scenario in ('va_late_fusion', 'va_early_fusion') & Annotation_Type == 'parents'")[
        ['Accuracy']].std()

avg_va_b_acc_models_parents = \
    data_results.query("Scenario in ('va_late_fusion', 'va_early_fusion') & Annotation_Type == 'parents'")[
        ['Accuracy_Balanced']].mean()
std_va_b_acc_models_parents = \
    data_results.query("Scenario in ('va_late_fusion', 'va_early_fusion') & Annotation_Type == 'parents'")[
        ['Accuracy_Balanced']].std()

avg_va_acc_models_specialist = \
    data_results.query("Scenario in ('va_late_fusion', 'va_early_fusion') & Annotation_Type == 'specialist'")[
        ['Accuracy']].mean()
std_va_acc_models_specialist = \
    data_results.query("Scenario in ('va_late_fusion', 'va_early_fusion') & Annotation_Type == 'specialist'")[
        ['Accuracy']].std()

avg_va_b_acc_models_specialist = \
    data_results.query("Scenario in ('va_late_fusion', 'va_early_fusion') & Annotation_Type == 'specialist'")[
        ['Accuracy_Balanced']].mean()
std_va_b_acc_models_specialist = \
    data_results.query("Scenario in ('va_late_fusion', 'va_early_fusion') & Annotation_Type == 'specialist'")[
        ['Accuracy_Balanced']].std()

################
# MULTIMODAL EARLY VS LATE FUSION (ACC & B_ACC)


################
# UNIMODAL VIDEO VS AUDIO (ACC & B_ACC)


################
# CLASSES CLASSIFICATION IN GENERAL (F1 SCORES)

print(max_acc_row)
print(max_b_acc_row)
