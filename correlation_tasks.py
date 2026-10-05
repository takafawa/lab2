"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations
import pandas as pd
from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput

def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """
    df = pd.read_csv(data.csv_path, sep='\t', na_values='NA')
    women_df = df[df['Gender'] == 'Female']
    men_df = df[df['Gender'] == 'Male']
    women_count = len(women_df)
    men_count = len(men_df)
    target_features = ['FSIQ', 'VIQ', 'PIQ', 'Weight', 'Height']
    women_mri_correlation = {
        feature: float(women_df[feature].corr(women_df['MRI_Count']))
        for feature in target_features
    }
    men_mri_correlation = {
        feature: float(men_df[feature].corr(men_df['MRI_Count']))
        for feature in target_features
    }
    max_abs_corr = -1.0
    strongest_mri_feature = ""
    for feature, corr_val in women_mri_correlation.items():
        if abs(corr_val) > max_abs_corr:
            max_abs_corr = abs(corr_val)
            strongest_mri_feature = feature
    for feature, corr_val in men_mri_correlation.items():
        if abs(corr_val) > max_abs_corr:
            max_abs_corr = abs(corr_val)
            strongest_mri_feature = feature

    return BrainCorrelationSummary(
        men_count=men_count,
        women_count=women_count,
        women_mri_correlation=women_mri_correlation,
        men_mri_correlation=men_mri_correlation,
        strongest_mri_feature=strongest_mri_feature
    )
