from pandas_tasks import analyze_titanic
from correlation_tasks import analyze_brain_correlations
from grader_contracts.pandas_tasks import TitanicInput
from grader_contracts.correlation_tasks import BrainDataInput

print("Titanic и Pandas")
t_input = TitanicInput(csv_path="titanic.csv")
t_res = analyze_titanic(t_input)
print("Всего строк:", t_res.row_count)
print("Пассажиров старше 30:", t_res.adults_over_30_count)
print("Средний возраст:", t_res.mean_age_by_pclass)
print("Доля выживших:", t_res.survival_rate_by_pclass)
print("Топ 5 билетов:", t_res.highest_fares)
print("\n")
print("Корреляционный анализ")
b_input = BrainDataInput(csv_path="brainsize.txt")
b_res = analyze_brain_correlations(b_input)
print("Количество женщин:", b_res.women_count)
print("Количество мужчин:", b_res.men_count)
print("Корреляции у женщин:", b_res.women_mri_correlation)
print("Корреляции у мужчин:", b_res.men_mri_correlation)
print("Самый сильный признак:", b_res.strongest_mri_feature)