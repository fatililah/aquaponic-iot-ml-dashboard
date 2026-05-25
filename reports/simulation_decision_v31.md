# Simulation Decision V3.1

## Ringkasan Tujuan

Simulasi ini menguji alur end-to-end sistem aquaponic ML-assisted automation: input sensor masuk ke Water Quality Classifier, kemudian Dosing Action Classifier, lalu hasil rekomendasi melewati Safety Rule Controller sebelum menentukan status pompa dan alert dashboard.

## Model yang Digunakan

- Water Quality Classifier: `water_quality_classifier_v31.joblib`
- Dosing Action Classifier: `dosing_action_classifier_v31.joblib`
- Feature metadata: `feature_columns_v31.json`
- Water quality model detail: `RandomForestClassifier(max_depth=3)`
- Dosing action model detail: `DecisionTreeClassifier(max_depth=3)`

## Fitur Input Utama

ph, temperature_c, water_level_pct, ammonia_ppm, nitrite_ppm, nitrate_ppm, sensor_status, time_since_last_dosing_min, dosing_cycle_count, confidence_score

## Tabel Input Tiap Skenario

| scenario_name | ph | temperature_c | water_level_pct | ammonia_ppm | nitrite_ppm | nitrate_ppm | sensor_status | time_since_last_dosing_min | dosing_cycle_count | confidence_score |
|---|---|---|---|---|---|---|---|---|---|---|
| normal_condition | 6.8 | 27 | 85 | 0.1 | 0.05 | 60 | valid | 30 | 0 | 0.9 |
| high_ph_condition | 7.8 | 28 | 85 | 0.2 | 0.1 | 80 | valid | 30 | 0 | 0.85 |
| low_ph_condition | 6.0 | 27 | 82 | 0.2 | 0.1 | 70 | valid | 30 | 0 | 0.85 |
| low_water_level_condition | 7.7 | 28 | 40 | 0.2 | 0.1 | 80 | valid | 30 | 0 | 0.85 |
| sensor_error_condition | 7.8 | 28 | 85 | 0.2 | 0.1 | 80 | error | 30 | 0 | 0.6 |

## Tabel Hasil Prediksi dan Keputusan Sistem

| scenario_name | predicted_water_quality_status | predicted_dosing_action | safety_status | safety_reason | pump_status | dashboard_alert |
|---|---|---|---|---|---|---|
| normal_condition | Normal | no_action | safety_pass | no_action_required | off | Kondisi normal, pompa off |
| high_ph_condition | Normal | acid_low_dose | safety_pass | safety_pass_all_checks | on | pH tinggi ringan, pompa acid low dose aktif |
| low_ph_condition | Kritis | base_medium_dose | safety_pass | safety_pass_all_checks | on | pH rendah sedang, pompa base medium dose aktif |
| low_water_level_condition | Waspada | acid_low_dose | safety_fail | water_level_low | blocked | Level air rendah, pompa diblokir |
| sensor_error_condition | Darurat | manual_check | safety_fail | sensor_not_valid | blocked | Sensor bermasalah, pompa diblokir |

## Catatan Keputusan Sistem

Output machine learning tidak langsung mengaktifkan pompa. Prediksi `dosing_action` hanya menjadi rekomendasi awal. Safety Rule Controller tetap memeriksa sensor_status, water_level_pct, confidence_score, cooldown, dosing_cycle_count, pH ekstrem, ammonia_ppm, nitrite_ppm, dan kondisi manual_check sebelum pompa boleh aktif.

Jika Safety Rule Controller gagal, `safety_status` menjadi `safety_fail` dan `pump_status` menjadi `blocked`. Jika aksi adalah `no_action`, pompa tetap `off`. Pompa hanya `on` ketika rekomendasi dosing adalah acid/base dose dan seluruh pemeriksaan safety lolos.

## Keterbatasan

Simulasi ini masih berbasis model yang dilatih dari synthetic-realistic dataset v3.1. Hasilnya berguna untuk demonstrasi alur sistem dan pengujian awal, tetapi belum boleh dianggap sebagai validasi lapangan. Threshold, akurasi model, dan respons pompa tetap perlu diuji menggunakan data aquaponic nyata.
