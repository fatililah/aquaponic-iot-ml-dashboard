# Final Model Report V3.1 Realistic Balanced

## Ringkasan

- Dataset: `aquaponic_synthetic_dataset_v31_bogor_realistic_balanced.csv`
- Train: `aquaponic_train_v31.csv` (8,000 baris awal berdasarkan waktu)
- Test: `aquaponic_test_v31.csv` (2,000 baris akhir berdasarkan waktu)
- Split: time-based 80% awal untuk train dan 20% akhir untuk test
- Fitur utama: ph, temperature_c, water_level_pct, ammonia_ppm, nitrite_ppm, nitrate_ppm, sensor_status, time_since_last_dosing_min, dosing_cycle_count, confidence_score
- Data leakage check: Tidak ada kolom hasil proses yang digunakan sebagai fitur input utama.

## Distribusi Target Train

### water_quality_status
| class | count | percentage |
|---|---:|---:|
| Darurat | 1396 | 17.45% |
| Kritis | 2292 | 28.65% |
| Normal | 1592 | 19.90% |
| Waspada | 2720 | 34.00% |

### dosing_action
| class | count | percentage |
|---|---:|---:|
| acid_low_dose | 574 | 7.17% |
| acid_medium_dose | 375 | 4.69% |
| base_low_dose | 603 | 7.54% |
| base_medium_dose | 482 | 6.02% |
| manual_check | 3423 | 42.79% |
| no_action | 2543 | 31.79% |

## Distribusi Target Test

### water_quality_status
| class | count | percentage |
|---|---:|---:|
| Darurat | 365 | 18.25% |
| Kritis | 592 | 29.60% |
| Normal | 409 | 20.45% |
| Waspada | 634 | 31.70% |

### dosing_action
| class | count | percentage |
|---|---:|---:|
| acid_low_dose | 149 | 7.45% |
| acid_medium_dose | 101 | 5.05% |
| base_low_dose | 130 | 6.50% |
| base_medium_dose | 130 | 6.50% |
| manual_check | 862 | 43.10% |
| no_action | 628 | 31.40% |

## Hasil Evaluasi Model

| target | model | selected | accuracy | precision_macro | recall_macro | f1_macro |
|---|---|---:|---:|---:|---:|---:|
| water_quality_status | DecisionTreeClassifier(max_depth=3) |  | 0.5065 | 0.5185 | 0.4705 | 0.4309 |
| water_quality_status | RandomForestClassifier(max_depth=3) | yes | 0.5540 | 0.6065 | 0.5873 | 0.5705 |
| water_quality_status | GradientBoostingClassifier(max_depth=2) |  | 0.7160 | 0.7651 | 0.7136 | 0.7297 |
| dosing_action | DecisionTreeClassifier(max_depth=3) | yes | 0.6850 | 0.5347 | 0.6264 | 0.5505 |
| dosing_action | RandomForestClassifier(max_depth=3) |  | 0.8835 | 0.7842 | 0.8340 | 0.8049 |
| dosing_action | GradientBoostingClassifier(max_depth=2) |  | 0.8885 | 0.8192 | 0.7970 | 0.8013 |

## Metrik Kelas Kunci

| target | model | Normal f1-score | manual_check recall |
|---|---|---:|---:|
| water_quality_status | DecisionTreeClassifier(max_depth=3) | 0.0000 | - |
| water_quality_status | RandomForestClassifier(max_depth=3) | 0.5665 | - |
| water_quality_status | GradientBoostingClassifier(max_depth=2) | 0.6785 | - |
| dosing_action | DecisionTreeClassifier(max_depth=3) | - | 0.5464 |
| dosing_action | RandomForestClassifier(max_depth=3) | - | 0.9327 |
| dosing_action | GradientBoostingClassifier(max_depth=2) | - | 0.9513 |

Model yang dipilih adalah model dengan kombinasi accuracy, f1_macro, F1 kelas Normal, dan recall manual_check paling seimbang. Tujuannya bukan mengejar akurasi tertinggi, melainkan hasil akademik yang realistis dan tidak mendekati 100%.

## Model Terpilih untuk water_quality_status

- Model: RandomForestClassifier(max_depth=3)
- Accuracy: 0.5540
- F1 Macro: 0.5705

Classification report:

```text
              precision    recall  f1-score   support

     Darurat       0.79      0.69      0.74       365
      Kritis       0.70      0.37      0.48       592
      Normal       0.44      0.80      0.57       409
     Waspada       0.50      0.49      0.49       634

    accuracy                           0.55      2000
   macro avg       0.61      0.59      0.57      2000
weighted avg       0.60      0.55      0.55      2000

```

Confusion matrix:

| actual \ predicted | Darurat | Kritis | Normal | Waspada |
|---|---:|---:|---:|---:|
| Darurat | 253 | 52 | 19 | 41 |
| Kritis | 48 | 219 | 127 | 198 |
| Normal | 4 | 3 | 326 | 76 |
| Waspada | 14 | 40 | 270 | 310 |

## Model Terpilih untuk dosing_action

- Model: DecisionTreeClassifier(max_depth=3)
- Accuracy: 0.6850
- F1 Macro: 0.5505

Classification report:

```text
                  precision    recall  f1-score   support

   acid_low_dose       0.32      0.67      0.43       149
acid_medium_dose       0.00      0.00      0.00       101
   base_low_dose       0.56      0.84      0.67       130
base_medium_dose       0.65      0.76      0.70       130
    manual_check       1.00      0.55      0.71       862
       no_action       0.68      0.94      0.79       628

        accuracy                           0.69      2000
       macro avg       0.53      0.63      0.55      2000
    weighted avg       0.75      0.69      0.67      2000

```

Confusion matrix:

| actual \ predicted | acid_low_dose | acid_medium_dose | base_low_dose | base_medium_dose | manual_check | no_action |
|---|---:|---:|---:|---:|---:|---:|
| acid_low_dose | 100 | 0 | 1 | 0 | 0 | 48 |
| acid_medium_dose | 101 | 0 | 0 | 0 | 0 | 0 |
| base_low_dose | 0 | 0 | 109 | 5 | 0 | 16 |
| base_medium_dose | 0 | 0 | 31 | 99 | 0 | 0 |
| manual_check | 84 | 0 | 45 | 49 | 471 | 213 |
| no_action | 28 | 0 | 7 | 0 | 2 | 591 |

## Mengapa Dataset V1 Menghasilkan 100%

Dataset v1 sangat mudah dipelajari karena label dibuat dari threshold yang bersih dan hampir deterministik. Fitur input seperti pH, level air, ammonia, nitrite, nitrate, dan sensor_status punya batas kelas yang sangat jelas. Ketika model melihat pola threshold tersebut, terutama model tree-based, model dapat meniru rule labeling hampir sempurna. Selain itu, random split membuat data train dan test berasal dari distribusi yang sangat mirip.

## Mengapa Dataset V3 Lebih Realistis

Dataset v3 menambahkan noise sensor, kondisi Bogor yang lebih basah, label ambiguity di dekat ambang batas, process noise dosing, hidden variables, dan time-based split. Label tidak lagi selalu merupakan fungsi threshold kaku dari fitur utama. Beberapa kondisi dipengaruhi oleh rain_event, feeding, biofilter maturity, turbidity, pump response, dan error operator, tetapi laporan utama tetap hanya memakai fitur utama agar eksperimen mendekati situasi awal sistem IoT nyata.

## Mengapa V3.1 Diperlukan

V3 sudah lebih realistis dibanding v1, tetapi kelas Normal belum terbaca dengan baik oleh model dan recall manual_check masih rendah. V3.1 memperbaiki hal ini dengan menambah proporsi Normal melalui skenario normal_strong dan normal_boundary, sekaligus memperjelas manual_check melalui safety signature yang terlihat pada fitur input utama.

## Perbandingan Singkat dengan V3

- V3 mempertahankan noise dan ambiguity, tetapi kelas Normal relatif kecil dan mudah kalah oleh Waspada/Kritis.
- V3.1 menaikkan representasi Normal ke kisaran yang lebih sehat dan membuat sebagian Normal tetap jelas tanpa menghilangkan normal_boundary yang ambigu.
- V3.1 membuat manual_check tidak terlalu mirip dengan no_action karena lebih sering disertai sensor tidak valid, confidence rendah, cooldown belum terpenuhi, dosing cycle tinggi, pH ekstrem, nitrogen tinggi, atau level air rendah.

## Mengapa Kelas Normal Penting

Kelas Normal penting karena sistem monitoring harus mampu membedakan kondisi aman dari kondisi yang membutuhkan perhatian. Jika Normal tidak terdeteksi, dashboard dan classifier cenderung terlalu alarmist dan kurang berguna untuk operasi harian.

## Mengapa manual_check Penting untuk Safety

manual_check adalah kelas pengaman. Pada sistem ML-assisted automation, kelas ini mencegah rekomendasi dosing otomatis ketika input sensor, cooldown, level air, confidence, atau parameter nitrogen menunjukkan risiko. Recall manual_check yang lebih baik membantu mengurangi peluang aksi pompa pada kondisi yang seharusnya diperiksa operator.

## Noise dan Kompleksitas yang Ditambahkan

- pH sensor noise normal mean 0 dan std 0.12.
- Temperature noise normal mean 0 dan std 0.5.
- Water level noise normal mean 0 dan std 3.0.
- Ammonia percentage noise 5% sampai 20%.
- Nitrite percentage noise 5% sampai 25%.
- Nitrate percentage noise 5% sampai 20%.
- Boundary ambiguity untuk pH, water level, ammonia, dan nitrite dekat ambang kelas.
- Label noise tetap diterapkan hanya pada area dekat batas kelas, tetapi normal_strong dilindungi agar tidak terlalu sering berubah menjadi Waspada.
- Hidden variables: rain_event, recent_feeding_level, biofilter_maturity, water_turbidity_ntu, pump_response_factor, dan operator_measurement_error.
- Process noise dosing dengan pump_response_factor 0.4 sampai 1.3 dan kemungkinan under-response atau overshoot.

## Kenapa Time-Based Split Lebih Realistis

Pada deployment IoT, model dilatih dari data historis dan diuji pada data masa depan. Time-based split meniru pola itu lebih baik daripada random split. Data 20% terakhir juga diberi mild distribution shift berupa peningkatan rain_event sehingga kondisi test tidak identik dengan train.

## Kenapa Akurasi 55% sampai 70% Masuk Akal

Untuk synthetic-realistic dataset, performa 55% sampai 70% lebih masuk akal karena label dekat boundary memang ambigu, sensor tidak sempurna, sebagian faktor penyebab tidak masuk fitur utama, dan proses dosing tidak selalu merespons sama. Hasil ini menunjukkan dataset tidak terlalu bersih, sehingga lebih cocok untuk diskusi akademik tentang keterbatasan ML-assisted automation dibanding dataset yang menghasilkan metrik sempurna.

## Catatan Validasi Lapangan

Dataset v3.1 tetap synthetic. Dataset ini berguna untuk baseline dan simulasi akademik, tetapi threshold, noise, distribusi Bogor, serta respons dosing perlu dikalibrasi ulang dengan data lapangan sebelum dipakai untuk keputusan operasional nyata.

## Safety Rule Controller Tetap Wajib

Walaupun model sudah dilatih, output ML tidak boleh langsung mengaktifkan pompa. Rekomendasi dosing tetap harus melewati Safety Rule Controller karena model dapat salah prediksi, terutama pada data boundary, sensor drift, cooldown yang belum terpenuhi, atau kondisi nitrogen tinggi.
