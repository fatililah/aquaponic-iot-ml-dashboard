# Prototype Dashboard Aquaponic IoT-ML-MBSE

Dashboard Streamlit ini adalah proof of concept akademik untuk project:

**Prototype Dashboard Sistem Otomasi Aquaponic Berbasis IoT dan Machine Learning untuk Pemantauan, Klasifikasi Risiko, dan Koreksi pH Kualitas Air**

Dashboard ini dibuat untuk konteks TIN2511 ADSPA. Sistem bersifat **ML-assisted automation**, bukan sistem kontrol lapangan penuh. Prediksi machine learning hanya menjadi rekomendasi awal dan tidak boleh langsung mengaktifkan pompa. Keputusan akhir selalu melewati **Safety Rule Controller**.

## Disclaimer Akademik

Dataset yang digunakan adalah `synthetic-realistic v3.1`, bukan data lapangan aquaponic aktual. Dashboard ini belum boleh diklaim sebagai sistem operasional lapangan yang tervalidasi. Implementasi nyata tetap membutuhkan validasi sensor, validasi model, uji pompa, dan data aquaponic aktual.

## Struktur Penting

```text
aquaponic_ml_dataset/
|-- app.py
|-- requirements.txt
|-- data/
|   |-- aquaponic_synthetic_dataset_v31_bogor_realistic_balanced.csv
|   |-- aquaponic_train_v31.csv
|   `-- aquaponic_test_v31.csv
|-- models/
|   |-- water_quality_classifier_v31.joblib
|   |-- dosing_action_classifier_v31.joblib
|   `-- feature_columns_v31.json
|-- reports/
|   |-- final_model_report_v31_realistic_balanced.md
|   `-- simulation_decision_v31.md
|-- assets/
|   `-- physical_architecture.png
`-- logs/
    |-- dashboard_decision_log.csv
    `-- automation_cycle_log.csv
```

Folder `logs/` dan file `dashboard_decision_log.csv` akan dibuat otomatis jika belum ada.

## Cara Menjalankan

Instal dependency:

```bash
pip install -r requirements.txt
```

Jalankan dashboard:

```bash
streamlit run app.py
```

Jika Python di Windows menggunakan launcher:

```bash
py -m streamlit run app.py
```

## Menu Dashboard

1. **Overview Sistem**
   Menampilkan ringkasan sistem, alur ML-assisted automation, disclaimer akademik, dan gambar arsitektur jika tersedia.

2. **Input Sensor & Simulasi**
   Menyediakan preset skenario dan input manual sensor untuk menjalankan prediksi model dan safety check.

3. **Hasil Prediksi ML**
   Menampilkan `predicted_water_quality_status`, `predicted_dosing_action`, dan input yang dipakai model.

4. **Safety Rule & Status Pompa**
   Menampilkan `safety_status`, `safety_reason`, `pump_status`, dashboard alert, dan tabel rule pengaman.

5. **Evaluasi Model**
   Menampilkan dataset v3.1, split train/test, performa model, dan distribusi kelas.

6. **Data Log**
   Menampilkan log keputusan dashboard, tombol download CSV, dan dataset explorer sederhana.

7. **Simulated Automation**
   Menampilkan mode digital twin untuk simulasi otomasi end-to-end berbasis sensor virtual dan aktuator virtual.

## Fitur Input Model

Fitur dibaca dari `models/feature_columns_v31.json`:

- `ph`
- `temperature_c`
- `water_level_pct`
- `ammonia_ppm`
- `nitrite_ppm`
- `nitrate_ppm`
- `sensor_status`
- `time_since_last_dosing_min`
- `dosing_cycle_count`
- `confidence_score`

## Model yang Digunakan

- Water Quality Classifier: `models/water_quality_classifier_v31.joblib`
- Dosing Action Classifier: `models/dosing_action_classifier_v31.joblib`

Informasi performa akademik:

| Target | Model | Accuracy | F1 Macro |
|---|---|---:|---:|
| water_quality_status | RandomForestClassifier(max_depth=3) | 0.5540 | 0.5705 |
| dosing_action | DecisionTreeClassifier(max_depth=3) | 0.6850 | 0.5505 |

## Safety Rule Controller

Pompa akan diblokir jika:

- `sensor_status != valid`
- `water_level_pct < 50`
- `confidence_score < 0.70`
- `time_since_last_dosing_min < 10`
- `dosing_cycle_count >= 3`
- `predicted_dosing_action == manual_check`
- `ph < 5.8` atau `ph > 8.2`
- `ammonia_ppm > 1.0`
- `nitrite_ppm > 1.0`

Jika semua safety lolos dan `predicted_dosing_action == no_action`, pompa `off`. Jika semua safety lolos dan rekomendasi adalah acid/base dose, pompa `on`.

## Preset Skenario

Dashboard menyediakan preset:

- `normal_condition`
- `high_ph_condition`
- `low_ph_condition`
- `low_water_level_condition`
- `sensor_error_condition`

Preset tersebut digunakan untuk acceptance test dan demonstrasi alur keputusan sistem.

## Simulated Automation / Digital Twin Mode

Menu **Simulated Automation** menjalankan simulasi end-to-end:

```text
Virtual Sensor Reading
-> ML Prediction
-> Safety Rule Controller
-> Virtual Pump Decision
-> Simulated Dosing
-> Recheck pH
-> Log
```

Mode ini menggunakan `st.session_state` untuk menyimpan:

- `automation_history`
- `current_virtual_sensor_state`
- `latest_automation_result`
- `automation_cycle_count`

Kontrol yang tersedia:

- Start Simulation
- Stop Simulation
- Reset Simulation
- pilihan skenario awal
- interval simulasi 2 sampai 10 detik
- jumlah siklus 5 sampai 50 siklus

Sensor yang ditampilkan adalah **virtual sensor**. Pompa yang ditampilkan adalah **virtual actuator**, bukan koneksi hardware fisik. Ketika safety lolos dan model merekomendasikan acid/base dosing, dashboard menampilkan Virtual Acid Pump atau Virtual Base Pump sebagai ON. Jika safety gagal, pompa virtual menjadi BLOCKED / OFF.

Setiap siklus menyimpan:

- `ph_before`
- `ph_after_simulated`
- `delta_ph`
- `dosing_success`
- status virtual pump
- output ML
- safety status

Log otomasi disimpan ke:

```text
logs/automation_cycle_log.csv
```

Log keputusan biasa tetap disimpan ke:

```text
logs/dashboard_decision_log.csv
```

## Pengembangan Hardware Berikutnya

Integrasi hardware nyata dapat dikembangkan sebagai tahap lanjutan melalui:

- ESP32 atau Arduino sebagai pembaca sensor pH, suhu, dan level air.
- MQTT atau HTTP API untuk mengirim data sensor ke dashboard/server.
- Relay atau driver pompa untuk aktuator acid/base.
- Safety interlock fisik terpisah dari model ML.
- Kalibrasi sensor dan validasi dosing menggunakan data aquaponic aktual.

Tahap hardware tersebut belum termasuk dalam dashboard ini. Dashboard saat ini hanya simulasi akademik dan digital twin proof of concept.

## Catatan

Dashboard ini cocok untuk screenshot PPT 16:9 dan demonstrasi akademik. Untuk deployment nyata, sistem perlu integrasi sensor fisik, validasi aktuator, pengujian fail-safe, kalibrasi data lapangan, dan audit keamanan operasional.

Footer dashboard:

`TIN2511 - Analisis dan Desain Sistem Produksi Agro Industri | Fadlilah Akbar | 2026`
