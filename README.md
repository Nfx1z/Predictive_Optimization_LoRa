### **Revised & Corrected Implementation Plan**  

#### **1. Predict Communication Quality (RSSI/PDR/Latency)**  
**Tujuan**: Membangun model yang memprediksi kualitas sinyal di **setiap titik kandidat** antara Titik Awal (S) dan Titik Tujuan (D) menggunakan **konteks rute spesifik**.  

**Input Model (Harus Dihitung untuk Setiap Rute Baru)**:  
- **Koordinat & Elevasi**: `latitude`, `longitude`, `elevation` (dari DEM/Google Earth).  
- **Jarak Dinamis** (Kunci Utama):  
  - `distance_to_start` = Jarak **lurus** dari titik kandidat ke **Titik Awal (S)**.  
  - `distance_to_destination` = Jarak **lurus** dari titik kandidat ke **Titik Tujuan (D)**.  
- **Tutupan Lahan**: Kategori (hutan, kota, dll.) dari Google/Sentinel-2.  
- **Fitur Fisika Tambahan** (Wajib):  
  - `path_loss` = Perhitungan *free-space path loss* berbasis frekuensi LoRa (868 MHz) dan jarak.  
  - `terrain_penalty` = Bobot berdasarkan tutupan lahan (hutan=0.9, kota=0.6, dll.).  

**Output Model**:  
- `RSSI` (dBm), `PDR` (0–1), `Latency` (ms) **di titik kandidat tersebut**.  

**Dataset Latihan**:  
- Data eksperimen (dikumpulkan via pengukuran lapangan):  
  
  | latitude | longitude | elevation | land_cover | distance_to_start | distance_to_destination | RSSI | PDR | Latency |
  |----------|-----------|-----------|------------|-------------------|-------------------------|------|-----|---------|
  | -74.0060 | 40.7128   | 10m       | urban      | 100m              | 900m                    | -85  | 0.85| 200ms   |
  
- **Catatan**: Kolom `distance_to_start` dan `distance_to_destination` **berbeda untuk setiap rute** (misal: jika S/D berubah, nilai jarak dihitung ulang).  

**Model**:  
- **Random Forest/XGBoost**: Prioritaskan karena interpretabilitas dan kemampuan menangani noise spasial.  
- **Neural Network**: Gunakan jika data > 10.000 titik (tambahkan *spatial dropout* untuk mencegah overfitting).  
- **Validasi**: *Spatial k-fold* (bukan random split) untuk memastikan generalisasi ke area baru.  

**Hasil Akhir Tahap Ini**:  
- **Peta Kualitas Spesifik Rute**: Grid 50m×50m antara S dan D, di mana setiap cell berisi prediksi `PDR` (misal: PDR=0.75 di cell [X,Y]).  
- **Kriteria**: Hanya cell dengan `PDR > 0.7` yang layak menjadi kandidat beacon.  

---

#### **2. Optimisasi Penempatan Beacon**  
**Tujuan**: Menentukan **urutan lokasi beacon** yang memaksimalkan PDR end-to-end dan meminimalkan jumlah beacon.  

**Langkah-Langkah**:  
1. **Buat Graph**:  
   - **Node**: Titik-titik dalam grid dengan `PDR > 0.7` (dari hasil prediksi).  
   - **Edge**: Hubungkan node yang berjarak ≤ 150m (batas maksimum LoRa).  

2. **Hitung Bobot (Cost)**:  
   ```python
   # Bobot harus mencerminkan kualitas kumulatif
   cost = 1 / (PDR_sumber * PDR_tujuan)  # Semakin tinggi PDR, semakin rendah cost
   # Atau: cost = 1 / (PDR_sumber * PDR_tujuan * exp(0.05 * path_loss))
   ```  
   - *Contoh*:  
     - Jika PDR di node A=0.8 dan node B=0.9 → `cost = 1/(0.8*0.9) = 1.39`.  
     - Jika PDR di node C=0.5 (tidak layak) → node C **dihapus** dari graph.  

3. **Jalankan A***:  
   - **Sumber**: Titik Awal (S).  
   - **Tujuan**: Titik Tujuan (D).  
   - **Heuristik**: Jarak lurus ke Titik Tujuan (mempercepat pencarian).  
   - **Output**: Jalur dengan total *cost* terendah → urutan lokasi beacon optimal.  

**Hasil Akhir**:  
- **Urutan Beacon**: `S → [Lokasi 1] → [Lokasi 2] → ... → D`  
- **Jumlah Beacon**: Ditentukan oleh panjang jalur (misal: 3 beacon untuk rute 5km).  
- **Metrik Keberhasilan**:  
  - PDR end-to-end = `PDR_Lokasi1 × PDR_Lokasi2 × ...`  
  - Jumlah beacon **minimal** yang memenuhi PDR target (misal: ≥ 0.7).  

---

### **Mengapa Ini Berhasil?**  
1. **`distance_to_start`/`distance_to_destination` BUKAN Input Statis**:  
   - Nilai ini **dihitung ulang untuk setiap rute baru** (misal: jika S/D berubah dari Jakarta→Surabaya ke Bandung→Yogyakarta).  
   - Model mempelajari: *"Pada 200m dari Titik Awal di area kota, PDR=0.8"* → pola ini hanya berlaku untuk rute yang sedang diproses.  

2. **A* Tidak Mencari "Jarak Terpendek"**:  
   - Bobot berbasis `PDR` memastikan jalur dengan kualitas terbaik (bukan yang tercepat).  
   - *Contoh nyata*: A* akan memilih jalur melalui lahan terbuka (PDR=0.9) meski lebih panjang, bukan melalui hutan (PDR=0.3).  

3. **Fisika Propagasi Diintegrasikan**:  
   - Fitur `path_loss` dan `terrain_penalty` memastikan prediksi realistis (misal: jarak 500m di hutan ≠ 500m di lahan terbuka).  

---

### **Contoh Alur Kerja**  
1. Pengguna input:  
   - Titik Awal (S): Jakarta (-6.2088°, 106.8456°)  
   - Titik Tujuan (D): Surabaya (-7.2575°, 112.7521°)  

2. Sistem:  
   - Generate grid 50m×50m antara S dan D.  
   - Hitung `distance_to_start`/`distance_to_destination` untuk setiap cell.  
   - Prediksi `PDR` di tiap cell → buat peta kualitas.  

3. Optimasi:  
   - A* menghasilkan jalur: `S → [Cirebon] → [Semarang] → [Surabaya]` (3 beacon).  
   - PDR end-to-end = 0.8 × 0.75 × 0.85 = **0.51** (51% paket terkirim).  

4. Validasi:  
   - Jika PDR target = 0.6, sistem merekomendasikan tambah 1 beacon di Semarang.  
   - Hasil akhir: `S → [Cirebon] → [Kudus] → [Semarang] → [Surabaya]` (PDR=0.8×0.7×0.75×0.85=**0.36** → tidak memenuhi? Sistem akan mencari solusi baru).  

---

### **Apa yang Harus Dihindari**  
- ❌ Menggunakan `distance_to_start` dari beacon sebelumnya (harus dari Titik Awal).  
- ❌ Menghitung bobot A* hanya berdasarkan RSSI (harus berdasarkan PDR kumulatif).  
- ❌ Membangun heatmap statis (peta harus diregenerasi untuk setiap rute).  

**Poin Kunci untuk Kesuksesan**:  
> *"Model tidak memprediksi lokasi beacon secara langsung. Model memprediksi kualitas sinyal di setiap titik, lalu A* menggunakan prediksi tersebut untuk membangun jalur optimal. Tanpa fitur `distance_to_start`/`distance_to_destination`, model tidak akan pernah memahami konteks rute."*
