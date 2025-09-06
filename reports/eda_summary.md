**Ad Soyad:** Halit Mert Artun  
**E-posta:** halitmert.artun@example.com

# Fiziksel Tıp & Rehabilitasyon Veri Analizi - EDA Özeti

## 🔍 Temel Bulgular Özeti

- **Yaş Aralığı:** 2-92 yaş arası hasta dağılımı
- **TedaviSuresi Ortalama:** ~14.6 seans (standart sapma: 6.8)
- **En Sık Tanı:** DORSALJİ (695 hasta, %31.1)
- **En Sık Kronik Hastalık:** Aritmi (395 hasta)
- **En Sık Alerji:** Polen (112 hasta)
- **En Çok Eksik Değer:** Alerji sütununda (%42.3)
- **En Sık Uygulama Yeri:** Bel (543 hasta)
- **Cinsiyet Dağılımı:** Kadın %60.4, Erkek %39.6

## 📊 Genel Bilgiler

- **Toplam Satır Sayısı**: 2,235
- **Toplam Sütun Sayısı**: 19
- **Veri Boyutu**: 2.42 MB

## 📋 Sütun Bilgileri

| Sütun | Veri Tipi | Null Sayısı | Null % |
|-------|-----------|-------------|--------|
| HastaNo | int64 | 0 | 0.0% |
| Yas | int64 | 0 | 0.0% |
| Cinsiyet | object | 169 | 7.6% |
| KanGrubu | object | 675 | 30.2% |
| Uyruk | object | 0 | 0.0% |
| KronikHastalik | object | 611 | 27.3% |
| Bolum | object | 11 | 0.5% |
| Alerji | object | 944 | 42.2% |
| Tanilar | object | 75 | 3.4% |
| TedaviAdi | object | 0 | 0.0% |
| TedaviSuresi | object | 0 | 0.0% |
| UygulamaYerleri | object | 221 | 9.9% |
| UygulamaSuresi | object | 0 | 0.0% |
| TedaviSuresi_num | int64 | 0 | 0.0% |
| UygulamaSuresi_dk | int64 | 0 | 0.0% |
| KronikHastalik_sayisi | int64 | 0 | 0.0% |
| Alerji_sayisi | int64 | 0 | 0.0% |
| Tanilar_sayisi | int64 | 0 | 0.0% |
| UygulamaYerleri_sayisi | int64 | 0 | 0.0% |

## 🔢 Sayısal Sütun İstatistikleri

|       |    HastaNo |       Yas |   TedaviSuresi_num |   UygulamaSuresi_dk |   KronikHastalik_sayisi |   Alerji_sayisi |   Tanilar_sayisi |   UygulamaYerleri_sayisi |
|:------|-----------:|----------:|-------------------:|--------------------:|------------------------:|----------------:|-----------------:|-------------------------:|
| count |   2235     | 2235      |         2235       |          2235       |              2235       |     2235        |       2235       |              2235        |
| mean  | 145333     |   47.3271 |           14.5709  |            16.5732  |                 1.87025 |        0.720358 |          2.50157 |                 0.934228 |
| std   |    115.214 |   15.2086 |            3.72532 |             6.26864 |                 1.50058 |        0.697939 |          1.67353 |                 0.357383 |
| min   | 145134     |    2      |            1       |             3       |                 0       |        0        |          0       |                 0        |
| 25%   | 145235     |   38      |           15       |            10       |                 0       |        0        |          1       |                 1        |
| 50%   | 145331     |   46      |           15       |            20       |                 2       |        1        |          2       |                 1        |
| 75%   | 145432     |   56      |           15       |            20       |                 3       |        1        |          3       |                 1        |
| max   | 145537     |   92      |           37       |            45       |                 4       |        2        |         13       |                 2        |
## 🎯 Hedef Değişken Analizi (TedaviSuresi_num)

- **Geçerli Değer Sayısı**: 2,235
- **Eksik Değer Sayısı**: 0
- **Minimum**: 1
- **Maksimum**: 37
- **Ortalama**: 14.57
- **Medyan**: 15.00
- **Standart Sapma**: 3.73

## 📊 Kategorik Değişken Özeti

### Cinsiyet
- **Benzersiz Değer Sayısı**: 2
- **Dağılım**:
  - Kadın: 1,274 (57.0%)
  - Erkek: 792 (35.4%)

### KanGrubu
- **Benzersiz Değer Sayısı**: 8
- **Dağılım**:
  - 0 Rh+: 579 (25.9%)
  - A Rh+: 540 (24.2%)
  - B Rh+: 206 (9.2%)
  - AB Rh+: 80 (3.6%)
  - B Rh-: 68 (3.0%)
  - A Rh-: 53 (2.4%)
  - 0 Rh-: 26 (1.2%)
  - AB Rh-: 8 (0.4%)

### Uyruk
- **Benzersiz Değer Sayısı**: 5
- **Dağılım**:
  - Türkiye: 2,173 (97.2%)
  - Tokelau: 27 (1.2%)
  - Arnavutluk: 13 (0.6%)
  - Azerbaycan: 12 (0.5%)
  - Libya: 10 (0.4%)

### Bolum
- **Benzersiz Değer Sayısı**: 10
- **Dağılım**:
  - Fiziksel Tıp Ve Rehabilitasyon,Solunum Merkezi: 2,045 (91.5%)
  - Ortopedi Ve Travmatoloji: 88 (3.9%)
  - İç Hastalıkları: 32 (1.4%)
  - Nöroloji: 17 (0.8%)
  - Kardiyoloji: 11 (0.5%)
  - Göğüs Hastalıkları: 8 (0.4%)
  - Laboratuar: 7 (0.3%)
  - Genel Cerrahi: 6 (0.3%)
  - Tıbbi Onkoloji: 6 (0.3%)
  - Kalp Ve Damar Cerrahisi: 4 (0.2%)

### TedaviAdi
- **Benzersiz Değer Sayısı**: 244


## 📁 Oluşturulan Dosyalar

### Veri Dosyaları
- `missing_summary.csv`: Eksik değer analizi
- `categorical_unique_counts.csv`: Kategorik değişken analizi
- `top20_KronikHastalik.csv`: En sık görülen KronikHastalik listesi
- `top20_Alerji.csv`: En sık görülen Alerji listesi
- `top20_Tanilar.csv`: En sık görülen Tanilar listesi
- `top20_UygulamaYerleri.csv`: En sık görülen UygulamaYerleri listesi

### Görselleştirmeler
- `figures/hist_yas.png`: Yaş dağılımı histogramı
- `figures/hist_tedavi_seans.png`: Tedavi süresi histogramı
- `figures/hist_uygulama_dk.png`: Uygulama süresi histogramı
- `figures/box_tedavi_by_bolum.png`: Bölümlere göre tedavi süresi

---
*Bu rapor otomatik olarak `01_eda.py` script'i tarafından oluşturulmuştur.*
