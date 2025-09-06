# Pusula Fiziksel Tıp & Rehabilitasyon Veri Analizi

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Status](https://img.shields.io/badge/status-ready-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**Geliştirici**: Halit Mert Artun  
**E-posta**: halitmert.artun@example.com

## 📋 Proje Özeti

Bu proje, 2235 satır × 13 sütunluk fiziksel tıp ve rehabilitasyon verisini analiz ederek model-ready özellik matrisi oluşturmayı amaçlamaktadır. Hedef değişken olarak tedavi süresi (seans sayısı) kullanılmaktadır.

### 🎯 Temel Özellikler
- **Kapsamlı EDA**: Eksik değer analizi, görselleştirmeler, kategorik analiz
- **Akıllı Veri Dönüşümleri**: Süreler sayısallaştırıldı, çoklu değerli alanlar işlendi
- **Model-Ready Çıktı**: X_model_ready.csv ve y.csv dosyaları
- **Türkçe Raporlar**: Tüm çıktılar Türkçe etiketlerle
- **Hata Yönetimi**: Kullanıcı dostu hata mesajları

## 🗂️ Veri Yapısı

| Sütun | Açıklama | Tip |
|-------|----------|-----|
| **HastaNo** | Hasta kimlik numarası | ID |
| **Yas** | Hasta yaşı | Sayısal |
| **Cinsiyet** | Hasta cinsiyeti | Kategorik |
| **KanGrubu** | Kan grubu | Kategorik |
| **Uyruk** | Uyruk bilgisi | Kategorik |
| **KronikHastalik** | Kronik hastalıklar | Çoklu değerli |
| **Bolum** | Tedavi bölümü | Kategorik |
| **Alerji** | Alerji bilgileri | Çoklu değerli |
| **Tanilar** | Tanı bilgileri | Çoklu değerli |
| **TedaviAdi** | Tedavi adı | Kategorik |
| **TedaviSuresi** | Tedavi süresi ("15 Seans" formatında) | **HEDEF** |
| **UygulamaYerleri** | Uygulama yerleri | Çoklu değerli |
| **UygulamaSuresi** | Uygulama süresi | Süre |

## 📁 Proje Yapısı

```
Pusula_HalitMert_Artun/
├── data/
│   ├── Talent_Academy_Case_DT_2025.xlsx    # Ham veri
│   └── processed/                          # İşlenmiş veriler
│       ├── clean_minimal.csv
│       ├── model_ready_minimal.csv
│       ├── X_model_ready.csv              # 🎯 Özellik matrisi
│       └── y.csv                          # 🎯 Hedef değişken
├── reports/
│   ├── eda_summary.md                     # EDA özet raporu
│   ├── missing_summary.csv               # Eksik değer analizi
│   ├── categorical_unique_counts.csv      # Kategorik analiz
│   ├── top20_*.csv                        # En sık görülen öğeler
│   └── figures/                           # Görselleştirmeler
│       ├── hist_yas.png
│       ├── hist_tedavi_seans.png
│       ├── hist_uygulama_dk.png
│       └── box_tedavi_by_bolum.png
├── src/
│   ├── utils.py                           # Yardımcı fonksiyonlar
│   ├── 01_eda.py                          # Keşifsel veri analizi
│   ├── 02_preprocess.py                   # Veri temizleme
│   └── 03_build_features.py               # Özellik mühendisliği
├── requirements.txt                       # Python bağımlılıkları
└── README.md                             # Bu dosya
```

## 🚀 Hızlı Başlangıç

### 1. Kurulum
```bash
# Repository'yi klonla
git clone https://github.com/username/Pusula_HalitMert_Artun.git
cd Pusula_HalitMert_Artun

# Sanal ortam oluştur
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Bağımlılıkları yükle
pip install -r requirements.txt
```

### 2. Veri Dosyasını Hazırla
`Talent_Academy_Case_DT_2025.xlsx` dosyasını `data/` klasörüne yerleştirin.

### 3. Analizi Çalıştır

#### Adım 1: Keşifsel Veri Analizi (EDA)
```bash
python src/01_eda.py --excel-path data/Talent_Academy_Case_DT_2025.xlsx --sheet Sheet1
```
**Çıktılar:** `reports/` klasöründe EDA raporları ve görseller

#### Adım 2: Veri Temizleme
```bash
python src/02_preprocess.py --excel-path data/Talent_Academy_Case_DT_2025.xlsx --sheet Sheet1
```
**Çıktılar:** `data/processed/` klasöründe temizlenmiş veriler

#### Adım 3: Özellik Mühendisliği
```bash
python src/03_build_features.py --input-csv data/processed/clean_minimal.csv --top_k 50
```
**Çıktılar:** Model-ready `X_model_ready.csv` ve `y.csv` dosyaları

## 🔧 Veri Dönüşüm Özellikleri

### Temel Dönüşümler
- ✅ **TedaviSuresi** → **TedaviSuresi_num** (hedef değişken)
- ✅ **UygulamaSuresi** → **UygulamaSuresi_dk** (dakika cinsinden)
- ✅ **Çoklu değerli alanlar** → **sayı kolonları** + **top-K binary özellikler**

### İşleme Stratejileri
- ✅ **Sayısal eksik değerler**: Median imputation
- ✅ **Kategorik eksik değerler**: "Bilinmiyor" ile doldurma
- ✅ **One-Hot Encoding**: Tüm kategorik değişkenler
- ✅ **Standardizasyon**: Sayısal özellikler için
- ✅ **Çoklu Etiket**: Binary özellik matrisi (ML_* prefix)

## 📊 Örnek Sonuçlar

### EDA Bulgular
- **En çok eksik değer**: Alerji (%42), Kan Grubu (%30)
- **En sık kronik hastalık**: Aritmi (395 hasta)
- **En sık tanı**: DORSALJİ (695 hasta)
- **En sık uygulama yeri**: Bel (543 hasta)

### Özellik Matrisi
- **Boyut**: 2235 örnek × 200+ özellik
- **Özellik Tipleri**: Sayısal, kategorik (one-hot), çoklu etiket (binary)
- **Hedef**: Tedavi süresi (seans sayısı)

## 📈 Görselleştirmeler

| Grafik | Açıklama |
|--------|----------|
| `hist_yas.png` | Hasta yaş dağılımı |
| `hist_tedavi_seans.png` | Tedavi süresi dağılımı |
| `hist_uygulama_dk.png` | Uygulama süresi dağılımı |
| `box_tedavi_by_bolum.png` | Bölümlere göre tedavi süresi |

## 💡 Kullanım İpuçları

### Farklı Sheet'ler için
```bash
python src/01_eda.py --excel-path data/veri.xlsx --sheet "Veri_Sayfası"
```

### Farklı Top-K değerleri için
```bash
python src/03_build_features.py --input-csv data/processed/clean_minimal.csv --top_k 30
```

### Yardım almak için
```bash
python src/01_eda.py --help
python src/02_preprocess.py --help
python src/03_build_features.py --help
```

## 🔧 Teknik Detaylar

- **Python Sürümü**: 3.8+
- **Ana Kütüphaneler**: pandas, numpy, scikit-learn, matplotlib, seaborn
- **Veri Formatı**: Excel (.xlsx) girdi, CSV çıktı
- **Kod Standardı**: PEP8 uyumlu, docstring'ler ile açıklanmış

## 📝 Önemli Notlar

1. **Veri Dosyası**: `data/Talent_Academy_Case_DT_2025.xlsx` dosyasını manuel eklemeniz gerekmektedir.
2. **Hedef Değişken**: Eksik değerli satırlar model-ready veri setinden çıkarılır.
3. **Tekrarlanabilirlik**: Tüm random işlemler sabit seed kullanır.
4. **Bellek Kullanımı**: Büyük veri setleri için chunk processing önerilir.

## 🆘 Sorun Giderme

| Hata | Çözüm |
|------|-------|
| "Excel dosyası bulunamadı" | `data/` klasörüne doğru dosyayı koyun |
| "Sheet bulunamadı" | Excel dosyasındaki sheet adını kontrol edin |
| "Modül bulunamadı" | `pip install -r requirements.txt` |
| "tabulate eksik" | `pip install tabulate` |

## 📞 İletişim

Herhangi bir sorun veya soru için:
- **E-posta**: halitmert.artun@example.com
- **GitHub Issues**: Bu repository'de issue açabilirsiniz

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

---

*Bu proje Fiziksel Tıp & Rehabilitasyon veri analizi kapsamında geliştirilmiştir.*