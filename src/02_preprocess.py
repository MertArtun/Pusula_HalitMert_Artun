#!/usr/bin/env python3
"""
Fiziksel Tıp & Rehabilitasyon Veri Analizi - Veri Temizleme ve Ön İşleme

Bu script Excel verisini okur, temizler ve model için hazır hale getirir.
İki farklı çıktı üretir:
1. clean_minimal.csv: Temel temizlik uygulanmış veri
2. model_ready_minimal.csv: Eksik değer işleme uygulanmış model-ready veri

Kullanım:
    python src/02_preprocess.py --excel-path data/Talent_Academy_Case_DT_2025.xlsx --sheet Sheet1
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer, KNNImputer

# utils modülünü import et
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import (
    parse_tedavi_suresi_to_int,
    parse_sure_to_minutes,
    split_list,
    add_count_columns,
    ensure_directory_exists
)
from common_logging import preprocess_logger as logger


def load_data(excel_path: str, sheet_name: str) -> pd.DataFrame:
    """
    Excel dosyasından veriyi yükler.
    
    Args:
        excel_path (str): Excel dosya yolu
        sheet_name (str): Sheet adı
        
    Returns:
        pd.DataFrame: Yüklenen veri
    """
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        logger.info(f"✅ Veri başarıyla yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")
        return df
    except FileNotFoundError:
        logger.error(f"❌ Hata: Excel dosyası bulunamadı: {excel_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Hata: Excel dosyası okunurken hata oluştu: {e}")
        sys.exit(1)


def apply_data_transformations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Temel veri dönüşümlerini uygular.
    
    Args:
        df (pd.DataFrame): Ham veri
        
    Returns:
        pd.DataFrame: Dönüştürülmüş veri
    """
    df_clean = df.copy()
    
    logger.info("🔄 Veri dönüşümleri uygulanıyor...")
    
    # 1. Hedef değişken dönüşümü
    if 'TedaviSuresi' in df_clean.columns:
        df_clean['TedaviSuresi_num'] = df_clean['TedaviSuresi'].apply(
            parse_tedavi_suresi_to_int
        )
        successful_parse = df_clean['TedaviSuresi_num'].notna().sum()
        print(f"   ✅ TedaviSuresi → TedaviSuresi_num: {successful_parse}/{len(df_clean)} başarılı")
    
    # 2. Uygulama süresi dönüşümü
    if 'UygulamaSuresi' in df_clean.columns:
        df_clean['UygulamaSuresi_dk'] = df_clean['UygulamaSuresi'].apply(
            parse_sure_to_minutes
        )
        successful_parse = df_clean['UygulamaSuresi_dk'].notna().sum()
        print(f"   ✅ UygulamaSuresi → UygulamaSuresi_dk: {successful_parse}/{len(df_clean)} başarılı")
    
    # 3. Çoklu değerli alanlar için sayı sütunları
    multi_value_columns = ['KronikHastalik', 'Alerji', 'Tanilar', 'UygulamaYerleri']
    existing_multi_cols = [col for col in multi_value_columns if col in df_clean.columns]
    
    if existing_multi_cols:
        df_clean = add_count_columns(df_clean, existing_multi_cols)
        print(f"   ✅ Çoklu değerli alan sayıları eklendi: {existing_multi_cols}")
    
    # 4. Yaş sütununu sayısal yap
    if 'Yas' in df_clean.columns:
        df_clean['Yas'] = pd.to_numeric(df_clean['Yas'], errors='coerce')
        print(f"   ✅ Yaş sütunu sayısal formata dönüştürüldü")
    
    return df_clean


def create_clean_minimal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Temizlenmiş minimal veri seti oluşturur.
    Hedefi NaN olan satırları çıkarır.
    
    Args:
        df (pd.DataFrame): Dönüştürülmüş veri
        
    Returns:
        pd.DataFrame: Temizlenmiş minimal veri
    """
    print("🧹 Temizlenmiş minimal veri seti oluşturuluyor...")
    
    df_clean = df.copy()
    
    # Hedef değişkeni NaN olan satırları çıkar
    if 'TedaviSuresi_num' in df_clean.columns:
        initial_count = len(df_clean)
        df_clean = df_clean[df_clean['TedaviSuresi_num'].notna()]
        removed_count = initial_count - len(df_clean)
        print(f"   📊 Hedef değişkeni eksik olan {removed_count} satır çıkarıldı")
        print(f"   📊 Kalan satır sayısı: {len(df_clean)}")
    
    return df_clean


def apply_imputation(df: pd.DataFrame, imputer_type: str = "median") -> pd.DataFrame:
    """
    Eksik değer işleme (imputation) uygular.
    
    Args:
        df (pd.DataFrame): Temizlenmiş veri
        imputer_type (str): "median" veya "knn"
        
    Returns:
        pd.DataFrame: Imputation uygulanmış veri
    """
    print("🔧 Eksik değer işleme uygulanıyor...")
    
    df_imputed = df.copy()
    
    # Sayısal sütunlar için imputation
    numeric_columns = df_imputed.select_dtypes(include=[np.number]).columns
    numeric_columns = [col for col in numeric_columns if col != 'HastaNo']  # ID sütununu hariç tut
    
    if len(numeric_columns) > 0:
        if imputer_type == "median":
            print(f"   🔢 Sayısal sütunlar için median imputation: {list(numeric_columns)}")
            
            for col in numeric_columns:
                missing_count = df_imputed[col].isnull().sum()
                if missing_count > 0:
                    median_value = df_imputed[col].median()
                    df_imputed[col] = df_imputed[col].fillna(median_value)
                    print(f"      - {col}: {missing_count} eksik değer {median_value} ile dolduruldu")
        
        elif imputer_type == "knn":
            print(f"   🔢 Sayısal sütunlar için KNN imputation: {list(numeric_columns)}")
            
            # Eksik değeri olan sütunları kontrol et
            missing_cols = [col for col in numeric_columns if df_imputed[col].isnull().sum() > 0]
            
            if missing_cols:
                # KNNImputer uygula
                knn_imputer = KNNImputer(n_neighbors=5, weights="uniform")
                df_imputed[numeric_columns] = knn_imputer.fit_transform(df_imputed[numeric_columns])
                
                for col in missing_cols:
                    missing_count = df[col].isnull().sum()  # Orijinal eksik sayısı
                    print(f"      - {col}: {missing_count} eksik değer KNN ile dolduruldu")
            else:
                print(f"      - Sayısal sütunlarda eksik değer yok")
    
    # Kategorik sütunlar için "Bilinmiyor" ile doldur
    categorical_columns = [
        'Cinsiyet', 'KanGrubu', 'Uyruk', 'Bolum', 'TedaviAdi'
    ]
    existing_cat_cols = [col for col in categorical_columns if col in df_imputed.columns]
    
    if existing_cat_cols:
        print(f"   📝 Kategorik sütunlar için 'Bilinmiyor' ile doldurma: {existing_cat_cols}")
        
        for col in existing_cat_cols:
            missing_count = df_imputed[col].isnull().sum()
            if missing_count > 0:
                df_imputed[col] = df_imputed[col].fillna('Bilinmiyor')
                print(f"      - {col}: {missing_count} eksik değer 'Bilinmiyor' ile dolduruldu")
    
    # Çoklu değerli sütunlar için boş string ile doldur
    multi_value_columns = ['KronikHastalik', 'Alerji', 'Tanilar', 'UygulamaYerleri']
    existing_multi_cols = [col for col in multi_value_columns if col in df_imputed.columns]
    
    if existing_multi_cols:
        print(f"   📋 Çoklu değerli sütunlar için boş string ile doldurma: {existing_multi_cols}")
        
        for col in existing_multi_cols:
            missing_count = df_imputed[col].isnull().sum()
            if missing_count > 0:
                df_imputed[col] = df_imputed[col].fillna('')
                print(f"      - {col}: {missing_count} eksik değer boş string ile dolduruldu")
    
    # Diğer string sütunları için "Bilinmiyor" ile doldur
    string_columns = df_imputed.select_dtypes(include=['object']).columns
    remaining_string_cols = [col for col in string_columns 
                           if col not in existing_cat_cols + existing_multi_cols + ['TedaviSuresi', 'UygulamaSuresi']]
    
    if remaining_string_cols:
        print(f"   📄 Diğer string sütunlar için 'Bilinmiyor' ile doldurma: {remaining_string_cols}")
        
        for col in remaining_string_cols:
            missing_count = df_imputed[col].isnull().sum()
            if missing_count > 0:
                df_imputed[col] = df_imputed[col].fillna('Bilinmiyor')
                print(f"      - {col}: {missing_count} eksik değer 'Bilinmiyor' ile dolduruldu")
    
    return df_imputed


def validate_processed_data(df_clean: pd.DataFrame, df_model_ready: pd.DataFrame) -> None:
    """
    İşlenmiş verileri doğrular ve rapor verir.
    
    Args:
        df_clean (pd.DataFrame): Temizlenmiş veri
        df_model_ready (pd.DataFrame): Model-ready veri
    """
    print("\n📋 Veri İşleme Doğrulama Raporu:")
    print("=" * 50)
    
    # Temel istatistikler
    print(f"📊 Clean Minimal - Satır: {len(df_clean):,}, Sütun: {len(df_clean.columns)}")
    print(f"📊 Model Ready - Satır: {len(df_model_ready):,}, Sütun: {len(df_model_ready.columns)}")
    
    # Eksik değer kontrolü
    clean_missing = df_clean.isnull().sum().sum()
    model_missing = df_model_ready.isnull().sum().sum()
    
    print(f"\n🔍 Eksik Değer Kontrolü:")
    print(f"   Clean Minimal: {clean_missing:,} eksik değer")
    print(f"   Model Ready: {model_missing:,} eksik değer")
    
    # Hedef değişken kontrolü
    if 'TedaviSuresi_num' in df_clean.columns:
        target_stats_clean = df_clean['TedaviSuresi_num'].describe()
        target_stats_model = df_model_ready['TedaviSuresi_num'].describe()
        
        print(f"\n🎯 Hedef Değişken İstatistikleri:")
        print(f"   Clean Minimal - Min: {target_stats_clean['min']:.1f}, "
              f"Max: {target_stats_clean['max']:.1f}, "
              f"Ortalama: {target_stats_clean['mean']:.2f}")
        print(f"   Model Ready - Min: {target_stats_model['min']:.1f}, "
              f"Max: {target_stats_model['max']:.1f}, "
              f"Ortalama: {target_stats_model['mean']:.2f}")
    
    # Veri tipleri
    print(f"\n📋 Veri Tipleri:")
    for dtype in ['int64', 'float64', 'object']:
        clean_count = len(df_clean.select_dtypes(include=[dtype]).columns)
        model_count = len(df_model_ready.select_dtypes(include=[dtype]).columns)
        print(f"   {dtype}: Clean={clean_count}, Model Ready={model_count}")


def save_processed_data(df_clean: pd.DataFrame, df_model_ready: pd.DataFrame, output_dir: str) -> None:
    """
    İşlenmiş verileri kaydeder.
    
    Args:
        df_clean (pd.DataFrame): Temizlenmiş veri
        df_model_ready (pd.DataFrame): Model-ready veri
        output_dir (str): Çıktı dizini
    """
    ensure_directory_exists(output_dir)
    
    # Clean minimal kaydet
    clean_path = os.path.join(output_dir, 'clean_minimal.csv')
    df_clean.to_csv(clean_path, index=False, encoding='utf-8-sig')
    print(f"✅ Clean minimal veri kaydedildi: {clean_path}")
    
    # Model ready kaydet
    model_ready_path = os.path.join(output_dir, 'model_ready_minimal.csv')
    df_model_ready.to_csv(model_ready_path, index=False, encoding='utf-8-sig')
    print(f"✅ Model-ready veri kaydedildi: {model_ready_path}")
    
    # Sütun bilgileri kaydet
    column_info_path = os.path.join(output_dir, 'column_info.csv')
    column_info = pd.DataFrame({
        'sutun_adi': df_model_ready.columns,
        'veri_tipi': [str(df_model_ready[col].dtype) for col in df_model_ready.columns],
        'eksik_deger_sayisi': [df_model_ready[col].isnull().sum() for col in df_model_ready.columns],
        'benzersiz_deger_sayisi': [df_model_ready[col].nunique() for col in df_model_ready.columns]
    })
    
    column_info.to_csv(column_info_path, index=False, encoding='utf-8-sig')
    print(f"✅ Sütun bilgileri kaydedildi: {column_info_path}")


def main():
    """Ana fonksiyon - komut satırı argümanlarını işler ve veri temizleme yapar."""
    parser = argparse.ArgumentParser(
        description="Fiziksel Tıp & Rehabilitasyon Veri Analizi - Veri Temizleme Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python src/02_preprocess.py --excel-path data/Talent_Academy_Case_DT_2025.xlsx --sheet Sheet1
  python src/02_preprocess.py --excel-path data/veri.xlsx --sheet "Veri Sayfası"

Çıktılar:
  - data/processed/clean_minimal.csv: Temel temizlik uygulanmış veri
  - data/processed/model_ready_minimal.csv: Model için hazır veri (imputation uygulanmış)
  - data/processed/column_info.csv: Sütun bilgileri
        """
    )
    
    parser.add_argument(
        '--excel-path',
        required=True,
        help='Excel dosya yolu (örn: data/Talent_Academy_Case_DT_2025.xlsx)'
    )
    
    parser.add_argument(
        '--sheet',
        default='Sheet1',
        help='Excel sheet adı (varsayılan: Sheet1)'
    )
    
    parser.add_argument(
        '--imputer',
        choices=['median', 'knn'],
        default='median',
        help='Sayısal eksik değer doldurma yöntemi (varsayılan: median)'
    )
    
    args = parser.parse_args()
    
    logger.info("🚀 Fiziksel Tıp & Rehabilitasyon Veri Temizleme Script'i Başlatılıyor...")
    logger.info(f"📁 Excel Dosyası: {args.excel_path}")
    logger.info(f"📄 Sheet: {args.sheet}")
    logger.info(f"🔧 Imputer: {args.imputer}")
    
    # Çıktı dizinini hazırla
    output_dir = 'data/processed'
    ensure_directory_exists(output_dir)
    
    # Veriyi yükle
    df = load_data(args.excel_path, args.sheet)
    
    # Veri dönüşümlerini uygula
    df_transformed = apply_data_transformations(df)
    
    # Clean minimal veri seti oluştur
    df_clean = create_clean_minimal(df_transformed)
    
    # Model-ready veri seti için imputation uygula
    df_model_ready = apply_imputation(df_clean, args.imputer)
    
    # Doğrulama
    validate_processed_data(df_clean, df_model_ready)
    
    # Verileri kaydet
    logger.info("💾 Veriler kaydediliyor...")
    save_processed_data(df_clean, df_model_ready, output_dir)
    
    logger.info("✅ Veri temizleme tamamlandı!")
    logger.info(f"📁 Çıktı dosyaları: {output_dir}/")
    logger.info("   - clean_minimal.csv: Temel temizlik uygulanmış veri")
    logger.info("   - model_ready_minimal.csv: Model için hazır veri")
    logger.info("   - column_info.csv: Sütun bilgileri")


if __name__ == "__main__":
    main()
