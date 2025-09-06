#!/usr/bin/env python3
"""
Fiziksel Tıp & Rehabilitasyon Veri Analizi - Keşifsel Veri Analizi (EDA)

Bu script Excel verisini okur, temel dönüşümleri yapar ve 
kapsamlı EDA raporu ile görselleştirmeler üretir.

Kullanım:
    python src/01_eda.py --excel-path data/Talent_Academy_Case_DT_2025.xlsx --sheet Sheet1
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# utils modülünü import et
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import (
    parse_tedavi_suresi_to_int,
    parse_sure_to_minutes,
    split_list,
    add_count_columns,
    top_items_series,
    get_missing_summary,
    get_categorical_summary,
    ensure_directory_exists
)
from common_logging import eda_logger as logger

# Türkçe font ayarları
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Seaborn stil ayarları
sns.set_style("whitegrid")
sns.set_palette("husl")


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


def apply_basic_transformations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Temel veri dönüşümlerini uygular.
    
    Args:
        df (pd.DataFrame): Ham veri
        
    Returns:
        pd.DataFrame: Dönüştürülmüş veri
    """
    df_transformed = df.copy()
    
    # Hedef değişken dönüşümü
    if 'TedaviSuresi' in df_transformed.columns:
        df_transformed['TedaviSuresi_num'] = df_transformed['TedaviSuresi'].apply(
            parse_tedavi_suresi_to_int
        )
        logger.info(f"✅ TedaviSuresi -> TedaviSuresi_num dönüşümü tamamlandı")
        logger.info(f"   Başarılı parse: {df_transformed['TedaviSuresi_num'].notna().sum()} / {len(df_transformed)}")
    
    # Uygulama süresi dönüşümü
    if 'UygulamaSuresi' in df_transformed.columns:
        df_transformed['UygulamaSuresi_dk'] = df_transformed['UygulamaSuresi'].apply(
            parse_sure_to_minutes
        )
        logger.info(f"✅ UygulamaSuresi -> UygulamaSuresi_dk dönüşümü tamamlandı")
        logger.info(f"   Başarılı parse: {df_transformed['UygulamaSuresi_dk'].notna().sum()} / {len(df_transformed)}")
    
    # Çoklu değerli alanlar için sayı sütunları
    multi_value_columns = ['KronikHastalik', 'Alerji', 'Tanilar', 'UygulamaYerleri']
    existing_multi_cols = [col for col in multi_value_columns if col in df_transformed.columns]
    
    if existing_multi_cols:
        df_transformed = add_count_columns(df_transformed, existing_multi_cols)
        logger.info(f"✅ Çoklu değerli alanlar için sayı sütunları eklendi: {existing_multi_cols}")
    
    return df_transformed


def generate_missing_analysis(df: pd.DataFrame, output_dir: str) -> None:
    """
    Eksik değer analizi yapar ve kaydeder.
    
    Args:
        df (pd.DataFrame): Veri çerçevesi
        output_dir (str): Çıktı dizini
    """
    missing_summary = get_missing_summary(df)
    
    # CSV olarak kaydet
    missing_path = os.path.join(output_dir, 'missing_summary.csv')
    missing_summary.to_csv(missing_path, index=False, encoding='utf-8-sig')
    print(f"✅ Eksik değer analizi kaydedildi: {missing_path}")
    
    # Konsol çıktısı
    if len(missing_summary) > 0:
        print("\n📊 Eksik Değer Özeti:")
        print(missing_summary.to_string(index=False))
    else:
        print("✅ Hiç eksik değer yok!")


def generate_categorical_analysis(df: pd.DataFrame, output_dir: str) -> None:
    """
    Kategorik değişken analizi yapar ve kaydeder.
    
    Args:
        df (pd.DataFrame): Veri çerçevesi
        output_dir (str): Çıktı dizini
    """
    categorical_columns = [
        'Cinsiyet', 'KanGrubu', 'Uyruk', 'Bolum', 'TedaviAdi'
    ]
    existing_cat_cols = [col for col in categorical_columns if col in df.columns]
    
    if existing_cat_cols:
        cat_summary = get_categorical_summary(df, existing_cat_cols)
        
        # CSV olarak kaydet
        cat_path = os.path.join(output_dir, 'categorical_unique_counts.csv')
        cat_summary.to_csv(cat_path, index=False, encoding='utf-8-sig')
        print(f"✅ Kategorik analiz kaydedildi: {cat_path}")
        
        # Konsol çıktısı
        print("\n📊 Kategorik Değişken Özeti:")
        print(cat_summary.to_string(index=False))


def generate_top_items_analysis(df: pd.DataFrame, output_dir: str) -> None:
    """
    Çoklu değerli alanlar için top-20 analizi yapar.
    
    Args:
        df (pd.DataFrame): Veri çerçevesi
        output_dir (str): Çıktı dizini
    """
    multi_value_columns = ['KronikHastalik', 'Alerji', 'Tanilar', 'UygulamaYerleri']
    
    for col in multi_value_columns:
        if col in df.columns:
            top_items_df = top_items_series(df[col], top_n=20)
            
            if len(top_items_df) > 0:
                # CSV olarak kaydet
                top_path = os.path.join(output_dir, f'top20_{col}.csv')
                top_items_df.to_csv(top_path, index=False, encoding='utf-8-sig')
                print(f"✅ Top-20 {col} analizi kaydedildi: {top_path}")
                
                # İlk 10'u konsola yazdır
                print(f"\n📊 En Sık Görülen {col} (İlk 10):")
                print(top_items_df.head(10).to_string(index=False))


def create_visualizations(df: pd.DataFrame, output_dir: str) -> None:
    """
    Görselleştirmeler oluşturur ve kaydeder.
    
    Args:
        df (pd.DataFrame): Veri çerçevesi
        output_dir (str): Çıktı dizini
    """
    figures_dir = os.path.join(output_dir, 'figures')
    ensure_directory_exists(figures_dir)
    
    # 1. Yaş histogramı
    if 'Yas' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(df['Yas'].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Hasta Yaş Dağılımı', fontsize=14, fontweight='bold')
        plt.xlabel('Yaş')
        plt.ylabel('Hasta Sayısı')
        plt.grid(True, alpha=0.3)
        
        # İstatistikler ekle
        mean_age = df['Yas'].mean()
        median_age = df['Yas'].median()
        plt.axvline(mean_age, color='red', linestyle='--', label=f'Ortalama: {mean_age:.1f}')
        plt.axvline(median_age, color='green', linestyle='--', label=f'Medyan: {median_age:.1f}')
        plt.legend()
        
        hist_path = os.path.join(figures_dir, 'hist_yas.png')
        plt.savefig(hist_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Yaş histogramı kaydedildi: {hist_path}")
    
    # 2. Tedavi süresi histogramı
    if 'TedaviSuresi_num' in df.columns:
        plt.figure(figsize=(10, 6))
        valid_tedavi = df['TedaviSuresi_num'].dropna()
        
        if len(valid_tedavi) > 0:
            plt.hist(valid_tedavi, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            plt.title('Tedavi Süresi (Seans Sayısı) Dağılımı', fontsize=14, fontweight='bold')
            plt.xlabel('Seans Sayısı')
            plt.ylabel('Hasta Sayısı')
            plt.grid(True, alpha=0.3)
            
            # İstatistikler ekle
            mean_tedavi = valid_tedavi.mean()
            median_tedavi = valid_tedavi.median()
            plt.axvline(mean_tedavi, color='red', linestyle='--', label=f'Ortalama: {mean_tedavi:.1f}')
            plt.axvline(median_tedavi, color='green', linestyle='--', label=f'Medyan: {median_tedavi:.1f}')
            plt.legend()
            
            hist_tedavi_path = os.path.join(figures_dir, 'hist_tedavi_seans.png')
            plt.savefig(hist_tedavi_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Tedavi süresi histogramı kaydedildi: {hist_tedavi_path}")
    
    # 3. Uygulama süresi histogramı
    if 'UygulamaSuresi_dk' in df.columns:
        plt.figure(figsize=(10, 6))
        valid_uygulama = df['UygulamaSuresi_dk'].dropna()
        
        if len(valid_uygulama) > 0:
            plt.hist(valid_uygulama, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            plt.title('Uygulama Süresi (Dakika) Dağılımı', fontsize=14, fontweight='bold')
            plt.xlabel('Süre (Dakika)')
            plt.ylabel('Hasta Sayısı')
            plt.grid(True, alpha=0.3)
            
            # İstatistikler ekle
            mean_uygulama = valid_uygulama.mean()
            median_uygulama = valid_uygulama.median()
            plt.axvline(mean_uygulama, color='red', linestyle='--', label=f'Ortalama: {mean_uygulama:.1f}')
            plt.axvline(median_uygulama, color='green', linestyle='--', label=f'Medyan: {median_uygulama:.1f}')
            plt.legend()
            
            hist_uygulama_path = os.path.join(figures_dir, 'hist_uygulama_dk.png')
            plt.savefig(hist_uygulama_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Uygulama süresi histogramı kaydedildi: {hist_uygulama_path}")
    
    # 4. Bölüme göre tedavi süresi kutu grafiği
    if 'Bolum' in df.columns and 'TedaviSuresi_num' in df.columns:
        plt.figure(figsize=(12, 8))
        
        # Sadece geçerli değerleri al
        box_data = df[['Bolum', 'TedaviSuresi_num']].dropna()
        
        if len(box_data) > 0:
            sns.boxplot(data=box_data, x='Bolum', y='TedaviSuresi_num')
            plt.title('Bölümlere Göre Tedavi Süresi Dağılımı', fontsize=14, fontweight='bold')
            plt.xlabel('Bölüm')
            plt.ylabel('Tedavi Süresi (Seans)')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            
            box_path = os.path.join(figures_dir, 'box_tedavi_by_bolum.png')
            plt.savefig(box_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Bölüm kutu grafiği kaydedildi: {box_path}")


def generate_eda_summary(df: pd.DataFrame, output_dir: str) -> None:
    """
    Kapsamlı EDA özet raporu oluşturur.
    
    Args:
        df (pd.DataFrame): Veri çerçevesi
        output_dir (str): Çıktı dizini
    """
    summary_path = os.path.join(output_dir, 'eda_summary.md')
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# Fiziksel Tıp & Rehabilitasyon Veri Analizi - EDA Özeti\n\n")
        
        # Genel bilgiler
        f.write("## 📊 Genel Bilgiler\n\n")
        f.write(f"- **Toplam Satır Sayısı**: {len(df):,}\n")
        f.write(f"- **Toplam Sütun Sayısı**: {len(df.columns)}\n")
        f.write(f"- **Veri Boyutu**: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n")
        
        # Sütun bilgileri
        f.write("## 📋 Sütun Bilgileri\n\n")
        f.write("| Sütun | Veri Tipi | Null Sayısı | Null % |\n")
        f.write("|-------|-----------|-------------|--------|\n")
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df)) * 100
            f.write(f"| {col} | {dtype} | {null_count:,} | {null_pct:.1f}% |\n")
        
        # Sayısal sütun istatistikleri
        f.write("\n## 🔢 Sayısal Sütun İstatistikleri\n\n")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            desc_stats = df[numeric_cols].describe()
            f.write(desc_stats.to_markdown())
        else:
            f.write("Sayısal sütun bulunamadı.\n")
        
        # Hedef değişken analizi
        if 'TedaviSuresi_num' in df.columns:
            f.write("\n## 🎯 Hedef Değişken Analizi (TedaviSuresi_num)\n\n")
            target_valid = df['TedaviSuresi_num'].dropna()
            
            if len(target_valid) > 0:
                f.write(f"- **Geçerli Değer Sayısı**: {len(target_valid):,}\n")
                f.write(f"- **Eksik Değer Sayısı**: {df['TedaviSuresi_num'].isnull().sum():,}\n")
                f.write(f"- **Minimum**: {target_valid.min()}\n")
                f.write(f"- **Maksimum**: {target_valid.max()}\n")
                f.write(f"- **Ortalama**: {target_valid.mean():.2f}\n")
                f.write(f"- **Medyan**: {target_valid.median():.2f}\n")
                f.write(f"- **Standart Sapma**: {target_valid.std():.2f}\n")
            
        # Kategorik değişken özeti
        f.write("\n## 📊 Kategorik Değişken Özeti\n\n")
        categorical_cols = ['Cinsiyet', 'KanGrubu', 'Uyruk', 'Bolum', 'TedaviAdi']
        existing_cat_cols = [col for col in categorical_cols if col in df.columns]
        
        for col in existing_cat_cols:
            unique_count = df[col].nunique()
            f.write(f"### {col}\n")
            f.write(f"- **Benzersiz Değer Sayısı**: {unique_count}\n")
            
            if unique_count <= 10:  # Az sayıda kategori varsa dağılımı göster
                value_counts = df[col].value_counts()
                f.write("- **Dağılım**:\n")
                for val, count in value_counts.items():
                    pct = (count / len(df)) * 100
                    f.write(f"  - {val}: {count:,} ({pct:.1f}%)\n")
            f.write("\n")
        
        # Dosya listesi
        f.write("\n## 📁 Oluşturulan Dosyalar\n\n")
        f.write("### Veri Dosyaları\n")
        f.write("- `missing_summary.csv`: Eksik değer analizi\n")
        f.write("- `categorical_unique_counts.csv`: Kategorik değişken analizi\n")
        
        multi_cols = ['KronikHastalik', 'Alerji', 'Tanilar', 'UygulamaYerleri']
        for col in multi_cols:
            if col in df.columns:
                f.write(f"- `top20_{col}.csv`: En sık görülen {col} listesi\n")
        
        f.write("\n### Görselleştirmeler\n")
        f.write("- `figures/hist_yas.png`: Yaş dağılımı histogramı\n")
        f.write("- `figures/hist_tedavi_seans.png`: Tedavi süresi histogramı\n")
        f.write("- `figures/hist_uygulama_dk.png`: Uygulama süresi histogramı\n")
        f.write("- `figures/box_tedavi_by_bolum.png`: Bölümlere göre tedavi süresi\n")
        
        f.write("\n---\n")
        f.write("*Bu rapor otomatik olarak `01_eda.py` script'i tarafından oluşturulmuştur.*\n")
    
    print(f"✅ EDA özet raporu kaydedildi: {summary_path}")


def main():
    """Ana fonksiyon - komut satırı argümanlarını işler ve EDA yapar."""
    parser = argparse.ArgumentParser(
        description="Fiziksel Tıp & Rehabilitasyon Veri Analizi - EDA Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python src/01_eda.py --excel-path data/Talent_Academy_Case_DT_2025.xlsx --sheet Sheet1
  python src/01_eda.py --excel-path data/veri.xlsx --sheet "Veri Sayfası"
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
    
    args = parser.parse_args()
    
    logger.info("🚀 Fiziksel Tıp & Rehabilitasyon EDA Script'i Başlatılıyor...")
    logger.info(f"📁 Excel Dosyası: {args.excel_path}")
    logger.info(f"📄 Sheet: {args.sheet}")
    
    # Çıktı dizinini hazırla
    output_dir = 'reports'
    ensure_directory_exists(output_dir)
    
    # Veriyi yükle
    df = load_data(args.excel_path, args.sheet)
    
    # Temel dönüşümleri uygula
    logger.info("🔄 Temel dönüşümler uygulanıyor...")
    df_transformed = apply_basic_transformations(df)
    
    # Analizleri yap
    logger.info("📊 Analizler yapılıyor...")
    generate_missing_analysis(df_transformed, output_dir)
    generate_categorical_analysis(df_transformed, output_dir)
    generate_top_items_analysis(df_transformed, output_dir)
    
    # Görselleştirmeler oluştur
    logger.info("📈 Görselleştirmeler oluşturuluyor...")
    create_visualizations(df_transformed, output_dir)
    
    # Özet rapor oluştur
    logger.info("📝 Özet rapor oluşturuluyor...")
    generate_eda_summary(df_transformed, output_dir)
    
    logger.info(f"✅ EDA tamamlandı! Tüm çıktılar '{output_dir}' klasöründe.")
    logger.info(f"📋 Özet rapor: {output_dir}/eda_summary.md")


if __name__ == "__main__":
    main()
