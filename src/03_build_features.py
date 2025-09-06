#!/usr/bin/env python3
"""
Fiziksel Tıp & Rehabilitasyon Veri Analizi - Özellik Mühendisliği

Bu script temizlenmiş veriyi alır ve model için hazır özellik matrisi oluşturur.
One-Hot Encoding, çoklu değerli alanlar için binary özellikler ve 
standardizasyon uygular.

Kullanım:
    python src/03_build_features.py --input-csv data/processed/clean_minimal.csv --top_k 50
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# utils modülünü import et
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import (
    create_multilabel_features,
    ensure_directory_exists
)


def load_data(csv_path: str) -> pd.DataFrame:
    """
    CSV dosyasından veriyi yükler.
    
    Args:
        csv_path (str): CSV dosya yolu
        
    Returns:
        pd.DataFrame: Yüklenen veri
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Veri başarıyla yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")
        return df
    except FileNotFoundError:
        print(f"❌ Hata: CSV dosyası bulunamadı: {csv_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Hata: CSV dosyası okunurken hata oluştu: {e}")
        sys.exit(1)


def create_multilabel_features_all(df: pd.DataFrame, top_k: int = 50) -> tuple:
    """
    Tüm çoklu değerli alanlar için binary özellikler oluşturur.
    
    Args:
        df (pd.DataFrame): Veri çerçevesi
        top_k (int): Her alan için en sık görülen kaç öğe
        
    Returns:
        tuple: (özellik_eklenmis_df, yeni_kolon_listesi)
    """
    print(f"🔧 Çoklu değerli alanlar için top-{top_k} binary özellikler oluşturuluyor...")
    
    df_features = df.copy()
    all_new_columns = []
    
    multi_value_columns = ['KronikHastalik', 'Alerji', 'Tanilar', 'UygulamaYerleri']
    existing_multi_cols = [col for col in multi_value_columns if col in df.columns]
    
    for col in existing_multi_cols:
        print(f"   📋 {col} için binary özellikler oluşturuluyor...")
        df_features, new_cols = create_multilabel_features(
            df_features, col, top_k=top_k, prefix="ML"
        )
        all_new_columns.extend(new_cols)
        print(f"      ✅ {len(new_cols)} binary özellik eklendi")
    
    print(f"✅ Toplam {len(all_new_columns)} binary özellik oluşturuldu")
    return df_features, all_new_columns


def prepare_feature_columns(df: pd.DataFrame, ml_columns: list) -> dict:
    """
    Farklı veri tiplerindeki sütunları kategorilere ayırır.
    
    Args:
        df (pd.DataFrame): Veri çerçevesi
        ml_columns (list): ML binary sütunları
        
    Returns:
        dict: Sütun kategorileri
    """
    print("📊 Özellik sütunları kategorize ediliyor...")
    
    # Hedef değişken ve ID sütunları
    target_column = 'TedaviSuresi_num'
    id_columns = ['HastaNo']
    
    # Kategorik sütunlar (One-Hot encoding için)
    categorical_columns = [
        'Cinsiyet', 'KanGrubu', 'Uyruk', 'Bolum', 'TedaviAdi'
    ]
    existing_categorical = [col for col in categorical_columns if col in df.columns]
    
    # Sayısal sütunlar (standardizasyon için)
    numeric_columns = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if (col not in id_columns and 
            col != target_column and 
            not col.endswith('_sayisi') and
            col not in ml_columns):
            numeric_columns.append(col)
    
    # Sayı sütunları (çoklu değerli alanların sayıları)
    count_columns = [col for col in df.columns if col.endswith('_sayisi')]
    
    # Atlanacak sütunlar
    skip_columns = (id_columns + [target_column] + 
                   ['TedaviSuresi', 'UygulamaSuresi'] +  # Orijinal string sütunları
                   ['KronikHastalik', 'Alerji', 'Tanilar', 'UygulamaYerleri'])  # Çoklu değerli orijinal sütunlar
    
    column_info = {
        'target': target_column,
        'id': id_columns,
        'categorical': existing_categorical,
        'numeric': numeric_columns,
        'count': count_columns,
        'ml_binary': ml_columns,
        'skip': skip_columns
    }
    
    print(f"   🎯 Hedef: {target_column}")
    print(f"   🏷️  Kategorik: {len(existing_categorical)} sütun")
    print(f"   🔢 Sayısal: {len(numeric_columns)} sütun")
    print(f"   📊 Sayı: {len(count_columns)} sütun")
    print(f"   🤖 ML Binary: {len(ml_columns)} sütun")
    print(f"   ⏭️  Atlanan: {len(skip_columns)} sütun")
    
    return column_info


def create_feature_pipeline(column_info: dict) -> ColumnTransformer:
    """
    Özellik işleme pipeline'ı oluşturur.
    
    Args:
        column_info (dict): Sütun bilgileri
        
    Returns:
        ColumnTransformer: Özellik işleme pipeline'ı
    """
    print("🔧 Özellik işleme pipeline'ı oluşturuluyor...")
    
    transformers = []
    
    # Sayısal sütunlar için: median imputation + standardization
    if column_info['numeric']:
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('numeric', numeric_pipeline, column_info['numeric']))
        print(f"   🔢 Sayısal pipeline: median imputation + standardization")
    
    # Sayı sütunları için: 0 ile doldur + standardization
    if column_info['count']:
        count_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('scaler', StandardScaler())
        ])
        transformers.append(('count', count_pipeline, column_info['count']))
        print(f"   📊 Sayı pipeline: 0 ile doldur + standardization")
    
    # Kategorik sütunlar için: most frequent + one-hot encoding
    if column_info['categorical']:
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        transformers.append(('categorical', categorical_pipeline, column_info['categorical']))
        print(f"   🏷️  Kategorik pipeline: most frequent + one-hot encoding")
    
    # ML binary sütunları için: hiçbir işlem yapmaz (zaten 0/1)
    if column_info['ml_binary']:
        from sklearn.preprocessing import FunctionTransformer
        ml_pipeline = FunctionTransformer(validate=False)
        transformers.append(('ml_binary', ml_pipeline, column_info['ml_binary']))
        print(f"   🤖 ML Binary: hiçbir işlem uygulanmaz")
    
    # ColumnTransformer oluştur
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'  # Diğer sütunları at
    )
    
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer, column_info: dict) -> list:
    """
    İşlenmiş özellik isimlerini oluşturur.
    
    Args:
        preprocessor (ColumnTransformer): Fit edilmiş preprocessor
        column_info (dict): Sütun bilgileri
        
    Returns:
        list: Özellik isimleri
    """
    feature_names = []
    
    # Sayısal sütun isimleri
    if column_info['numeric']:
        feature_names.extend([f"num_{col}" for col in column_info['numeric']])
    
    # Sayı sütun isimleri
    if column_info['count']:
        feature_names.extend([f"count_{col}" for col in column_info['count']])
    
    # Kategorik sütun isimleri (One-Hot encoding sonrası)
    if column_info['categorical']:
        # One-Hot encoder'dan özellik isimlerini al
        categorical_transformer = preprocessor.named_transformers_['categorical']
        onehot_encoder = categorical_transformer.named_steps['onehot']
        
        cat_feature_names = []
        for i, col in enumerate(column_info['categorical']):
            categories = onehot_encoder.categories_[i]
            # drop='first' kullandığımız için ilk kategoriyi atla
            for cat in categories[1:]:
                cat_feature_names.append(f"cat_{col}_{cat}")
        
        feature_names.extend(cat_feature_names)
    
    # ML binary sütun isimleri
    if column_info['ml_binary']:
        feature_names.extend(column_info['ml_binary'])
    
    return feature_names


def build_features(df: pd.DataFrame, top_k: int = 50) -> tuple:
    """
    Özellik matrisini oluşturur.
    
    Args:
        df (pd.DataFrame): Temizlenmiş veri
        top_k (int): Çoklu değerli alanlar için top-K
        
    Returns:
        tuple: (X, y, feature_names, column_info)
    """
    print("🏗️  Özellik matrisi oluşturuluyor...")
    
    # Çoklu değerli alanlar için binary özellikler oluştur
    df_with_ml, ml_columns = create_multilabel_features_all(df, top_k)
    
    # Sütunları kategorize et
    column_info = prepare_feature_columns(df_with_ml, ml_columns)
    
    # Hedef değişkeni ayır
    if column_info['target'] in df_with_ml.columns:
        y = df_with_ml[column_info['target']].values
        print(f"✅ Hedef değişken ayrıldı: {len(y)} örnek")
    else:
        print(f"❌ Hedef değişken bulunamadı: {column_info['target']}")
        sys.exit(1)
    
    # Özellik sütunlarını seç
    feature_columns = (column_info['numeric'] + 
                      column_info['count'] + 
                      column_info['categorical'] + 
                      column_info['ml_binary'])
    
    X_raw = df_with_ml[feature_columns]
    print(f"✅ Özellik sütunları seçildi: {len(feature_columns)} sütun")
    
    # Pipeline oluştur ve uygula
    preprocessor = create_feature_pipeline(column_info)
    X = preprocessor.fit_transform(X_raw)
    
    # Özellik isimlerini oluştur
    feature_names = get_feature_names(preprocessor, column_info)
    
    print(f"✅ Özellik matrisi oluşturuldu: {X.shape}")
    print(f"   📊 Boyut: {X.shape[0]} örnek × {X.shape[1]} özellik")
    
    return X, y, feature_names, column_info


def save_features(X: np.ndarray, y: np.ndarray, feature_names: list, 
                 column_info: dict, output_dir: str) -> None:
    """
    Özellik matrisini ve hedef değişkeni kaydeder.
    
    Args:
        X (np.ndarray): Özellik matrisi
        y (np.ndarray): Hedef değişken
        feature_names (list): Özellik isimleri
        column_info (dict): Sütun bilgileri
        output_dir (str): Çıktı dizini
    """
    ensure_directory_exists(output_dir)
    
    # Özellik matrisini kaydet
    X_df = pd.DataFrame(X, columns=feature_names)
    X_path = os.path.join(output_dir, 'X_model_ready.csv')
    X_df.to_csv(X_path, index=False, encoding='utf-8-sig')
    print(f"✅ Özellik matrisi kaydedildi: {X_path}")
    
    # Hedef değişkeni kaydet
    y_df = pd.DataFrame(y, columns=['TedaviSuresi_num'])
    y_path = os.path.join(output_dir, 'y.csv')
    y_df.to_csv(y_path, index=False, encoding='utf-8-sig')
    print(f"✅ Hedef değişken kaydedildi: {y_path}")
    
    # Özellik bilgilerini kaydet
    feature_info = pd.DataFrame({
        'feature_name': feature_names,
        'feature_type': ['unknown'] * len(feature_names)  # Tip bilgisini güncelle
    })
    
    # Özellik tiplerini güncelle
    for i, name in enumerate(feature_names):
        if name.startswith('num_'):
            feature_info.loc[i, 'feature_type'] = 'numeric_standardized'
        elif name.startswith('count_'):
            feature_info.loc[i, 'feature_type'] = 'count_standardized'
        elif name.startswith('cat_'):
            feature_info.loc[i, 'feature_type'] = 'categorical_onehot'
        elif name.startswith('ML_'):
            feature_info.loc[i, 'feature_type'] = 'multilabel_binary'
    
    feature_info_path = os.path.join(output_dir, 'feature_info.csv')
    feature_info.to_csv(feature_info_path, index=False, encoding='utf-8-sig')
    print(f"✅ Özellik bilgileri kaydedildi: {feature_info_path}")
    
    # Özet bilgileri kaydet
    summary_info = {
        'total_features': len(feature_names),
        'total_samples': len(y),
        'numeric_features': len([n for n in feature_names if n.startswith('num_')]),
        'count_features': len([n for n in feature_names if n.startswith('count_')]),
        'categorical_features': len([n for n in feature_names if n.startswith('cat_')]),
        'multilabel_features': len([n for n in feature_names if n.startswith('ML_')]),
        'target_mean': float(y.mean()),
        'target_std': float(y.std()),
        'target_min': float(y.min()),
        'target_max': float(y.max())
    }
    
    summary_df = pd.DataFrame([summary_info])
    summary_path = os.path.join(output_dir, 'feature_summary.csv')
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"✅ Özellik özeti kaydedildi: {summary_path}")


def main():
    """Ana fonksiyon - komut satırı argümanlarını işler ve özellik mühendisliği yapar."""
    parser = argparse.ArgumentParser(
        description="Fiziksel Tıp & Rehabilitasyon Veri Analizi - Özellik Mühendisliği Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python src/03_build_features.py --input-csv data/processed/clean_minimal.csv --top_k 50
  python src/03_build_features.py --input-csv data/processed/model_ready_minimal.csv --top_k 30

Çıktılar:
  - data/processed/X_model_ready.csv: Özellik matrisi
  - data/processed/y.csv: Hedef değişken
  - data/processed/feature_info.csv: Özellik bilgileri
  - data/processed/feature_summary.csv: Özellik özeti
        """
    )
    
    parser.add_argument(
        '--input-csv',
        required=True,
        help='Temizlenmiş CSV dosya yolu (örn: data/processed/clean_minimal.csv)'
    )
    
    parser.add_argument(
        '--top_k',
        type=int,
        default=50,
        help='Çoklu değerli alanlar için en sık görülen kaç öğe (varsayılan: 50)'
    )
    
    args = parser.parse_args()
    
    print("🚀 Fiziksel Tıp & Rehabilitasyon Özellik Mühendisliği Script'i Başlatılıyor...")
    print(f"📁 Input CSV: {args.input_csv}")
    print(f"🔢 Top-K: {args.top_k}")
    
    # Çıktı dizinini hazırla
    output_dir = 'data/processed'
    ensure_directory_exists(output_dir)
    
    # Veriyi yükle
    df = load_data(args.input_csv)
    
    # Özellik matrisini oluştur
    X, y, feature_names, column_info = build_features(df, top_k=args.top_k)
    
    # Sonuçları kaydet
    print(f"\n💾 Özellik matrisi kaydediliyor...")
    save_features(X, y, feature_names, column_info, output_dir)
    
    print(f"\n✅ Özellik mühendisliği tamamlandı!")
    print(f"📁 Çıktı dosyaları: {output_dir}/")
    print("   - X_model_ready.csv: Özellik matrisi")
    print("   - y.csv: Hedef değişken")
    print("   - feature_info.csv: Özellik bilgileri")
    print("   - feature_summary.csv: Özellik özeti")
    
    print(f"\n📊 Final Özetler:")
    print(f"   🎯 Örnekler: {len(y):,}")
    print(f"   🔧 Özellikler: {len(feature_names):,}")
    print(f"   📈 Hedef ortalama: {y.mean():.2f}")
    print(f"   📉 Hedef std: {y.std():.2f}")


if __name__ == "__main__":
    main()
