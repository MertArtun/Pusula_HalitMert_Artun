"""
Fiziksel Tıp & Rehabilitasyon Veri Analizi - Yardımcı Fonksiyonlar

Bu modül veri işleme ve dönüştürme için gerekli yardımcı fonksiyonları içerir.
"""

import re
import pandas as pd
import numpy as np
from typing import Union, List, Optional, Tuple


def parse_tedavi_suresi_to_int(tedavi_suresi: str) -> Union[int, float]:
    """
    Tedavi süresi stringini sayısal değere dönüştürür.
    
    Args:
        tedavi_suresi (str): "15 Seans", "10 seans", "5 SEANS" gibi formatlar
        
    Returns:
        Union[int, float]: Seans sayısı, parse edilemezse NaN
        
    Examples:
        >>> parse_tedavi_suresi_to_int("15 Seans")
        15
        >>> parse_tedavi_suresi_to_int("10 seans")
        10
        >>> parse_tedavi_suresi_to_int("Belirsiz")
        nan
    """
    if pd.isna(tedavi_suresi) or not isinstance(tedavi_suresi, str):
        return np.nan
    
    # Sayı ve "seans" kelimesini içeren pattern
    pattern = r'(\d+)\s*(?:seans|Seans|SEANS)'
    match = re.search(pattern, tedavi_suresi.strip())
    
    if match:
        return int(match.group(1))
    
    # Eğer sadece sayı varsa
    pattern_only_number = r'^(\d+)$'
    match_number = re.search(pattern_only_number, tedavi_suresi.strip())
    if match_number:
        return int(match_number.group(1))
    
    return np.nan


def parse_sure_to_minutes(sure_str: str) -> Union[float, int]:
    """
    Süre stringini dakikaya dönüştürür.
    
    Args:
        sure_str (str): "20 Dakika", "1 Saat 30 Dakika", "15 dk", "2 saat" gibi formatlar
        
    Returns:
        Union[float, int]: Toplam dakika, parse edilemezse NaN
        
    Examples:
        >>> parse_sure_to_minutes("20 Dakika")
        20
        >>> parse_sure_to_minutes("1 Saat 30 Dakika")
        90
        >>> parse_sure_to_minutes("15 dk")
        15
        >>> parse_sure_to_minutes("2 saat")
        120
    """
    if pd.isna(sure_str) or not isinstance(sure_str, str):
        return np.nan
    
    sure_str = sure_str.strip().lower()
    total_minutes = 0
    
    # Saat pattern'i
    saat_pattern = r'(\d+)\s*(?:saat|hour|hr)'
    saat_match = re.search(saat_pattern, sure_str)
    if saat_match:
        total_minutes += int(saat_match.group(1)) * 60
    
    # Dakika pattern'i
    dakika_pattern = r'(\d+)\s*(?:dakika|dk|minute|min)'
    dakika_match = re.search(dakika_pattern, sure_str)
    if dakika_match:
        total_minutes += int(dakika_match.group(1))
    
    # Eğer hiçbir pattern eşleşmezse, sadece sayı olup olmadığını kontrol et
    if total_minutes == 0:
        only_number_pattern = r'^(\d+)$'
        number_match = re.search(only_number_pattern, sure_str)
        if number_match:
            # Varsayılan olarak dakika kabul et
            total_minutes = int(number_match.group(1))
    
    return total_minutes if total_minutes > 0 else np.nan


def split_list(list_str: str, separator: str = ',') -> List[str]:
    """
    Virgülle (veya başka ayıraçla) ayrılmış string'i temizlenmiş listeye dönüştürür.
    
    Args:
        list_str (str): "Diabetes, Hipertansiyon, Kalp hastalığı" gibi string
        separator (str): Ayıraç karakteri (varsayılan: ',')
        
    Returns:
        List[str]: Temizlenmiş liste, boş string'ler çıkarılmış
        
    Examples:
        >>> split_list("Diabetes, Hipertansiyon, ")
        ['Diabetes', 'Hipertansiyon']
        >>> split_list("A,B,C")
        ['A', 'B', 'C']
    """
    if pd.isna(list_str) or not isinstance(list_str, str):
        return []
    
    # Ayıraç ile böl, whitespace'leri temizle, boş string'leri çıkar
    items = [item.strip() for item in list_str.split(separator)]
    items = [item for item in items if item]  # Boş string'leri çıkar
    
    return items


def add_count_columns(df: pd.DataFrame, list_columns: List[str]) -> pd.DataFrame:
    """
    Çoklu değerli sütunlar için sayı sütunları ekler.
    
    Args:
        df (pd.DataFrame): Veri çerçevesi
        list_columns (List[str]): Çoklu değerli sütun isimleri
        
    Returns:
        pd.DataFrame: Sayı sütunları eklenmiş veri çerçevesi
        
    Examples:
        >>> df = pd.DataFrame({'KronikHastalik': ['A,B,C', 'D,E', 'F']})
        >>> add_count_columns(df, ['KronikHastalik'])
        # KronikHastalik_sayisi sütunu eklenir: [3, 2, 1]
    """
    df_copy = df.copy()
    
    for col in list_columns:
        if col in df_copy.columns:
            # Her satır için öğe sayısını hesapla
            count_col_name = f"{col}_sayisi"
            df_copy[count_col_name] = df_copy[col].apply(
                lambda x: len(split_list(x)) if pd.notna(x) else 0
            )
    
    return df_copy


def top_items_series(series: pd.Series, top_n: int = 20) -> pd.DataFrame:
    """
    Çoklu değerli sütunda en sık görülen öğeleri bulur.
    
    Args:
        series (pd.Series): Çoklu değerli sütun
        top_n (int): Kaç tane en sık görülen öğe döndürüleceği
        
    Returns:
        pd.DataFrame: En sık görülen öğeler ve sayıları
        
    Examples:
        >>> s = pd.Series(['A,B', 'B,C', 'A,C'])
        >>> top_items_series(s, top_n=3)
        # A: 2, B: 2, C: 2 döndürür
    """
    # Tüm öğeleri topla
    all_items = []
    for value in series.dropna():
        items = split_list(str(value))
        all_items.extend(items)
    
    # Sayıları hesapla
    if all_items:
        item_counts = pd.Series(all_items).value_counts().head(top_n)
        return pd.DataFrame({
            'oge': item_counts.index,
            'sayi': item_counts.values
        })
    else:
        return pd.DataFrame({'oge': [], 'sayi': []})


def create_multilabel_features(df: pd.DataFrame, 
                             column: str, 
                             top_k: int = 50, 
                             prefix: str = "ML") -> Tuple[pd.DataFrame, List[str]]:
    """
    Çoklu değerli sütun için binary özellik sütunları oluşturur.
    
    Args:
        df (pd.DataFrame): Veri çerçevesi
        column (str): Çoklu değerli sütun adı
        top_k (int): En sık görülen kaç öğe için özellik oluşturulacağı
        prefix (str): Yeni sütun isimleri için önek
        
    Returns:
        Tuple[pd.DataFrame, List[str]]: (Özellik sütunları eklenmiş df, yeni sütun isimleri)
        
    Examples:
        >>> df = pd.DataFrame({'Hastalik': ['A,B', 'B,C', 'A']})
        >>> new_df, cols = create_multilabel_features(df, 'Hastalik', top_k=2)
        # ML_Hastalik_A, ML_Hastalik_B sütunları eklenir
    """
    df_copy = df.copy()
    
    # En sık görülen öğeleri bul
    top_items_df = top_items_series(df[column], top_n=top_k)
    top_items = top_items_df['oge'].tolist()
    
    new_columns = []
    
    # Her top öğe için binary sütun oluştur
    for item in top_items:
        col_name = f"{prefix}_{column}_{item}"
        # Özel karakterleri temizle
        col_name = re.sub(r'[^\w_]', '_', col_name)
        
        df_copy[col_name] = df[column].apply(
            lambda x: 1 if item in split_list(str(x)) else 0
        )
        new_columns.append(col_name)
    
    return df_copy, new_columns


def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Veri çerçevesi için eksik değer özeti oluşturur.
    
    Args:
        df (pd.DataFrame): Veri çerçevesi
        
    Returns:
        pd.DataFrame: Eksik değer özeti (sütun, eksik_sayi, eksik_yuzde)
    """
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100
    
    missing_summary = pd.DataFrame({
        'sutun': missing_count.index,
        'eksik_sayi': missing_count.values,
        'eksik_yuzde': missing_percentage.values
    })
    
    # Sadece eksik değeri olan sütunları göster
    missing_summary = missing_summary[missing_summary['eksik_sayi'] > 0]
    missing_summary = missing_summary.sort_values('eksik_sayi', ascending=False)
    
    return missing_summary


def get_categorical_summary(df: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
    """
    Kategorik sütunlar için benzersizlik özeti oluşturur.
    
    Args:
        df (pd.DataFrame): Veri çerçevesi
        categorical_columns (List[str]): Kategorik sütun isimleri
        
    Returns:
        pd.DataFrame: Kategorik özet (sutun, benzersiz_sayi, en_sik_deger, en_sik_frekans)
    """
    categorical_summary = []
    
    for col in categorical_columns:
        if col in df.columns:
            unique_count = df[col].nunique()
            most_frequent = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "N/A"
            most_frequent_count = df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0
            
            categorical_summary.append({
                'sutun': col,
                'benzersiz_sayi': unique_count,
                'en_sik_deger': most_frequent,
                'en_sik_frekans': most_frequent_count
            })
    
    return pd.DataFrame(categorical_summary)


def safe_to_numeric(series: pd.Series) -> pd.Series:
    """
    Seriyi güvenli bir şekilde sayısal değere dönüştürür.
    
    Args:
        series (pd.Series): Dönüştürülecek seri
        
    Returns:
        pd.Series: Sayısal seri, dönüştürülemeyenler NaN olur
    """
    return pd.to_numeric(series, errors='coerce')


def ensure_directory_exists(directory_path: str) -> None:
    """
    Dizinin var olduğundan emin olur, yoksa oluşturur.
    
    Args:
        directory_path (str): Dizin yolu
    """
    import os
    os.makedirs(directory_path, exist_ok=True)


if __name__ == "__main__":
    # Test fonksiyonları
    print("=== Tedavi Süresi Parse Testi ===")
    test_tedavi = ["15 Seans", "10 seans", "5 SEANS", "Belirsiz", "20"]
    for t in test_tedavi:
        result = parse_tedavi_suresi_to_int(t)
        print(f"'{t}' -> {result}")
    
    print("\n=== Süre Parse Testi ===")
    test_sure = ["20 Dakika", "1 Saat 30 Dakika", "15 dk", "2 saat", "45"]
    for s in test_sure:
        result = parse_sure_to_minutes(s)
        print(f"'{s}' -> {result} dakika")
    
    print("\n=== Liste Split Testi ===")
    test_liste = ["Diabetes, Hipertansiyon, Kalp hastalığı", "A,B,C", "Tek öğe"]
    for l in test_liste:
        result = split_list(l)
        print(f"'{l}' -> {result}")
