# Gerekli Kütüphaneler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from mlxtend.frequent_patterns import apriori, association_rules
import warnings

warnings.filterwarnings('ignore')

# 1. Veri Yükleme
df = pd.read_csv("online_retail_II.csv", encoding='ISO-8859-1')

# 2. Eksik Veri Analizi ve Haritası
print("Eksik Veri Sayısı (Sütun Bazında):")
print(df.isnull().sum())

plt.figure(figsize=(12,6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Eksik Veri Haritası")
plt.show()

# 3. Eksik Verilerin Temizlenmesi
df_clean = df.dropna()
print("\nTemizlendikten Sonra Eksik Veri Sayısı:")
print(df_clean.isnull().sum())

# 4. Aykırı Değer Analizi: Quantity
plt.figure(figsize=(10,6))
sns.boxplot(x=df_clean['Quantity'])
plt.title('Quantity Aykırı Değer Analizi')
plt.show()

# 5. Aykırı Değer Analizi: Price
plt.figure(figsize=(10,6))
sns.boxplot(x=df_clean['Price'])
plt.title('Price Aykırı Değer Analizi')
plt.show()

# 6. Aykırı Değerlerin Temizlenmesi (Quantity ve Price pozitif olmalı)
df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['Price'] > 0)]

# 7. Normalizasyon (Quantity ve Price)
scaler = MinMaxScaler()
df_clean[['Quantity', 'Price']] = scaler.fit_transform(df_clean[['Quantity', 'Price']])

# 8. Kategorik Verilerin Dönüştürülmesi (Country -> One-hot)
df_encoded = pd.get_dummies(df_clean, columns=['Country'])

# 9. Quantity ve Price Dağılımı (Normalizasyon Sonrası)
plt.figure(figsize=(10,6))
sns.scatterplot(x='Quantity', y='Price', data=df_clean, alpha=0.5)
plt.title('Quantity ve Price Dağılımı (Normalizasyon Sonrası)')
plt.xlabel('Quantity (Normalized)')
plt.ylabel('Price (Normalized)')
plt.show()

# 10. Öncesi ve Sonrası Özet Karşılaştırma
print("\n=== Veri Ön İşleme Özeti ===")
print(f"Orijinal Veri Satır Sayısı: {df.shape[0]}")
print(f"Temizlenmiş Veri Satır Sayısı: {df_clean.shape[0]}")
print(f"Orijinal Ortalama Price: {df['Price'].mean():.2f}")
print(f"Temizlenmiş Ortalama Price: {df_clean['Price'].mean():.2f}")
print(f"Orijinal Ortalama Quantity: {df['Quantity'].mean():.2f}")
print(f"Temizlenmiş Ortalama Quantity: {df_clean['Quantity'].mean():.2f}")

# 11. Birliktelik Analizi için Sepet Hazırlama
basket = df_clean.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0)

# Ürün var/yok şeklinde kodlama
def encode_units(x):
    return 0 if x <= 0 else 1

basket_sets = basket.applymap(encode_units)

# 12. Apriori ile Sık Ürün Kümeleri (min_support=0.01)
frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)

# 13. Birliktelik Kuralları (min_confidence=0.6)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

# 14. En Güçlü 10 Kuralı Lift’e göre sıralama
top_rules = rules.sort_values('lift', ascending=False).head(10)

print("\n=== Birliktelik Analizi Sonuçları ===")
print(top_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# 15. Destek-Güven Grafiği (Lift renkli)
plt.figure(figsize=(10,6))
scatter = plt.scatter(rules['support'], rules['confidence'], 
                      c=rules['lift'], cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Lift')
plt.xlabel('Destek (Support)')
plt.ylabel('Güven (Confidence)')
plt.title('Birliktelik Kuralları: Destek ve Güven Dağılımı (Lift Renkli)')
plt.show()

plt.figure(figsize=(10,6))
scatter = plt.scatter(rules['support'], rules['confidence'], 
                      c=rules['lift'], cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Lift')
plt.xlabel('Destek (Support)')
plt.ylabel('Güven (Confidence)')
plt.title('Birliktelik Kuralları: Destek ve Güven Dağılımı (Lift Renkli)')
plt.show()

frequent_itemsets.sort_values('support', ascending=False).head(10).plot.bar(x='itemsets', y='support', legend=False)
plt.title("En Sık Ürün Kümeleri")
plt.ylabel("Destek (Support)")
plt.xlabel("Ürün Kümeleri")
plt.xticks(rotation=45, ha='right')
plt.show()

frequent_itemsets.sort_values('support', ascending=False).head(10).plot.bar(x='itemsets', y='support', legend=False)
plt.title("En Sık Ürün Kümeleri")
plt.ylabel("Destek (Support)")
plt.xlabel("Ürün Kümeleri")
plt.xticks(rotation=45, ha='right')
plt.show()

plt.figure(figsize=(12,6))
df_clean['Description'].value_counts().head(10).plot(kind='bar')
plt.title('En Çok Satılan İlk 10 Ürün')
plt.ylabel('Satış Adedi')
plt.xlabel('Ürünler')
plt.xticks(rotation=45, ha='right')
plt.show()


df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['Price']

invoice_total = df_clean.groupby('Invoice')['TotalPrice'].sum()

plt.figure(figsize=(12,6))
sns.histplot(invoice_total, bins=50, kde=True)
plt.title('Fatura Başına Ortalama Harcama Dağılımı')
plt.xlabel('Fatura Tutarı')
plt.ylabel('Frekans')
plt.show()

plt.figure(figsize=(12,6))
df_clean['Country'].value_counts().head(10).plot(kind='bar', color='orange')
plt.title('En Çok Satış Yapılan Ülkeler')
plt.ylabel('Satış Adedi')
plt.xlabel('Ülke')
plt.xticks(rotation=45, ha='right')
plt.show()

