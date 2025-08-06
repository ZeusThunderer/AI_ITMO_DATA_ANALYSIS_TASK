#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exploratory Data Analysis (EDA) для датасета транзакций
Анализ данных для выявления мошеннических операций
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime, timedelta
import os
from pathlib import Path

# Настройки для корректного отображения
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Создание директории для графиков
os.makedirs('plots', exist_ok=True)

# Настройка для корректного отображения русских символов
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']

def convert_to_usd(transactions_df, currency_df):
    """
    Конвертирует все суммы транзакций в доллары США
    используя исторические курсы валют
    """
    print("Конвертация сумм в доллары США...")
    
    # Создаем копию датафрейма
    df = transactions_df.copy()
    
    # Преобразуем timestamp в datetime для объединения
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    
    # Преобразуем date в currency_df в datetime.date
    currency_df['date'] = pd.to_datetime(currency_df['date']).dt.date
    
    # Объединяем данные по дате и валюте
    # Сначала объединяем по дате
    df_merged = df.merge(currency_df, on='date', how='left')
    
    # Создаем столбец с курсом валюты для каждой транзакции
    df_merged['exchange_rate'] = 1.0  # По умолчанию для USD
    
    # Для каждой валюты устанавливаем соответствующий курс
    currencies = ['AUD', 'BRL', 'CAD', 'EUR', 'GBP', 'JPY', 'MXN', 'NGN', 'RUB', 'SGD']
    
    for currency in currencies:
        if currency in df_merged.columns:
            mask = df_merged['currency'] == currency
            df_merged.loc[mask, 'exchange_rate'] = df_merged.loc[mask, currency]
    
    # Конвертируем суммы в USD
    df_merged['amount_usd'] = df_merged['amount'] / df_merged['exchange_rate']
    
    # Заменяем оригинальный столбец amount на конвертированный
    df_merged['amount'] = df_merged['amount_usd']
    
    # Удаляем временные столбцы
    df_merged = df_merged.drop(['amount_usd', 'exchange_rate', 'date'] + currencies + ['USD'], axis=1, errors='ignore')
    
    print(f"Конвертация завершена. Все суммы теперь в долларах США.")
    print(f"Диапазон сумм после конвертации: ${df_merged['amount'].min():.2f} - ${df_merged['amount'].max():.2f}")
    
    return df_merged

def load_data():
    """Загрузка данных и конвертация в USD"""
    print("Загрузка данных...")
    
    # Загрузка основного датасета
    transactions_df = pd.read_parquet('../data/transaction_fraud_data.parquet')
    
    # Загрузка данных о курсах валют
    currency_df = pd.read_parquet('../data/historical_currency_exchange.parquet')
    
    # Конвертация всех сумм в USD
    transactions_df = convert_to_usd(transactions_df, currency_df)
    
    return transactions_df, currency_df

def basic_info(df, currency_df, output_file):
    """Базовая информация о датасетах"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ОСНОВНАЯ ИНФОРМАЦИЯ О ДАТАСЕТАХ\n")
        f.write("=" * 80 + "\n\n")
        
        # Информация о транзакциях
        f.write("ДАТАСЕТ ТРАНЗАКЦИЙ:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Размер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов\n")
        f.write(f"Память, занимаемая датасетом: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n")
        
        f.write("Типы данных:\n")
        f.write(str(df.dtypes) + "\n\n")
        
        f.write("Первые 5 строк:\n")
        f.write(str(df.head()) + "\n\n")
        
        f.write("Последние 5 строк:\n")
        f.write(str(df.tail()) + "\n\n")
        
        # Информация о валютах
        f.write("ДАТАСЕТ КУРСОВ ВАЛЮТ:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Размер датасета: {currency_df.shape[0]} строк, {currency_df.shape[1]} столбцов\n")
        f.write(f"Период: с {currency_df['date'].min()} по {currency_df['date'].max()}\n")
        f.write(f"Доступные валюты: {', '.join(currency_df.columns[1:])}\n\n")
        f.write("ВАЖНО: Все суммы транзакций конвертированы в доллары США (USD)\n")
        f.write("используя исторические курсы валют на дату транзакции.\n\n")
        
        f.write("Первые 5 строк курсов валют:\n")
        f.write(str(currency_df.head()) + "\n\n")

def missing_values_analysis(df, output_file):
    """Анализ пропущенных значений"""
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write("АНАЛИЗ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ\n")
        f.write("=" * 50 + "\n")
        
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        
        missing_df = pd.DataFrame({
            'Количество пропусков': missing_data,
            'Процент пропусков': missing_percent
        }).sort_values('Количество пропусков', ascending=False)
        
        f.write(str(missing_df) + "\n\n")
        
        # Визуализация пропущенных значений
        plt.figure(figsize=(12, 6))
        missing_percent.plot(kind='bar')
        plt.title('Процент пропущенных значений по столбцам')
        plt.xlabel('Столбцы')
        plt.ylabel('Процент пропусков')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('plots/missing_values.png', dpi=300, bbox_inches='tight')
        plt.close()

def fraud_analysis(df, output_file):
    """Анализ мошеннических транзакций"""
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write("АНАЛИЗ МОШЕННИЧЕСКИХ ТРАНЗАКЦИЙ\n")
        f.write("=" * 50 + "\n")
        
        fraud_counts = df['is_fraud'].value_counts()
        fraud_percent = (fraud_counts / len(df)) * 100
        
        f.write(f"Общее количество транзакций: {len(df):,}\n")
        f.write(f"Легитимные транзакции: {fraud_counts[False]:,} ({fraud_percent[False]:.2f}%)\n")
        f.write(f"Мошеннические транзакции: {fraud_counts[True]:,} ({fraud_percent[True]:.2f}%)\n\n")
        
        # Визуализация распределения мошенничества
        plt.figure(figsize=(10, 6))
        
        # Круговая диаграмма
        plt.subplot(1, 2, 1)
        plt.pie(fraud_counts.values, labels=['Легитимные', 'Мошеннические'], 
                autopct='%1.1f%%', startangle=90)
        plt.title('Распределение транзакций')
        
        # Столбчатая диаграмма
        plt.subplot(1, 2, 2)
        fraud_counts.plot(kind='bar')
        plt.title('Количество транзакций по типу')
        plt.xlabel('Тип транзакции')
        plt.ylabel('Количество')
        plt.xticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig('plots/fraud_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

def temporal_analysis(df, output_file):
    """Временной анализ транзакций"""
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write("ВРЕМЕННОЙ АНАЛИЗ ТРАНЗАКЦИЙ\n")
        f.write("=" * 50 + "\n")
        
        # Преобразование timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        
        # Анализ по дням недели
        day_fraud = df.groupby(['day_of_week', 'is_fraud']).size().unstack(fill_value=0)
        day_fraud_pct = day_fraud.div(day_fraud.sum(axis=1), axis=0) * 100
        
        f.write("Распределение мошенничества по дням недели:\n")
        f.write(str(day_fraud_pct) + "\n\n")
        
        # Анализ по часам
        hour_fraud = df.groupby(['hour', 'is_fraud']).size().unstack(fill_value=0)
        hour_fraud_pct = hour_fraud.div(hour_fraud.sum(axis=1), axis=0) * 100
        
        f.write("Распределение мошенничества по часам:\n")
        f.write(str(hour_fraud_pct) + "\n\n")
        
        # Визуализация
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # По дням недели
        day_fraud.plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Количество транзакций по дням недели')
        axes[0,0].set_xlabel('День недели')
        axes[0,0].set_ylabel('Количество')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        day_fraud_pct.plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('Процент мошенничества по дням недели')
        axes[0,1].set_xlabel('День недели')
        axes[0,1].set_ylabel('Процент')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # По часам
        hour_fraud.plot(kind='line', ax=axes[1,0], marker='o')
        axes[1,0].set_title('Количество транзакций по часам')
        axes[1,0].set_xlabel('Час')
        axes[1,0].set_ylabel('Количество')
        
        hour_fraud_pct.plot(kind='line', ax=axes[1,1], marker='o')
        axes[1,1].set_title('Процент мошенничества по часам')
        axes[1,1].set_xlabel('Час')
        axes[1,1].set_ylabel('Процент')
        
        plt.tight_layout()
        plt.savefig('plots/temporal_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

def amount_analysis(df, output_file):
    """Анализ сумм транзакций (в долларах США)"""
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write("АНАЛИЗ СУММ ТРАНЗАКЦИЙ (В ДОЛЛАРАХ США)\n")
        f.write("=" * 50 + "\n")
        
        # Статистики по суммам
        amount_stats = df.groupby('is_fraud')['amount'].describe()
        f.write("Статистики по суммам транзакций:\n")
        f.write(str(amount_stats) + "\n\n")
        
        # Анализ по валютам
        currency_fraud = df.groupby(['currency', 'is_fraud']).size().unstack(fill_value=0)
        currency_fraud_pct = currency_fraud.div(currency_fraud.sum(axis=1), axis=0) * 100
        
        f.write("Распределение мошенничества по валютам:\n")
        f.write(str(currency_fraud_pct) + "\n\n")
        
        # Визуализация
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Распределение сумм
        df[df['is_fraud'] == False]['amount'].hist(bins=50, alpha=0.7, ax=axes[0,0], label='Легитимные')
        df[df['is_fraud'] == True]['amount'].hist(bins=50, alpha=0.7, ax=axes[0,0], label='Мошеннические')
        axes[0,0].set_title('Распределение сумм транзакций')
        axes[0,0].set_xlabel('Сумма')
        axes[0,0].set_ylabel('Частота')
        axes[0,0].legend()
        axes[0,0].set_yscale('log')
        
        # Box plot сумм
        df.boxplot(column='amount', by='is_fraud', ax=axes[0,1])
        axes[0,1].set_title('Распределение сумм по типу транзакции')
        axes[0,1].set_xlabel('Тип транзакции')
        axes[0,1].set_ylabel('Сумма')
        
        # Мошенничество по валютам
        currency_fraud_pct.plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Процент мошенничества по валютам')
        axes[1,0].set_xlabel('Валюта')
        axes[1,0].set_ylabel('Процент')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Средние суммы по валютам
        avg_amount_by_currency = df.groupby(['currency', 'is_fraud'])['amount'].mean().unstack()
        avg_amount_by_currency.plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Средние суммы по валютам')
        axes[1,1].set_xlabel('Валюта')
        axes[1,1].set_ylabel('Средняя сумма')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('plots/amount_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

def categorical_analysis(df, output_file):
    """Анализ категориальных переменных"""
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write("АНАЛИЗ КАТЕГОРИАЛЬНЫХ ПЕРЕМЕННЫХ\n")
        f.write("=" * 50 + "\n")
        
        categorical_cols = ['vendor_category', 'vendor_type', 'card_type', 'channel', 
                           'city_size', 'is_card_present', 'is_outside_home_country', 
                           'is_high_risk_vendor', 'is_weekend']
        
        # Анализ каждой категориальной переменной
        for col in categorical_cols:
            if col in df.columns:
                f.write(f"\n{col.upper()}:\n")
                f.write("-" * 30 + "\n")
                
                # Распределение значений
                value_counts = df[col].value_counts()
                f.write("Распределение значений:\n")
                f.write(str(value_counts) + "\n\n")
                
                # Процент мошенничества по категориям
                fraud_by_cat = df.groupby([col, 'is_fraud']).size().unstack(fill_value=0)
                fraud_pct_by_cat = fraud_by_cat.div(fraud_by_cat.sum(axis=1), axis=0) * 100
                
                f.write("Процент мошенничества по категориям:\n")
                f.write(str(fraud_pct_by_cat) + "\n\n")
        
        # Визуализация основных категориальных переменных
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        
        # Vendor Category
        vendor_cat_fraud = df.groupby(['vendor_category', 'is_fraud']).size().unstack(fill_value=0)
        vendor_cat_fraud_pct = vendor_cat_fraud.div(vendor_cat_fraud.sum(axis=1), axis=0) * 100
        vendor_cat_fraud_pct.plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Процент мошенничества по категориям вендоров')
        axes[0,0].set_xlabel('Категория вендора')
        axes[0,0].set_ylabel('Процент')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Card Type
        card_type_fraud = df.groupby(['card_type', 'is_fraud']).size().unstack(fill_value=0)
        card_type_fraud_pct = card_type_fraud.div(card_type_fraud.sum(axis=1), axis=0) * 100
        card_type_fraud_pct.plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('Процент мошенничества по типам карт')
        axes[0,1].set_xlabel('Тип карты')
        axes[0,1].set_ylabel('Процент')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Channel
        channel_fraud = df.groupby(['channel', 'is_fraud']).size().unstack(fill_value=0)
        channel_fraud_pct = channel_fraud.div(channel_fraud.sum(axis=1), axis=0) * 100
        channel_fraud_pct.plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Процент мошенничества по каналам')
        axes[1,0].set_xlabel('Канал')
        axes[1,0].set_ylabel('Процент')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # City Size
        city_size_fraud = df.groupby(['city_size', 'is_fraud']).size().unstack(fill_value=0)
        city_size_fraud_pct = city_size_fraud.div(city_size_fraud.sum(axis=1), axis=0) * 100
        city_size_fraud_pct.plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Процент мошенничества по размеру города')
        axes[1,1].set_xlabel('Размер города')
        axes[1,1].set_ylabel('Процент')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        # Boolean features
        bool_features = ['is_card_present', 'is_outside_home_country', 'is_high_risk_vendor', 'is_weekend']
        bool_fraud_data = []
        bool_labels = []
        
        for feature in bool_features:
            if feature in df.columns:
                fraud_pct = df.groupby(feature)['is_fraud'].mean() * 100
                bool_fraud_data.append(fraud_pct.values)
                bool_labels.extend([f'{feature}_False', f'{feature}_True'])
        
        if bool_fraud_data:
            axes[2,0].bar(range(len(bool_labels)), [item for sublist in bool_fraud_data for item in sublist])
            axes[2,0].set_title('Процент мошенничества для булевых признаков')
            axes[2,0].set_xlabel('Признаки')
            axes[2,0].set_ylabel('Процент мошенничества')
            axes[2,0].set_xticks(range(len(bool_labels)))
            axes[2,0].set_xticklabels(bool_labels, rotation=45, ha='right')
        
        # Убираем последний subplot если он не используется
        axes[2,1].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('plots/categorical_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

def geographical_analysis(df, output_file):
    """Географический анализ"""
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write("ГЕОГРАФИЧЕСКИЙ АНАЛИЗ\n")
        f.write("=" * 50 + "\n")
        
        # Анализ по странам
        country_fraud = df.groupby(['country', 'is_fraud']).size().unstack(fill_value=0)
        country_fraud_pct = country_fraud.div(country_fraud.sum(axis=1), axis=0) * 100
        
        f.write("Топ-10 стран по проценту мошенничества:\n")
        top_fraud_countries = country_fraud_pct[True].sort_values(ascending=False).head(10)
        f.write(str(top_fraud_countries) + "\n\n")
        
        f.write("Топ-10 стран по количеству транзакций:\n")
        top_countries_by_volume = country_fraud.sum(axis=1).sort_values(ascending=False).head(10)
        f.write(str(top_countries_by_volume) + "\n\n")
        
        # Анализ по городам
        city_fraud = df.groupby(['city', 'is_fraud']).size().unstack(fill_value=0)
        city_fraud_pct = city_fraud.div(city_fraud.sum(axis=1), axis=0) * 100
        
        f.write("Топ-10 городов по проценту мошенничества:\n")
        top_fraud_cities = city_fraud_pct[True].sort_values(ascending=False).head(10)
        f.write(str(top_fraud_cities) + "\n\n")
        
        # Визуализация
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Топ стран по мошенничеству
        top_fraud_countries.plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Топ-10 стран по проценту мошенничества')
        axes[0,0].set_xlabel('Страна')
        axes[0,0].set_ylabel('Процент мошенничества')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Топ стран по объему
        top_countries_by_volume.plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('Топ-10 стран по количеству транзакций')
        axes[0,1].set_xlabel('Страна')
        axes[0,1].set_ylabel('Количество транзакций')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Топ городов по мошенничеству
        top_fraud_cities.plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Топ-10 городов по проценту мошенничества')
        axes[1,0].set_xlabel('Город')
        axes[1,0].set_ylabel('Процент мошенничества')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Распределение по размеру города
        city_size_fraud = df.groupby(['city_size', 'is_fraud']).size().unstack(fill_value=0)
        city_size_fraud_pct = city_size_fraud.div(city_size_fraud.sum(axis=1), axis=0) * 100
        city_size_fraud_pct.plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Процент мошенничества по размеру города')
        axes[1,1].set_xlabel('Размер города')
        axes[1,1].set_ylabel('Процент')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('plots/geographical_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

def activity_analysis(df, output_file):
    """Анализ активности за последний час"""
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write("АНАЛИЗ АКТИВНОСТИ ЗА ПОСЛЕДНИЙ ЧАС\n")
        f.write("=" * 50 + "\n")
        
        # Извлечение данных из структуры last_hour_activity
        if 'last_hour_activity' in df.columns:
            # Преобразуем структуру в отдельные столбцы
            activity_cols = ['num_transactions', 'total_amount', 'unique_merchants', 
                           'unique_countries', 'max_single_amount']
            
            for col in activity_cols:
                df[f'last_hour_{col}'] = df['last_hour_activity'].apply(lambda x: x[col] if x else 0)
            
            # Анализ каждого показателя
            for col in activity_cols:
                full_col_name = f'last_hour_{col}'
                f.write(f"\n{col.upper()}:\n")
                f.write("-" * 30 + "\n")
                
                stats = df.groupby('is_fraud')[full_col_name].describe()
                f.write(str(stats) + "\n\n")
            
            # Визуализация
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            for i, col in enumerate(activity_cols):
                full_col_name = f'last_hour_{col}'
                row = i // 3
                col_idx = i % 3
                
                # Box plot
                df.boxplot(column=full_col_name, by='is_fraud', ax=axes[row, col_idx])
                axes[row, col_idx].set_title(f'{col} по типу транзакции')
                axes[row, col_idx].set_xlabel('Тип транзакции')
                axes[row, col_idx].set_ylabel(col)
            
            # Убираем лишний subplot
            axes[1, 2].set_visible(False)
            
            plt.tight_layout()
            plt.savefig('plots/activity_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()

def correlation_analysis(df, output_file):
    """Анализ корреляций"""
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write("АНАЛИЗ КОРРЕЛЯЦИЙ\n")
        f.write("=" * 50 + "\n")
        
        # Подготовка данных для корреляционного анализа
        # Выбираем числовые столбцы
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Добавляем булевы столбцы
        bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
        
        # Создаем датафрейм для корреляционного анализа
        corr_df = df[numeric_cols + bool_cols].copy()
        
        # Преобразуем булевы в числовые
        for col in bool_cols:
            corr_df[col] = corr_df[col].astype(int)
        
        # Вычисляем корреляции
        correlation_matrix = corr_df.corr()
        
        f.write("Корреляционная матрица:\n")
        f.write(str(correlation_matrix) + "\n\n")
        
        # Корреляции с целевой переменной
        fraud_correlations = correlation_matrix['is_fraud'].sort_values(ascending=False)
        f.write("Корреляции с целевой переменной (is_fraud):\n")
        f.write(str(fraud_correlations) + "\n\n")
        
        # Визуализация корреляционной матрицы
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Корреляционная матрица')
        plt.tight_layout()
        plt.savefig('plots/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Визуализация корреляций с целевой переменной
        plt.figure(figsize=(10, 8))
        fraud_correlations.plot(kind='bar')
        plt.title('Корреляции с целевой переменной (is_fraud)')
        plt.xlabel('Признаки')
        plt.ylabel('Корреляция')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('plots/fraud_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()

def summary_statistics(df, output_file):
    """Сводная статистика"""
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write("СВОДНАЯ СТАТИСТИКА\n")
        f.write("=" * 50 + "\n")
        
        f.write("Общая статистика:\n")
        f.write(f"Общее количество транзакций: {len(df):,}\n")
        f.write(f"Количество уникальных клиентов: {df['customer_id'].nunique():,}\n")
        f.write(f"Количество уникальных карт: {df['card_number'].nunique():,}\n")
        f.write(f"Количество уникальных вендоров: {df['vendor'].nunique():,}\n")
        f.write(f"Количество стран: {df['country'].nunique():,}\n")
        f.write(f"Количество городов: {df['city'].nunique():,}\n")
        f.write(f"Количество валют: {df['currency'].nunique():,}\n\n")
        
        f.write("Статистика по суммам:\n")
        f.write(f"Минимальная сумма: {df['amount'].min():.2f}\n")
        f.write(f"Максимальная сумма: {df['amount'].max():.2f}\n")
        f.write(f"Средняя сумма: {df['amount'].mean():.2f}\n")
        f.write(f"Медианная сумма: {df['amount'].median():.2f}\n")
        f.write(f"Стандартное отклонение: {df['amount'].std():.2f}\n\n")
        
        f.write("Временной период:\n")
        f.write(f"Начало периода: {df['timestamp'].min()}\n")
        f.write(f"Конец периода: {df['timestamp'].max()}\n")
        f.write(f"Продолжительность: {df['timestamp'].max() - df['timestamp'].min()}\n\n")

def main():
    """Основная функция"""
    print("Начинаем анализ данных...")
    
    # Загрузка данных
    transactions_df, currency_df = load_data()
    
    # Создание файла для вывода результатов
    output_file = 'eda_results.txt'
    
    # Выполнение анализа
    print("1. Базовая информация...")
    basic_info(transactions_df, currency_df, output_file)
    
    print("2. Анализ пропущенных значений...")
    missing_values_analysis(transactions_df, output_file)
    
    print("3. Анализ мошеннических транзакций...")
    fraud_analysis(transactions_df, output_file)
    
    print("4. Временной анализ...")
    temporal_analysis(transactions_df, output_file)
    
    print("5. Анализ сумм транзакций...")
    amount_analysis(transactions_df, output_file)
    
    print("6. Анализ категориальных переменных...")
    categorical_analysis(transactions_df, output_file)
    
    print("7. Географический анализ...")
    geographical_analysis(transactions_df, output_file)
    
    print("8. Анализ активности...")
    activity_analysis(transactions_df, output_file)
    
    print("9. Анализ корреляций...")
    correlation_analysis(transactions_df, output_file)
    
    print("10. Сводная статистика...")
    summary_statistics(transactions_df, output_file)
    
    print(f"\nАнализ завершен!")
    print(f"Результаты сохранены в файл: {output_file}")
    print(f"Графики сохранены в папку: plots/")

if __name__ == "__main__":
    main() 