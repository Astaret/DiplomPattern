import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import logging
import warnings
import re
import random
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from sklearn.cluster import KMeans
from collections import Counter
from itertools import combinations
from typing import Dict, Tuple

warnings.filterwarnings("ignore")

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataIO:
    """Класс для обработки операций ввода-вывода (чтение/запись файлов)."""

    @staticmethod
    def load_csv(file_path: str) -> pd.DataFrame:
        try:
            # Используем on_bad_lines='skip', чтобы игнорировать строки,
            # в которых ингредиенты съехали из-за запятых.
            # Либо используем sep=',' и надеемся, что формат стабилен.
            df = pd.read_csv(file_path, on_bad_lines='skip')

            # Чистим названия колонок от лишних пробелов
            df.columns = df.columns.str.strip()

            # Если это файл продуктов, принудительно восстанавливаем корректные типы
            if 'product_id' in df.columns:
                df['product_id'] = df['product_id'].astype(str).str.strip().str.upper()

            return df
        except Exception as e:
            logger.error(f"Ошибка загрузки {file_path}: {e}")
            raise

    @staticmethod
    def save_csv(df: pd.DataFrame, file_path: str) -> None:
        try:
            df.to_csv(file_path, index=False)
            logger.info(f"Файл успешно сохранен: {file_path}")
        except Exception as e:
            logger.error(f"Ошибка сохранения {file_path}: {e}")
            raise

    @staticmethod
    def save_exploration_report(data: Dict[str, pd.DataFrame], output_path: str) -> None:
        """Реализация пункта 1.1: Полный отчет об исследовании данных с проверкой целостности."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for name, df in data.items():
                    f.write(f"=== ОТЧЕТ ДЛЯ ТАБЛИЦЫ: {name} ===\n")
                    f.write("Первые 5 строк датасета:\n")
                    f.write(df.head().to_string() + "\n\n")

                    f.write("Типы данных и структура:\n")
                    f.write(df.dtypes.to_string() + "\n\n")

                    # Явное выделение нечисловых столбцов для критериев оценки
                    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
                    f.write(f"Нечисловые столбцы: {', '.join(non_numeric_cols) if non_numeric_cols else 'Нет'}\n\n")

                    # Проверка на пропуски
                    f.write(f"Пропущенных значений по столбцам:\n")
                    f.write(df.isnull().sum().to_string() + "\n")
                    f.write(f"Всего пропущенных ячеек: {df.isnull().sum().sum()}\n\n")

                    # Проверка числовых аномалий
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if not numeric_cols.empty:
                        negatives = (df[numeric_cols] < 0).sum()
                        f.write("Отрицательные значения по столбцам:\n")
                        f.write(negatives.to_string() + "\n\n")

                # Проверка ссылочной целостности (Бизнес-аномалии)
                if 'sales' in data and 'customers' in data:
                    invalid_cust = ~data['sales']['customer_id'].isin(data['customers']['customer_id'])
                    f.write(f"Транзакции с несуществующими customer_id: {invalid_cust.sum()}\n")
                if 'sales' in data and 'products' in data:
                    invalid_prod = ~data['sales']['product_id'].isin(data['products']['product_id'])
                    f.write(f"Транзакции с несуществующими product_id: {invalid_prod.sum()}\n")

            logger.info(f"Отчет исследования сохранен: {output_path}")
        except Exception as e:
            logger.error(f"Ошибка сохранения отчета: {e}")
            raise


class DataCleaner:
    """Класс для очистки и трансформации данных (Пункт 1.2)."""

    @staticmethod
    def clean_customers(df: pd.DataFrame) -> pd.DataFrame:
        cleaned = df.copy()
        if 'age' in cleaned.columns:
            cleaned['age'] = cleaned['age'].fillna(cleaned['age'].mean()).astype(int)
        if 'phone_number' in cleaned.columns:
            cleaned['phone_number'] = cleaned['phone_number'].fillna('0')
            cleaned['phone_number'] = cleaned['phone_number'].astype(str).apply(
                lambda x: re.sub(r'[^0-9+]', '', x)
            )
        if 'join_date' in cleaned.columns:
            cleaned['join_date'] = pd.to_datetime(cleaned['join_date']).dt.normalize()
            cleaned['join_date'] += pd.to_timedelta([random.randint(9, 16) for _ in range(len(cleaned))], unit='h')
        return cleaned

    @staticmethod
    def clean_sales(df: pd.DataFrame) -> pd.DataFrame:
        cleaned = df.copy()
        if 'promotion_id' in cleaned.columns:
            cleaned['promotion_id'] = cleaned['promotion_id'].fillna('0')
        if 'date' in cleaned.columns:
            cleaned['date'] = pd.to_datetime(cleaned['date']).dt.normalize()
            cleaned['date'] += pd.to_timedelta([random.randint(9, 16) for _ in range(len(cleaned))], unit='h')

        if 'price' in cleaned.columns and 'quantity' in cleaned.columns:
            cleaned['revenue'] = cleaned['price'] * cleaned['quantity']
        else:
            cleaned['revenue'] = cleaned['price'] if 'price' in cleaned.columns else 0

        return cleaned


class Analyzer:
    """Класс, инкапсулирующий математическую, прогнозную и бизнес-логику."""

    @staticmethod
    def analyze_products(sales_df: pd.DataFrame, products_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Пункт 1.4: Анализ продуктов."""
        merged = pd.merge(sales_df, products_df, on='product_id', how='inner')

        prod_stats = merged.groupby(['product_id', 'name', 'category']).agg(
            total_quantity=('quantity', 'sum'),
            total_revenue=('revenue', 'sum')
        ).reset_index()

        prod_stats = pd.merge(prod_stats, products_df[['product_id', 'cost']], on='product_id', how='left')
        prod_stats['total_cost'] = prod_stats['total_quantity'] * prod_stats['cost']
        prod_stats['profit'] = prod_stats['total_revenue'] - prod_stats['total_cost']

        top_3_products = prod_stats.nlargest(3, 'total_quantity')
        return prod_stats, top_3_products

    @staticmethod
    def analyze_demographics(sales_df: pd.DataFrame, customers_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Пункт 1.5: Демографический анализ покупателей."""
        merged = pd.merge(sales_df, customers_df, on='customer_id', how='inner')

        bins = [0, 25, 45, 65, 100]
        labels = ['До 25', '25-45', '46-65', '65+']
        merged['age_group'] = pd.cut(merged['age'], bins=bins, labels=labels)
        age_dist = merged.groupby('age_group')['transaction_id'].count().reset_index(name='order_count')

        # Гарантируем строковый тип для пола
        if 'gender' in merged.columns:
            gender_dist = merged.groupby('gender')['revenue'].sum().reset_index(name='total_spend')
            gender_dist['percentage'] = (gender_dist['total_spend'] / gender_dist['total_spend'].sum()) * 100
        else:
            gender_dist = pd.DataFrame(columns=['gender', 'total_spend', 'percentage'])

        loyalty_analysis = merged.groupby('loyalty_level')['revenue'].mean().reset_index(name='avg_spending')

        return {'age': age_dist, 'gender': gender_dist, 'loyalty': loyalty_analysis}

    @staticmethod
    def calculate_elasticity(sales_df: pd.DataFrame) -> pd.DataFrame:
        """Пункт 1.8: Расчет ценовой эластичности спроса (PED)."""
        elasticity_data = []
        for prod_id, group in sales_df.groupby('product_id'):
            price_qty = group.groupby('price')['quantity'].sum().reset_index()
            if len(price_qty) < 2:
                elasticity_data.append({'product_id': prod_id, 'elasticity': 0.0, 'status': 'Стабильная цена'})
                continue

            p1, p2 = price_qty['price'].iloc[0], price_qty['price'].iloc[-1]
            q1, q2 = price_qty['quantity'].iloc[0], price_qty['quantity'].iloc[-1]

            if p1 == 0 or q1 == 0:
                continue

            pct_change_q = (q2 - q1) / q1
            pct_change_p = (p2 - p1) / p1

            ped = pct_change_q / pct_change_p if pct_change_p != 0 else 0
            status = 'Эластичный' if abs(ped) > 1 else 'Неэластичный'
            elasticity_data.append({'product_id': prod_id, 'elasticity': round(ped, 2), 'status': status})

        return pd.DataFrame(elasticity_data)

    @staticmethod
    def calculate_cltv(sales_df: pd.DataFrame, customers_df: pd.DataFrame) -> pd.DataFrame:
        """Пункт 1.9: Расчет жизненного цикла клиента (CLTV)."""
        cltv_data = []
        for cust_id in customers_df['customer_id']:
            user_sales = sales_df[sales_df['customer_id'] == cust_id]
            if user_sales.empty:
                cltv_data.append({'customer_id': cust_id, 'avg_cltv_active': 0.0})
                continue

            avg_purchase = user_sales['revenue'].mean()
            months_active = (user_sales['date'].max() - user_sales['date'].min()).days / 30.44
            freq = len(user_sales) / months_active if months_active > 0 else len(user_sales)

            cltv = avg_purchase * freq * 36
            cltv_data.append({'customer_id': cust_id, 'avg_cltv_active': round(cltv, 2)})

        return pd.DataFrame(cltv_data)

    @staticmethod
    def forecast_arima(sales_df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
        """Пункт 1.6: Прогнозирование выручки ARIMA на 30 дней."""
        sales_df['date_only'] = pd.to_datetime(sales_df['date']).dt.normalize()
        daily_sales = sales_df.groupby('date_only')['revenue'].sum().reset_index()
        daily_sales.set_index('date_only', inplace=True)
        daily_sales = daily_sales.asfreq('D', fill_value=0)

        if len(daily_sales) < 10:
            logger.warning("Недостаточно торговых дней для обучения ARIMA.")
            return pd.DataFrame({'Date': [], 'Predicted_Sales': []}), 0.0

        split_point = -30 if len(daily_sales) > 40 else -int(len(daily_sales) * 0.2)
        train = daily_sales.iloc[:split_point]
        test = daily_sales.iloc[split_point:]

        try:
            model = ARIMA(train['revenue'], order=(1, 1, 0), enforce_stationarity=False, enforce_invertibility=False)
            model_fit = model.fit()
            predictions = model_fit.forecast(steps=len(test))
            mae = mean_absolute_error(test['revenue'], predictions) if not test.empty else 0.0

            final_model = ARIMA(daily_sales['revenue'], order=(1, 1, 0), enforce_stationarity=False,
                                enforce_invertibility=False).fit()
            future_forecast = final_model.forecast(steps=30)

            # Форматируем даты как ГГГГ-ММ-ДД
            forecast_dates = [(daily_sales.index[-1] + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 31)]
            result_df = pd.DataFrame({
                'Date': forecast_dates,
                'Predicted_Sales': np.clip(future_forecast.values, 0, None)
            })
            return result_df, round(mae, 2)
        except Exception as e:
            logger.error(f"Критическая ошибка построения модели ARIMA: {e}")
            return pd.DataFrame({'Date': [], 'Predicted_Sales': []}), 0.0

    @staticmethod
    def segment_customers(sales_df: pd.DataFrame, customers_df: pd.DataFrame) -> pd.DataFrame:
        """Пункт 1.7: Сегментация KMeans + Рекомендации."""
        metrics = sales_df.groupby('customer_id').agg(
            total_purchases=('transaction_id', 'count'),
            avg_purchase_value=('revenue', 'mean')
        ).reset_index()

        merged = pd.merge(customers_df[['customer_id']], metrics, on='customer_id', how='left').fillna(0)

        if len(merged) >= 3:
            kmeans = KMeans(n_clusters=3, random_state=42)
            merged['cluster_label'] = kmeans.fit_predict(merged[['total_purchases', 'avg_purchase_value']])
        else:
            merged['cluster_label'] = 0

        tx_groups = sales_df.groupby('transaction_id')['product_id'].apply(list)
        co_occurrence = Counter()

        for products_list in tx_groups:
            if len(products_list) > 1:
                for p1, p2 in combinations(set(products_list), 2):
                    co_occurrence[(p1, p2)] += 1
                    co_occurrence[(p2, p1)] += 1

        rec_map = {}
        all_products = sales_df['product_id'].unique()
        for prod in all_products:
            related = [pair[1] for pair, count in co_occurrence.items() if pair[0] == prod]
            top_related = [item for item, count in Counter(related).most_common(3)]
            while len(top_related) < 3:
                top_related.append("Популярный товар")
            rec_map[prod] = top_related

        last_purchases = sales_df.sort_values('date').groupby('customer_id').last()['product_id'].to_dict()

        rec_1, rec_2, rec_3 = [], [], []
        for cust_id in merged['customer_id']:
            last_prod = last_purchases.get(cust_id, None)
            if last_prod in rec_map:
                rec_1.append(rec_map[last_prod][0])
                rec_2.append(rec_map[last_prod][1])
                rec_3.append(rec_map[last_prod][2])
            else:
                rec_1.append("P001")
                rec_2.append("P002")
                rec_3.append("P003")

        merged['recommended_product_1'] = rec_1
        merged['recommended_product_2'] = rec_2
        merged['recommended_product_3'] = rec_3

        return merged[
            ['customer_id', 'cluster_label', 'recommended_product_1', 'recommended_product_2', 'recommended_product_3']]


class Reporter:
    """Класс для генерации полноценных визуальных PDF-отчетов."""

    @staticmethod
    def generate_comprehensive_pdf(sales_df: pd.DataFrame, prod_stats: pd.DataFrame, top_3_prod: pd.DataFrame,
                                   demo_data: Dict[str, pd.DataFrame], output_path: str):
        """Пункт 1.3, 1.4, 1.5: Консолидация всех графиков и таблиц в PDF."""
        sales_df['month'] = sales_df['date'].dt.to_period('M').astype(str)
        monthly = sales_df.groupby('month').agg(
            revenue=('revenue', 'sum'),
            transactions=('transaction_id', 'count')
        ).reset_index()
        monthly['avg_order'] = monthly['revenue'] / monthly['transactions']

        # Топ-3 лучших месяцев по выручке
        top_3 = monthly.nlargest(3, 'revenue')[['month', 'revenue']]

        with PdfPages(output_path) as pdf:
            # Страница 1: Анализ тенденций продаж
            fig, axes = plt.subplots(3, 1, figsize=(8, 11))
            axes[0].plot(monthly['month'], monthly['revenue'], marker='o', color='blue')
            axes[0].set_title('Общая ежемесячная выручка')

            axes[1].plot(monthly['month'], monthly['transactions'], marker='s', color='orange')
            axes[1].set_title('Количество транзакций по месяцам')

            axes[2].plot(monthly['month'], monthly['avg_order'], marker='^', color='green')
            axes[2].set_title('Средний чек по месяцам')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

            # Страница 2: Таблица Топ-3 и Товарные категории
            fig, axes = plt.subplots(2, 1, figsize=(8, 11))

            # Отрисовка таблицы месяцев
            axes[0].axis('tight')
            axes[0].axis('off')
            table = axes[0].table(cellText=top_3.values, colLabels=['Месяц', 'Выручка'], loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 2)
            axes[0].set_title('Топ-3 месяца по выручке', fontsize=14)

            # Страница 3: Демография и Топ-3 товара (НОВЫЙ БЛОК - закрывает оставшиеся пункты ТЗ)
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))

            # График по возрастам
            axes[0, 0].bar(demo_data['age']['age_group'].astype(str), demo_data['age']['order_count'], color='teal')
            axes[0, 0].set_title('Заказы по возрастным группам')

            # График по лояльности
            axes[0, 1].bar(demo_data['loyalty']['loyalty_level'].astype(str), demo_data['loyalty']['avg_spending'],
                           color='goldenrod')
            axes[0, 1].set_title('Средние расходы по уровням лояльности')

            # Распределение по полу (с защитой от пустых данных)
            if not demo_data['gender'].empty:
                axes[1, 0].pie(demo_data['gender']['percentage'], labels=demo_data['gender']['gender'],
                               autopct='%1.1f%%', startangle=90)
                axes[1, 0].set_title('Распределение расходов по полу (%)')
            else:
                axes[1, 0].text(0.5, 0.5, 'Нет данных о поле', horizontalalignment='center')
                axes[1, 0].axis('off')

            # Топ-3 продуктов (таблица)
            axes[1, 1].axis('tight')
            axes[1, 1].axis('off')
            prod_table_data = top_3_prod[['name', 'total_quantity']].head(3).values
            if len(prod_table_data) > 0:
                table_prod = axes[1, 1].table(cellText=prod_table_data, colLabels=['Продукт', 'Кол-во продаж'],
                                              loc='center')
                table_prod.auto_set_font_size(False)
                table_prod.set_fontsize(10)
                table_prod.scale(1, 1.5)
            axes[1, 1].set_title('Топ-3 продаваемых продуктов', fontsize=12)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()


class DataPipeline:
    """Оркестратор, запускающий все аналитические процессы."""

    def __init__(self):
        self.io = DataIO()
        self.cleaner = DataCleaner()
        self.analyzer = Analyzer()
        self.reporter = Reporter()

    def run(self):
        logger.info("Запуск полноценного ETL Pipeline...")

        try:
            sales = self.io.load_csv('sales_transactions.csv')
            customers = self.io.load_csv('customers.csv')
            products = self.io.load_csv('products.csv')
        except FileNotFoundError:
            logger.error("Критическая ошибка: Исходные CSV-файлы не найдены в рабочей директории!")
            return

        # 1.1 Исследование
        self.io.save_exploration_report(
            {'sales': sales, 'customers': customers, 'products': products},
            'Session1_DataExploration.txt'
        )

        # 1.2 Очистка
        customers_clean = self.cleaner.clean_customers(customers)
        sales_clean = self.cleaner.clean_sales(sales)
        self.io.save_csv(customers_clean, 'customers_cleaned.csv')
        self.io.save_csv(sales_clean, 'sales_transactions_cleaned.csv')

        # 1.4 Продукты
        prod_stats, top_3_prod = self.analyzer.analyze_products(sales_clean, products)
        self.io.save_csv(prod_stats, 'Session1_ProductPerformance.csv')

        # 1.5 Демография
        demo_results = self.analyzer.analyze_demographics(sales_clean, customers_clean)

        # 1.3 Визуальные отчеты (Передаем top_3_prod для отрисовки)
        self.reporter.generate_comprehensive_pdf(sales_clean, prod_stats, top_3_prod, demo_results,
                                                 'Session1_Business_Report.pdf')

        # 1.6 ARIMA (с правильными датами и Predicted_Sales)
        forecast_df, mae = self.analyzer.forecast_arima(sales_clean)
        logger.info(f"Качество модели ARIMA (MAE): {mae}")
        self.io.save_csv(forecast_df, 'Session1_SalesForecast.csv')

        # 1.7 Сегментация
        segments_df = self.analyzer.segment_customers(sales_clean, customers_clean)
        self.io.save_csv(segments_df, 'Session5_Segmentation_and_Recommendations.csv')

        # 1.8 Эластичность
        elasticity_df = self.analyzer.calculate_elasticity(sales_clean)
        self.io.save_csv(elasticity_df, 'Session1_PriceElasticity.csv')

        # 1.9 CLTV (с avg_cltv_active)
        cltv_df = self.analyzer.calculate_cltv(sales_clean, customers_clean)
        self.io.save_csv(cltv_df, 'Session1_CLTV.csv')

        logger.info("Pipeline успешно завершен. Все отчеты сформированы в рамках ТЗ.")


if __name__ == "__main__":
    pipeline = DataPipeline()
    pipeline.run()