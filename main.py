import logging
import random
import re
import warnings
from collections import Counter
from datetime import timedelta
from itertools import combinations
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataIO:

    @staticmethod
    def load_csv(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
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
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for name, df in data.items():
                    f.write(f"=== ОТЧЕТ ДЛЯ ТАБЛИЦЫ: {name} ===\n")
                    f.write("Первые 5 строк датасета:\n")
                    f.write(df.head().to_string() + "\n\n")

                    f.write("Типы данных и структура:\n")
                    f.write(df.dtypes.to_string() + "\n\n")

                    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
                    f.write(f"Нечисловые столбцы: {', '.join(non_numeric_cols) if non_numeric_cols else 'Нет'}\n\n")

                    f.write(f"Пропущенных значений по столбцам:\n")
                    f.write(df.isnull().sum().to_string() + "\n")
                    f.write(f"Всего пропущенных ячеек: {df.isnull().sum().sum()}\n\n")

                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if not numeric_cols.empty:
                        negatives = (df[numeric_cols] < 0).sum()
                        f.write("Отрицательные значения по столбцам:\n")
                        f.write(negatives.to_string() + "\n\n")

                if 'sales' in data and 'customers' in data:
                    invalid_cust = ~data['sales']['customer_id'].isin(data['customers']['customer_id'])
                    f.write(f"Транзакции с несуществующими customer_id: {invalid_cust.sum()}\n")

                if 'sales' in data and 'products' in data:
                    invalid_prod = ~data['sales']['product_id'].isin(data['products']['product_id'])
                    f.write(f"Транзакции с несуществующими product_id: {invalid_prod.sum()}\n")

            logger.info(f"Отчет сохранен: {output_path}, задание 1.1 Готово!")
        except Exception as e:
            logger.error(f"Ошибка сохранения отчета: {e}")
            raise


class DataCleaner:

    @staticmethod
    def clean_customers(df: pd.DataFrame) -> pd.DataFrame:
        cleaned = df.copy()
        if 'customer_id' in cleaned.columns:
            cleaned['customer_id'] = cleaned['customer_id'].astype(str)
        if 'age' in cleaned.columns:
            cleaned['age'] = cleaned['age'].fillna(int(cleaned['age'].mean()))
        if 'phone_number' in cleaned.columns:
            cleaned['phone_number'] = cleaned['phone_number'].fillna('0')
            cleaned['phone_number'] = cleaned['phone_number'].astype(str).apply(
                lambda x: re.sub(r'[^0-9+]', '', x)
            )
        if 'join_date' in cleaned.columns:
            cleaned['join_date'] = pd.to_datetime(cleaned['join_date'])
            cleaned['join_date'] += pd.to_timedelta([random.randint(9, 16) for _ in range(len(cleaned))], unit='h')
        logger.info("Данные очищены, задание 1.2 Готово!")
        return cleaned

    @staticmethod
    def clean_sales(df: pd.DataFrame) -> pd.DataFrame:
        cleaned = df.copy()
        if 'product_id' in cleaned.columns:
            cleaned['product_id'] = cleaned['product_id'].astype(str)
        if 'customer_id' in cleaned.columns:
            cleaned['customer_id'] = cleaned['customer_id'].astype(str)
        if 'promotion_id' in cleaned.columns:
            cleaned['promotion_id'] = cleaned['promotion_id'].fillna('0')
        if 'date' in cleaned.columns:
            cleaned['date'] = pd.to_datetime(cleaned['date'])
            cleaned['date'] += pd.to_timedelta([random.randint(9, 16) for _ in range(len(cleaned))], unit='h')

        if 'price' in cleaned.columns and 'quantity' in cleaned.columns:
            cleaned['revenue'] = cleaned['price'] * cleaned['quantity']
        else:
            cleaned['revenue'] = cleaned.get('price', 0)

        return cleaned


class Analyzer:

    @staticmethod
    def analyze_products(sales_df: pd.DataFrame, products_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        products_df['product_id'] = products_df['product_id'].astype(str)

        merged = pd.merge(sales_df, products_df, on='product_id', how='inner')

        name_col = 'name' if 'name' in products_df.columns else (
            'product_name' if 'product_name' in products_df.columns else None)
        cat_col = 'category' if 'category' in products_df.columns else None

        group_cols = ['product_id']
        if name_col: group_cols.append(name_col)
        if cat_col: group_cols.append(cat_col)

        prod_stats = merged.groupby(group_cols).agg(
            total_quantity=('quantity', 'sum'),
            total_revenue=('revenue', 'sum')
        ).reset_index()

        if name_col and name_col != 'product_name':
            prod_stats = prod_stats.rename(columns={name_col: 'product_name'})
        if cat_col and cat_col != 'category':
            prod_stats = prod_stats.rename(columns={cat_col: 'category'})

        if 'cost' in products_df.columns:
            prod_stats = pd.merge(prod_stats, products_df[['product_id', 'cost']], on='product_id', how='left')
            prod_stats['total_cost'] = prod_stats['total_quantity'] * prod_stats['cost']
            prod_stats['profit'] = prod_stats['total_revenue'] - prod_stats['total_cost']
        else:
            prod_stats['total_cost'] = 0
            prod_stats['profit'] = prod_stats['total_revenue']

        top_3_products = prod_stats.nlargest(3, 'total_quantity')

        return prod_stats, top_3_products

    @staticmethod
    def analyze_demographics(sales_df: pd.DataFrame, customers_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        merged = pd.merge(sales_df, customers_df, on='customer_id', how='inner')

        bins = [0, 25, 45, 65, 100]
        labels = ['До 25', '25-45', '46-65', '65+']
        if 'age' in merged.columns:
            merged['age_group'] = pd.cut(merged['age'], bins=bins, labels=labels)
            age_dist = merged.groupby('age_group')['transaction_id'].count().reset_index(name='order_count')
        else:
            age_dist = pd.DataFrame(columns=['age_group', 'order_count'])

        if 'gender' in merged.columns:
            gender_dist = merged.groupby('gender')['revenue'].sum().reset_index(name='total_spend')
            gender_dist['percentage'] = (gender_dist['total_spend'] / gender_dist['total_spend'].sum()) * 100
        else:
            gender_dist = pd.DataFrame(columns=['gender', 'total_spend', 'percentage'])

        if 'loyalty_level' in merged.columns:
            loyalty_analysis = merged.groupby('loyalty_level')['revenue'].mean().reset_index(name='avg_spending')
        else:
            loyalty_analysis = pd.DataFrame(columns=['loyalty_level', 'avg_spending'])

        return {'age': age_dist, 'gender': gender_dist, 'loyalty': loyalty_analysis}

    @staticmethod
    def forecast_arima(sales_df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
        sales_df['date_only'] = pd.to_datetime(sales_df['date']).dt.normalize()

        daily_sales = sales_df.groupby('date_only')['revenue'].sum().reset_index()
        daily_sales.set_index('date_only', inplace=True)
        daily_sales.index.freq = daily_sales.index.inferred_freq

        if len(daily_sales) < 5:
            logger.warning("Слишком мало дней для ARIMA (< 5). Прогноз невозможен.")
            return pd.DataFrame({'Date': [], 'Predicted_Sales': []}), 0.0

        split_point = -30 if len(daily_sales) > 40 else -int(len(daily_sales) * 0.2)
        if split_point == 0: split_point = -1

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

            forecast_dates = [daily_sales.index[-1] + timedelta(days=i) for i in range(1, 31)]
            result_df = pd.DataFrame(
                {'Date': forecast_dates, 'Predicted_Sales': np.clip(future_forecast.values, 0, None)})
            logger.info("Задание 1.6 готово!")
            return result_df, round(mae, 2)

        except Exception as e:
            logger.error(f"Мат. ошибка ARIMA: {e}")
            return pd.DataFrame({'Date': [], 'Predicted_Sales': []}), 0.0

    @staticmethod
    def segment_customers(sales_df: pd.DataFrame, customers_df: pd.DataFrame) -> pd.DataFrame:
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
            rec_map[prod] = top_related

        top_overall = sales_df['product_id'].value_counts().head(3).index.tolist()
        while len(top_overall) < 3:
            top_overall.append("Популярный товар")

        last_purchases = sales_df.sort_values('date').groupby('customer_id').last()['product_id'].to_dict()

        rec_1, rec_2, rec_3 = [], [], []
        for cust_id in merged['customer_id']:
            last_prod = last_purchases.get(cust_id, None)

            customer_recs = []
            if last_prod in rec_map:
                customer_recs = rec_map[last_prod]

            final_recs = []
            for r in customer_recs:
                if r not in final_recs: final_recs.append(r)
            for r in top_overall:
                if len(final_recs) >= 3: break
                if r not in final_recs: final_recs.append(r)

            rec_1.append(final_recs[0])
            rec_2.append(final_recs[1])
            rec_3.append(final_recs[2])

        merged['recommended_product_1'] = rec_1
        merged['recommended_product_2'] = rec_2
        merged['recommended_product_3'] = rec_3

        logger.info("Задание 1.7 Готово!")

        return merged[
            ['customer_id', 'cluster_label', 'recommended_product_1', 'recommended_product_2', 'recommended_product_3']]

    @staticmethod
    def calculate_elasticity(sales_df: pd.DataFrame) -> pd.DataFrame:
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
        cltv_data = []
        for cust_id in customers_df['customer_id']:
            user_sales = sales_df[sales_df['customer_id'] == cust_id]
            if user_sales.empty:
                cltv_data.append({'customer_id': cust_id, 'cltv': 0.0})
                continue

            avg_purchase = user_sales['revenue'].mean() if 'revenue' in user_sales.columns else 0

            days_active = (user_sales['date'].max() - user_sales['date'].min()).days
            months_active = days_active / 30 if days_active > 0 else 1

            freq = user_sales['transaction_id'].nunique() / months_active

            cltv = avg_purchase * freq * 36
            cltv_data.append({'customer_id': cust_id, 'cltv': round(cltv, 2)})

        return pd.DataFrame(cltv_data)


class Reporter:
    @staticmethod
    def generate_sales_trends_pdf(sales_df: pd.DataFrame, output_path: str):
        sales_df['month'] = sales_df['date'].dt.to_period('M')

        monthly = sales_df.groupby('month').agg(
            revenue=('revenue', 'sum'),
            transactions=('transaction_id', 'count')
        ).reset_index()

        monthly['month'] = monthly['month'].astype(str)
        monthly['avg_order'] = monthly['revenue'] / monthly['transactions']

        top_3 = monthly.nlargest(3, 'revenue')[['month', 'revenue']]

        logger.info("Анализ завершен, задание 1.3 Готово!")

        with PdfPages(output_path) as pdf:
            fig, axes = plt.subplots(3, 1, figsize=(8, 12))

            axes[0].plot(monthly['month'], monthly['revenue'], marker='o', color='blue')
            axes[0].set_title('Общий доход по месяцам')

            axes[1].plot(monthly['month'], monthly['transactions'], marker='s', color='orange')
            axes[1].set_title('Количество транзакций по месяцам')

            axes[2].plot(monthly['month'], monthly['avg_order'], marker='^', color='green')
            axes[2].set_title('Средняя стоимость заказа')

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

            fig, ax = plt.subplots(figsize=(6, 3))
            ax.axis('tight')
            ax.axis('off')

            table = ax.table(cellText=top_3.values, colLabels=['Месяц', 'Выручка ($)'], loc='center')
            table.set_fontsize(12)
            table.scale(1, 1.5)
            ax.set_title('Топ-3 месяца по выручке', fontsize=14)

            pdf.savefig(fig)
            plt.close()

    @staticmethod
    def generate_product_report_pdf(prod_stats: pd.DataFrame, output_path: str):
        with PdfPages(output_path) as pdf:
            fig, ax = plt.subplots(figsize=(8, 6))

            if 'category' in prod_stats.columns and not prod_stats.empty:
                cat_revenue = prod_stats.groupby('category')['total_revenue'].sum().reset_index()
                ax.bar(cat_revenue['category'], cat_revenue['total_revenue'], color='purple')
                ax.set_title('Выручка по категориям товаров')
                ax.set_ylabel('Выручка')
                ax.tick_params(axis='x', rotation=45)
            else:
                ax.text(0.5, 0.5, 'Нет данных по категориям', ha='center', va='center')
                ax.axis('off')

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

    @staticmethod
    def generate_demographics_pdf(demo_data: Dict[str, pd.DataFrame], output_path: str):
        with PdfPages(output_path) as pdf:
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))

            if not demo_data['age'].empty:
                axes[0, 0].bar(demo_data['age']['age_group'].astype(str), demo_data['age']['order_count'], color='teal')
                axes[0, 0].set_title('Заказы по возрастным группам')
                axes[0, 0].tick_params(axis='x', rotation=45)

            if not demo_data['loyalty'].empty:
                axes[0, 1].bar(demo_data['loyalty']['loyalty_level'].astype(str), demo_data['loyalty']['avg_spending'],
                               color='goldenrod')
                axes[0, 1].set_title('Средние расходы по лояльности')

            if not demo_data['gender'].empty:
                axes[1, 0].pie(demo_data['gender']['percentage'], labels=demo_data['gender']['gender'],
                               autopct='%1.1f%%', startangle=90)
                axes[1, 0].set_title('Распределение расходов по полу (%)')
            else:
                axes[1, 0].text(0.5, 0.5, 'Нет данных о поле', ha='center', va='center')
                axes[1, 0].axis('off')

            axes[1, 1].axis('off')

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()


class DataPipeline:

    def __init__(self):
        self.io = DataIO()
        self.cleaner = DataCleaner()
        self.analyzer = Analyzer()
        self.reporter = Reporter()

    def run(self):
        logger.info("Запуск")

        try:
            sales = self.io.load_csv('sales_transactions.csv')
            customers = self.io.load_csv('customers.csv')
            products = self.io.load_csv('products.csv')
        except FileNotFoundError:
            logger.warning("Файлы не найдены")
            return

         #1.1
        self.io.save_exploration_report(
            {'sales': sales, 'customers': customers, 'products': products},
            'Session1_DataExploration.txt'
        )

         #1.2
        customers_clean = self.cleaner.clean_customers(customers)
        sales_clean = self.cleaner.clean_sales(sales)
        self.io.save_csv(customers_clean, 'customers_cleaned.csv')
        self.io.save_csv(sales_clean, 'sales_transactions_cleaned.csv')

         #1.3
        self.reporter.generate_sales_trends_pdf(sales_clean, 'Session1_SalesTrends.pdf')

         #1.4
        try:
            prod_stats, top_3_prod = self.analyzer.analyze_products(sales_clean, products)
            self.io.save_csv(prod_stats, 'Session1_ProductPerformance.csv')
            self.reporter.generate_product_report_pdf(prod_stats, 'Session1_CategoryRevenue.pdf')
            logger.info("задание 1.4 Готово!")
        except Exception as e:
            logger.error(f"Ошибка при выполнении пункта 1.4: {e}")

         #1.5
        try:
            demo_results = self.analyzer.analyze_demographics(sales_clean, customers_clean)
            self.reporter.generate_demographics_pdf(demo_results, 'Session1_Demographics.pdf')
            logger.info("Пункт 1.5 (Демография) успешно выполнен.")
        except Exception as e:
            logger.error(f"Ошибка при выполнении пункта 1.5: {e}")

         #1.6
        forecast_df, mae = self.analyzer.forecast_arima(sales_clean)
        logger.info(f"ARIMA MAE: {mae}")
        self.io.save_csv(forecast_df, 'Session1_SalesForecast.csv')

         #1.7
        segments_df = self.analyzer.segment_customers(sales_clean, customers_clean)
        self.io.save_csv(segments_df, 'Session5_Segmentation_and_Recommendations.csv')

         #1.8
        try:
            elasticity_df = self.analyzer.calculate_elasticity(sales_clean)
            self.io.save_csv(elasticity_df, 'Session1_PriceElasticity.csv')
            logger.info("Задание 1.8 Готово!")
        except Exception as e:
            logger.error(f"Ошибка при выполнении пункта 1.8: {e}")

         #1.9
        cltv_df = self.analyzer.calculate_cltv(sales_clean, customers_clean)
        self.io.save_csv(cltv_df, 'Session1_CLTV.csv')

        logger.info("Все готово")


if __name__ == "__main__":
    pipeline = DataPipeline()
    pipeline.run()