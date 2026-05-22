import pandas as pd
import csv

def load_and_explore_data_from_csv():

    sales, products, customers = None, None, None

    try:
        sales = pd.read_csv('sales_transactions.csv')
        products = pd.read_csv('products.csv')
        customers = pd.read_csv('customers.csv')

        sales['product_id'] = pd.to_numeric(sales['product_id'], errors='coerce').astype('Int64')
        products['product_id'] = pd.to_numeric(products['product_id'], errors ='coerce').astype('Int64')
        customers['product_id'] = pd.to_numeric(customers['customer_id'], errors ='coerce').astype('Int64')
    except FileNotFoundError as e:
        print(f"Not found files {e}")

    if sales is not None:
        print(sales.head())
        print(products.head())
        print(customers.head())

    dtypes_report = {
        'sales': sales.dtypes.to_string(),
        'products': products.dtypes.to_string(),
        'customers': customers.dtypes.to_string()
    }

    missing_report = {
        'sales': sales.isnull().sum.to_string(),
        'products': products.isnull().sum.to_string(),
        'customers': customers.isnull().sum.to_string()}

    return sales, products, customers

