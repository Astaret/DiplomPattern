import pandas as pd

def load_and_explore_data_from_csv():
    try:
        sales = pd.read_csv('sales.csv')
        products = pd.read_csv('products.csv')
        customers = pd.read_csv('customers.csv')

        sales['product_id'] = pd.to_numeric(sales['product_id'], errors='coerce').astype('Int64')
        