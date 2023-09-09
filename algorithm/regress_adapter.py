import regress
import matplotlib.pyplot as plt

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import mysql.connector
from vnstock import *

def get_news_date(date_start, date_end):
    # Establish a connection to the MySQL database
    connection = mysql.connector.connect(
        host='127.0.0.1',
        port=13306,
        user='root',
        password='root',
        database='pyml'
    )

    # Read the table data using pandas
    query = f"""
        SELECT title, content, date FROM crawl_data
        where date >= '{date_start}' and date <= '{date_end}'
    """
    df = pd.read_sql(query, connection)
    return df

# Lấy data tin tức từ ngày {start} đến ngày {end} và merge với giá stock thành dataframe.
# Return: dataframe with 3 columns: text, close, date
def get_data(stock, date_start, date_end, export_url=None):
    df = get_news_date(date_start, date_end)
    # Concatenate columns A and B vertically
    df['text'] = df['title'] + df['content']
    df['date_only'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S').dt.strftime('%Y-%m-%d')


    df_his = stock_historical_data(stock, date_start, date_end, "1D", 'stock')
    df_his['date'] = pd.to_datetime(df_his['time']).dt.strftime('%Y-%m-%d')


    dfMerge = pd.merge(df, df_his, left_on=['date_only'], right_on=['date'], how='inner')
    dfSumarize = dfMerge[['text', 'close', 'date_y']]

    # Sorting the DataFrame by the 'date_column' in ascending order

    df_sorted = dfSumarize.sort_values(by='date_y', ascending=True)
    if export_url:
        df_sorted.to_csv(export_url, index=True)

    return df_sorted


class BotRegress:
    modelClass = None

    """ 
        stock: mã cổ phiếu
        date_start: ngày bắt đầu training
        date_end: ngày kết thúc training
        transform_type: loại transform
        @param: algorithm, loại algorithm regression
    """
    def __init__(self, stock, date_start, date_end, transform_type='tfidf', algorithm='randomforest'):
        self.stock = stock
        self.date_start = date_start
        self.date_end = date_end
        self.df_data = get_data(stock, date_start, date_end)
        self.x_train_raw = self.df_data['text']
        self.y_train_raw = self.df_data['close']
        self.transform_type = transform_type
        self.algorithm = algorithm

    def fit(self, cache=False):
        self.modelClass = regress.RegressionTextToPrice(transform_type=self.transform_type, algorithm=self.algorithm)
        self.modelClass.fit(self.x_train_raw, self.y_train_raw, cache=cache)

    """
        Predict stock price by date
        Return: dataframe with 2 columns: date, predict
    """
    def predict_by_date(self, stock, date_start, date_end):
        df_data = get_data(stock, date_start, date_end)

        x_train_raw = df_data['text']
        y_train_raw = df_data['close']

        self.modelClass.fit(x_train_raw, y_train_raw, cache=True)
        df_news = get_news_date(date_start, date_end)
        x_test_raw = df_news['title']
        pred_t = self.modelClass.predict(x_test_raw)
        df_news['predict'] = pred_t
        df_news['date_formated'] = df_news['date'].dt.strftime('%Y-%m-%d')
        # Group by the 'Date' column and calculate the average for each date
        result = df_news.groupby('date_formated')['predict'].agg('mean').reset_index()
        result = result.rename(columns={'date_formated': 'date'})
        return result

    @staticmethod
    def draw_chart(df, title):
        plt.figure(figsize=(20,10))
        plt.plot(df['date'], df['actual'], label='Actual')
        plt.plot(df['date'], df['predict'], label='Predict')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(title)
        plt.legend()
        plt.show()

    def predict_chart(self, stock, date_start, date_end):
        df_his = stock_historical_data(stock, date_start, date_end, "1D", 'stock')
        df_his['date'] = pd.to_datetime(df_his['time']).dt.strftime('%Y-%m-%d')
        df_his = df_his[['date', 'close']]
        df_his = df_his.rename(columns={'close': 'actual'})
        df_predict = self.predict_by_date(stock, date_start, date_end)
        df_merge = pd.merge(df_his, df_predict, left_on=['date'], right_on=['date'], how='inner')
        df_merge = df_merge[['date', 'actual', 'predict']]
        self.draw_chart(df_merge, stock)
