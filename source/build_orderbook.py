# from main import *
#from main import itch_store, date, order_book_store, pd, Counter, timedelta, time, np
import numpy as np
import pandas as pd
from collections import Counter
from datetime import timedelta
from pathlib import Path
import gzip
import shutil
from pathlib import Path
from urllib.request import urlretrieve
import urllib.parse
from time import time

FTP_URL = 'ftp://emi.nasdaq.com/ITCH/Nasdaq_ITCH/'
SOURCE_FILE = '01302019.NASDAQ_ITCH50.gz'
data_path = Path('C://Users//jloss//PyCharmProjects//NASDAQ-ITCH-5.0-VWAP-PARSER//data')
itch_store=str(data_path/'itch.h5')
order_book_store = data_path/'order_book.h5'

# download the data and unzip it
def may_be_download(url):
    """Download & unzip ITCH data if not yet available"""
    filename = data_path / url.split('/')[-1]
    if not data_path.exists():
        print('Creating directory')
        data_path.mkdir()
    if not filename.exists():
        print('Downloading...', url)
        urlretrieve(url, filename)
    unzipped = data_path / (filename.stem + '.bin')
    if not (data_path / unzipped).exists():
        print('Unzipping to', unzipped)
        with gzip.open(str(filename), 'rb') as f_in:
            with open(unzipped, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    return unzipped

file_name = may_be_download(urllib.parse.urljoin(FTP_URL, SOURCE_FILE))
date = file_name.name.split('.')[0]

# build order book flow for the given day
stock = 'GE'
order_dict = {-1: 'sell', 1: 'buy'}

# get all messages for the chosen stock
def get_messages(date, stock = stock):
    with pd.HDFStore(itch_store) as store:
        stock_locate = store.select('R', where='stock = stock').stock_locate.iloc[0]
        target = 'stock_locate = stock_locate'

        data = {}
        trading_msgs = ['A', 'F', 'E', 'C', 'X', 'D', 'U', 'P', 'Q']
        for msg in trading_msgs:
            data[msg] = store.select(msg, where = target).drop('stock_locate', axis = 1).assign(type = msg)

        # public key records in each type of message (order_ref_number, stock locate code, etc)
        order_cols = ['order_reference_number', 'buy_sell_indicator', 'shares', 'price']

        # 'A' and 'F' message types are for Add Orders (with and without unattributed orders/quotes)
        orders = pd.concat([data['A'], data['F']], sort=False, ignore_index=True).loc[:, order_cols]

        for msg in trading_msgs[2: -3]:
            data[msg] = data[msg].merge(orders, how = 'left')

        # Msg for whenever an order on the book has been cancel-replaced
        data['U'] = data['U'].merge(orders, how = 'left',
                                    right_on = 'order_reference_number',
                                    left_on = 'original_order_reference_number',
                                    suffixes = ['', '_replaced'])

        # Cross Trade messages:
        data['Q'].rename(columns = {'cross_price': 'price'}, inplace = True)

        # Order Cancel Messages:
        data['X']['shares'] = data['X']['cancelled_shares']
        data['X'] = data['X'].dropna(subset = ['price'])

        data = pd.concat([data[msg] for msg in trading_msgs],
                         ignore_index = True,
                         sort = False)

        data['date'] = pd.to_datetime(date, format = '%m%d%Y')
        data.timestamp = data['date'].add(data.timestamp)
        data = data[data.printable != 0]

        drop_cols = ['tracking_number', 'order_reference_number', 'original_order_reference_number',
                     'cross_type', 'new_order_reference_number', 'attribution', 'match_number',
                     'printable', 'date', 'cancelled_shares']
        return data.drop(drop_cols, axis = 1).sort_values('timestamp').reset_index(drop = True)


messages = get_messages(date = date)
messages.info(null_counts = True)

with pd.HDFStore(order_book_store) as store:
    key = '{}/messages'.format(stock)
    store.put(key, messages)
    print(store.info())

# combine trade orders (reconstruct successful trades, ie orders that are executed)
def get_trades(msg):
    """Combine C, E, P and Q messages into trading records"""
    trade_dict = {'executed_shares': 'shares', 'execution_price': 'price'}
    cols = ['timestamp', 'executed_shares']
    trades = pd.concat([msg.loc[msg.type == 'E', cols + ['price']].rename(columns = trade_dict),
                        msg.loc[msg.type == 'C', cols + ['execution_price']].rename(columns = trade_dict),
                        msg.loc[msg.type == 'P', ['timestamp', 'price', 'shares']],
                        msg.loc[msg.type == 'Q', ['timestamp', 'price', 'shares']].assign(cross = 1),
                        ], sort=False).dropna(subset=['price']).fillna(0)
    return trades.set_index('timestamp').sort_index().astype(int)


trades = get_trades(messages)
print(trades.info())
with pd.HDFStore(order_book_store) as store:
    store.put('{}/trades'.format(stock), trades)

# create orders - accumulate sell orders in ascending and buy orders in desc. order for given timestamps
def add_orders(orders, buysell, nlevels):
    new_order = []
    items = sorted(orders.copy().items())
    if buysell == 1:
        items = reversed(items)
    for i, (p, s) in enumerate(items, 1):
        new_order.append((p, s))
        if i == nlevels:
            break
    return orders, new_order

# save orders
def save_orders(orders, append=False):
    cols = ['price', 'shares']
    for buysell, book in orders.items():
        df = (pd.concat([pd.DataFrame(data = data, columns = cols).assign(timestamp = t)
                         for t, data in book.items()]))
        key = '{}/{}'.format(stock, order_dict[buysell])
        df.loc[:, ['price', 'shares']] = df.loc[:, ['price', 'shares']].astype(int)
        with pd.HDFStore(order_book_store) as store:
            if append:
                store.append(key, df.set_index('timestamp'), format = 't')
            else:
                store.put(key, df.set_index('timestamp'))

## iterate over all ITCH msgs to process orders/replacement orders as specified:
order_book = {-1:{}, 1:{}}
current_orders = {-1: Counter(), 1: Counter()}
message_counter = Counter()
nlevels = 100

start = time()

for msg in messages.itertuples():
    i = msg[0]
    if i % 1e5 == 0 and i > 0:
        print('{:,.0f}\t\t{}'.format(i, timedelta(seconds=time() - start)))
        save_orders(order_book, append=True)
        order_book = {-1: {}, 1: {}}
        start=time()
    if np.isnan(msg.buy_sell_indicator):
        continue
    message_counter.update(msg.type)

    buysell = msg.buy_sell_indicator
    price, shares = None, None

    if msg.type in ['A', 'F', 'U']:
        price = int(msg.price)
        shares = int(msg.shares)
        current_orders[buysell].update({price: shares})
        current_orders[buysell], new_order = add_orders(current_orders[buysell], buysell, nlevels)
        order_book[buysell][msg.timestamp] = new_order

    if msg.type in ['E', 'C', 'X', 'D', 'U']:
        if msg.type == 'U':
            if not np.isnan(msg.shares_replaced):
                price = int(msg.price_replaced)
                shares = -int(msg.shares_replaced)
        else:
            if not np.isnan(msg.price):
                price = int(msg.price)
                shares = -int(msg.shares)
        if price is not None:
            current_orders[buysell].update({price: shares})
            if current_orders[buysell][price] <= 0:
                current_orders[buysell].pop(price)
            current_orders[buysell], new_order = add_orders(current_orders[buysell], buysell, nlevels)
            order_book[buysell][msg.timestamp] = new_order


message_counter = pd.Series(message_counter)
print(message_counter)

with pd.HDFStore(order_book_store) as store:
    print(store.info())

