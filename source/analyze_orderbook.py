from main import *
from build_orderbook import *
import seaborn as sns

# analyze order book depth
with pd.HDFStore(order_book_store) as store:
    buy = store['{}/buy'.format(stock)].reset_index().drop_duplicates()
    sell = store['{}/sell'.format(stock)].reset_index().drop_duplicates()

# convert price to decimals
buy.price = buy.price.mul(1e-4)
sell.price = sell.price.mul(1e-4)

# remove outliers from book data
percentiles = [.01, .02, .1, .25, .75, .9, .98, .99]
pd.concat([buy.price.describe(percentiles=percentiles).to_frame('buy'),
           sell.price.describe(percentiles=percentiles).to_frame('sell')], axis=1)
buy = buy[buy.price > buy.price.quantile(.01)]
sell = sell[sell.price < sell.price.quantile(.99)]

# ## ANALYZE BUY-SELL ORDER DISTRIBUTION
market_open='0930'
market_close = '1600'
# fig, ax = plt.subplots(figsize=(7,5))
# hist_kws = {'linewidth': 1, 'alpha': .5}
# sns.distplot(buy.set_index('timestamp').between_time(market_open, market_close).price, ax=ax, label='Buy', kde=False, hist_kws=hist_kws)
# sns.distplot(sell.set_index('timestamp').between_time(market_open, market_close).price, ax=ax, label='Sell', kde=False, hist_kws=hist_kws)
# plt.legend(fontsize=10)
# plt.title('Limit Order Price Distribution', fontsize=14)
# ax.set_yticklabels(['{:,}'.format(int(y/1000)) for y in ax.get_yticks().tolist()])
# ax.set_xticklabels(['${:,}'.format(int(x)) for x in ax.get_xticks().tolist()])
# plt.xlabel('Price', fontsize=12)
# plt.ylabel('Shares (\'000)', fontsize=12)
# plt.tight_layout()
# # plt.savefig('figures/price_distribution', dpi=600);


## ANALYZE ORDER BOOK DEPTH
utc_offset = timedelta(hours=4)
depth = 100

buy_per_min = (buy
               .groupby([pd.Grouper(key='timestamp', freq='Min'), 'price'])
               .shares
               .sum()
               .apply(np.log)
               .to_frame('shares')
               .reset_index('price')
               .between_time(market_open, market_close)
               .groupby(level='timestamp', as_index=False, group_keys=False)
               .apply(lambda x: x.nlargest(columns='price', n=depth))
               .reset_index())
buy_per_min.timestamp = buy_per_min.timestamp.add(utc_offset).astype(int)
buy_per_min.info()

sell_per_min = (sell
                .groupby([pd.Grouper(key='timestamp', freq='Min'), 'price'])
                .shares
                .sum()
                .apply(np.log)
                .to_frame('shares')
                .reset_index('price')
                .between_time(market_open, market_close)
                .groupby(level='timestamp', as_index=False, group_keys=False)
                .apply(lambda x: x.nsmallest(columns='price', n=depth))
                .reset_index())

sell_per_min.timestamp = sell_per_min.timestamp.add(utc_offset).astype(int)
sell_per_min.info()

with pd.HDFStore(order_book_store) as store:
    trades = store['{}/trades'.format(stock)]
trades.price = trades.price.mul(1e-4)
trades = trades[trades.cross == 0].between_time(market_open, market_close)

trades_per_min = (trades
                  .resample('Min')
                  .agg({'price': 'mean', 'shares': 'sum'}))
trades_per_min.index = trades_per_min.index.to_series().add(utc_offset).astype(int)
trades_per_min.info()

# plot the evolution of  limit orders and prices throughout the day
# dark line tracks prices for executed trades during market hours
# red and blue dots indicate individual limit orders on a per-min basis
# fig, ax = plt.subplots(figsize=(7, 5))
#
# buy_per_min.plot.scatter(x='timestamp',y='price', c='shares', ax=ax, colormap='Blues', colorbar=False, alpha=.25)
# sell_per_min.plot.scatter(x='timestamp',y='price', c='shares', ax=ax, colormap='Reds', colorbar=False, alpha=.25)
# trades_per_min.price.plot(figsize=(14, 8), c='k', ax=ax, lw=2,
#                           title=f'AAPL | {date} | Buy & Sell Limit Order Book | Depth = {depth}')
#
# xticks = [datetime.fromtimestamp(ts / 1e9).strftime('%H:%M') for ts in ax.get_xticks()]
# ax.set_xticklabels(xticks)
#
# ax.set_xlabel('')
# ax.set_ylabel('Price')
#
# fig.tight_layout()
# # fig.savefig('figures/order_book', dpi=600);
#
