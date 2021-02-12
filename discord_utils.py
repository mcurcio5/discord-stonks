# helper functions for cleaning discord messages
import pandas as pd
import numpy as np
from scipy.stats import expon

import json
from datetime import timedelta

import yfinance as yf
import nltk
from nltk.corpus import words

def extract_tickers(s, texting_words, setofwords):
    words = s.lower().split(' ')

    return [word for word in words if len(word) < 5 and word.isalpha() and \
            word not in setofwords and word not in texting_words]

def set_up_discord_df(df, texting_words):
    setofwords = set(words.words())
    df['tickers'] = df['content'].apply(lambda x: extract_tickers(x, texting_words, setofwords))
    # take out the most frequent tickers
    all_tickers = [item for sublist in df['tickers'].to_list() for item in sublist]
    counts = pd.DataFrame(pd.Series(all_tickers).value_counts()).reset_index().rename(
        columns={'index': 'stock', 0: 'count'})
    # take out the stocks with not many mentions (this may be bad...)
    not_stocks = counts.loc[counts['count'] < 450, 'stock'].to_list()
    texting_words = set(list(texting_words) + not_stocks)
    df = df.drop(columns=['tickers'])
    df['tickers'] = df['content'].apply(lambda x: extract_tickers(x, texting_words, setofwords))
    all_tickers = [item for sublist in df['tickers'].to_list() for item in sublist]
    pd.DataFrame(pd.Series(all_tickers).value_counts()).reset_index().rename(columns={'index': 'stock', 0: 'count'})

    df['date'] = df['timestamp'].apply(lambda x: x.date())
    df['message_len'] = df['content'].apply(lambda x: len(x))
    df['n_words'] = df['content'].apply(lambda x: len(x.split(' ')))
    df['n_tickers'] = df['tickers'].apply(lambda x: len(x))
    df['n_replies'] = df['mentions'].apply(lambda x: len(x))
    df['n_reactions'] = df['reactions'].apply(lambda x: len(x))

    return df, texting_words


def clean_messages(full_json):
    messages = pd.DataFrame(full_json["messages"])
    messages['id'] = messages['id'].astype(int)
    messages['timestamp'] = messages['timestamp'].apply(
        lambda x: pd.to_datetime(x).tz_convert(tz='US/Pacific').tz_localize(None))
    messages['author_id'] = messages['author'].apply(lambda x: int(x['id']))
    messages['author_name'] = messages['author'].apply(lambda x: x['name'])
    messages['is_bot'] = messages['author'].apply(lambda x: x['isBot'])
    messages = messages.loc[messages['is_bot'] == False].reset_index(drop=True)

    return messages

# general functions
def get_expo_z_score(l, n, emperical_var):  # divide by stdev
    local_emp_vars = []
    for i in range(1000):
        r = expon.rvs(scale=1 / l, size=n)
        local_l = 1 / np.mean(r)
        local_emp_vars.append((np.mean(r ** 2) - (1 / local_l) ** 2))

    return (emperical_var - (1 / l ** 2)) / np.std(local_emp_vars)  # the z-score

def get_ratio_of_variance(times, as_timestamps=False):
    """ determine how clustered the arrival times are - finds how much higher the variance is
    than would be expected by iid expo() arrival times. gets z score. higher value -> more clusters
    """
    if len(times) < 10:
        return 0
    elif as_timestamps:
        time_diffs = np.array(
            [(times[i] - times[i - 1]).astype('timedelta64[s]').astype(np.int32) for i in range(1, len(times))])
    else:
        time_diffs = np.array([(times[i] - times[i - 1]) for i in range(1, len(times))])

    l = 1 / np.mean(time_diffs)
    emperical_var = (np.mean(time_diffs ** 2) - (1 / l) ** 2)

    return get_expo_z_score(l, len(times), emperical_var)


def get_trading_days_to_test(period='3mo'):
    trading_days = yf.Ticker('gme').history(period=period).index
    # want to get all days in the last two months where day and the next day exists
    days_to_test = []
    for day in trading_days:
        if day + timedelta(days=1) in trading_days: # and day < pd.Timestamp('2021-01-30'):
            days_to_test.append(day)

    return days_to_test


def extract_general_stats_for_day(df):
    if df.shape[0] == 0:
        return None
    tickers_df = df.loc[df['n_tickers'] > 0].reset_index(drop=True)
    non_tickers_df = df.loc[df['n_tickers'] == 0].reset_index(drop=True)

    ticker_over_non_ticker_message_len_ratio = np.log(
        np.mean(tickers_df['message_len']) / np.mean(non_tickers_df['message_len']) + 1)
    avg_stocks_mentioned = np.mean(tickers_df['n_tickers'])
    percent_ticker_messages = tickers_df.shape[0] / df.shape[0]

    ticker_over_non_ticker_replies_ratio = np.log(np.mean(tickers_df['n_replies']) + 1) - \
                                           np.log(np.mean(non_tickers_df['n_replies']) + 1)
    ticker_over_non_ticker_reactions_ratio = np.log(np.mean(tickers_df['n_reactions']) + 1) - \
                                             np.log(np.mean(non_tickers_df['n_reactions']) + 1)
    ticker_over_non_ticker_distinct_users_ratio = np.log(len(set(tickers_df['author_id'])) + 1) - \
                                                  np.log(len(set(non_tickers_df['author_id'])) + 1)

    all_tickers = [item for sublist in tickers_df['tickers'].to_list() for item in sublist]
    percent_posts_on_biggest_ticker = pd.Series(all_tickers).value_counts()[0] / len(all_tickers)

    return pd.DataFrame({'len_ratio': [ticker_over_non_ticker_message_len_ratio],
                         'avg_stocks_mentioned': [avg_stocks_mentioned],
                         'percent_ticker_messages': [percent_ticker_messages],
                         'replies_ratio': [ticker_over_non_ticker_replies_ratio],
                         'reactions_ratio': [ticker_over_non_ticker_reactions_ratio],
                         'users_ratio': [ticker_over_non_ticker_distinct_users_ratio],
                         'biggest_ticker': [percent_posts_on_biggest_ticker]})

# extracting stock specific info
def extract_ticker_specific_stats(df, ticker, column_const=''):
    if df.shape[0] == 0:
        return
    ticker = ticker.lower()

    # previous_week_swing = np.mean((previous_week_df['High'] - previous_week_df['Low']) / previous_week_df['High'])
    # swing_ratio = np.log(daily_swing / previous_week_swing)
    # extract stats from words

    ticker_df = df[df['tickers'].apply(lambda x: ticker in x)]
    ticker_df_original_indices = np.array(ticker_df.index)  # for message arrival distribution
    ticker_df = ticker_df.reset_index(drop=True)
    other_tickers_df = df[df['tickers'].apply(lambda x: (ticker not in x) and (len(x) > 0))].reset_index(drop=True)

    perc_ticker_messages = ticker_df.shape[0] / (other_tickers_df.shape[0] + ticker_df.shape[0])
    ticker_len_ratio = np.log(np.mean(ticker_df['message_len']) / np.mean(other_tickers_df['message_len']))
    avg_stocks_mentioned = np.mean(ticker_df['n_tickers'])
    avg_ticker_replies = np.mean(ticker_df['n_replies']) / (np.mean(other_tickers_df['n_replies']) + 1)
    avg_ticker_reactions = np.mean(ticker_df['n_reactions']) / (np.mean(other_tickers_df['n_reactions']) + 1)
    most_active_user = pd.Series(ticker_df['author_id']).value_counts().values[0] / ticker_df.shape[0]

    # checking how the tickers are distributed over time - ratio of actual arrival time variance / expected variance for expo
    message_arrival_series = ticker_df_original_indices + np.random.random_sample(len(ticker_df_original_indices))
    message_clustering = get_ratio_of_variance(message_arrival_series)

    df = pd.DataFrame({'perc_ticker_messages': [perc_ticker_messages], 'ticker_len_ratio': [ticker_len_ratio],
                       'avg_ticker_stocks_mentioned': [avg_stocks_mentioned],
                       'avg_ticker_replies': [avg_ticker_replies],
                       'avg_ticker_reactions': [avg_ticker_reactions], 'most_active_user': [most_active_user],
                       'message_clustering': [message_clustering]})

    df.columns = [col + column_const for col in df.columns]

    return df

def get_relevent_stocks(df, day, thresh=10):
    """ get stocks mentioend in both AM and PM """
    market_hours_df = df.loc[(df['timestamp'] >= (day + timedelta(hours=6.5))) &
                             (df['timestamp'] < (day + timedelta(hours=13)))].reset_index(drop=True)
    market_hours_tickers = [item for sublist in market_hours_df['tickers'].to_list() for item in sublist]
    after_market_df = df.loc[(df['timestamp'] >= (day + timedelta(hours=13))) &
                             (df['timestamp'] < (day + timedelta(hours=24)))].reset_index(drop=True)
    after_market_tickers = [item for sublist in after_market_df['tickers'].to_list() for item in sublist]
    # at least one in each time frame
    all_tickers = [ticker for ticker in market_hours_tickers + after_market_tickers if (
            ticker in after_market_tickers) and (ticker in market_hours_tickers)]

    counts = pd.DataFrame(pd.Series(all_tickers).value_counts()).reset_index().rename(
        columns={'index': 'stock', 0: 'count'})
    stocks = counts.loc[counts['count'] > thresh, 'stock'].to_list()
    stocks = [stock for stock in stocks if stock not in ['jmia', 'uavs', 'acb', 'ghiv']]

    return stocks


class DiscordDataset():
    def __init__(self, json_path):
        self.json_path = json_path
        self.stats_df = None
        self.stock_to_history = {}
        self.stock_info = {}

        self.texting_words = set(
            ['lol', 'guys', 'lmao', 'shit', 'fuck', 'yall', 'bruh', 'lets', 'app', 'hype', 'okay', 'boys', \
             'cuz', 'imma', 'isnt', 'tmrw', 'gets', 'says', 'ayo', 'ik', 'abt', 'omg', 'stfu', 'goin', 'info', \
             'puts', 'spam', 'mins', 'buys', 'af', 'itll', 'yup', 'xd', 'cars', 'kid', 'mf', 'meme', 'held', \
             'lil', 'whos', 'ight', 'paid', 'btw', 'mod', 'atm', 'dms', 'bois', 'wins', 'ears', 'gooo', 'jk' \
                                                                                                        'asap', 'wdym',
             'tik', 'cmon', 'jus', 'pics', 'spac', 'max', 'uh', 'gg', 'oof', 'tok', 'mods', \
             'hehe', 'rlly', 'etc', 'tm', 'bets', 'thru', 'syn', 'tbh', 'pre', 'asf', 'jk', 'mans', 'asap', \
             'im', 'rn', 'vip', 'has', 'bro', 'dm', 'mvp', 'ok', 'idk', 'nah', 'haha', 'gf', 'hmm', 'oct', \
             'wtf', 'fr', 'hits', 'bc', 'tf', 'xl', 'elon', 'ive', 'ima', 'ppl', 'td', 'wym', 'ez', \
             'ev', 'kids', 'smh', 'pls', 'wsb', 'imo', 'jan', 'uk', 'ones', 'mom', 'apps', \
             'def', 'hes', 'alot', 'ipo', 'thx', 'est', 'ty', 'plz', 'fax', 'pm', 'ngl', 'sus', \
             'ig', 'nvm', 'ceo', 'haz', 'gn', 'mars', 'yolo', 'ofc', 'lmk', 'ahh', 'pc', 'vc', \
             'hbu', 'obv', 'hola', 'upp', 'hiii', 'et', 'calc', 'porn', 'bmw', 'ipod', 'guns', 'etf', \
             'vips', 'ohh', 'idc', 'dec', 'cali', 'soo', 'usa', 'vs', 'bio', 'feb', 'np', 'iq', 'gl' \
                                                                                                'dips', 'tmr', 'spaq',
             'btc', 'xrp', 'nuts', 'pfp', 'mfs', 'abnb', 'alr', 'mexi', 'ivan', \
             'owa', 'uone', 'ipoc', 'uwmc', 'dips'])

        self.setup()

    def setup(self):
        self.df, self.texting_words = set_up_discord_df(clean_messages(json.load(open(self.json_path))), self.texting_words)

    def get_cleaned_dataset(self):
        return self.df

    def get_stocks_daily_performance(self, df, ticker):
        if df.shape[0] == 0:
            return
        date = str(df.loc[0, 'timestamp'].date())
        ticker = ticker.lower()
        # print(ticker)
        if ticker not in self.stock_to_history.keys():
            self.stock_to_history[ticker] = yf.Ticker(ticker).history(period="1Y")

        if ticker not in self.stock_info.keys():
            self.stock_info[ticker] = yf.Ticker(ticker).info

        daily_return = (self.stock_to_history[ticker].loc[date]['Close'] -
                        self.stock_to_history[ticker].loc[date]['Open']) / self.stock_to_history[ticker].loc[date]['Open']
        daily_swing = (self.stock_to_history[ticker].loc[date]['High'] -
                       self.stock_to_history[ticker].loc[date]['Low']) / self.stock_to_history[ticker].loc[date]['High']
        next_day = str(pd.Timestamp(date) + timedelta(days=1))[:10]
        next_day_return = (self.stock_to_history[ticker].loc[next_day]['Close'] -
                           self.stock_to_history[ticker].loc[next_day]['Open']) / self.stock_to_history[ticker].loc[next_day][
                              'Open']

        previous_week_df = self.stock_to_history[ticker].loc[
                str(pd.Timestamp(date) - timedelta(days=7))[:10]:str(pd.Timestamp(date) - timedelta(days=1))[:10]]

        previous_week_volume = np.mean(previous_week_df['Volume'])

        closing_price = np.log(self.stock_to_history[ticker].loc[date]['Close'] + 1)  # add one for penny stocks linearity
        try:
            market_cap = np.log(self.stock_info[ticker]['sharesOutstanding'] * closing_price)
        except:
            market_cap = pd.NA

        log_volume = np.log(self.stock_to_history[ticker].loc[date]['Volume'] + 1)
        volume_diff = log_volume - np.log(previous_week_volume + 1)

        return pd.DataFrame({'daily_return': [daily_return], 'daily_swing': [daily_swing],
                             'log_volume': [log_volume], 'volume_diff': [volume_diff],
                             'next_day_return': [next_day_return], 'closing_price': [closing_price],
                             'market_cap': [market_cap]})

    def calculate_stats_for_df(self, days_to_test = None):
        if days_to_test is None:
            days_to_test = get_trading_days_to_test()

        output = []
        for day in days_to_test:
            print('-', end='')
            df = self.df.loc[self.df['date'] == day].reset_index(drop=True)
            general_stats = extract_general_stats_for_day(df)
            if general_stats is None:
                continue
            stocks = get_relevent_stocks(df, day)
            for stock in stocks:
                daily_performance = self.get_stocks_daily_performance(df, stock)
                # get during market hours chat info
                market_hours_df = df.loc[(df['timestamp'] >= (day + timedelta(hours=6.5))) &
                                         (df['timestamp'] < (day + timedelta(hours=13)))].reset_index(drop=True)
                market_hours_stats = extract_ticker_specific_stats(market_hours_df,
                                                                   stock)  # , column_const = '_MARKET')
                # get after market hours till midnight chat info
                after_market_df = df.loc[(df['timestamp'] >= (day + timedelta(hours=13))) &
                                         (df['timestamp'] < (day + timedelta(hours=24)))].reset_index(drop=True)
                after_hours_stats = extract_ticker_specific_stats(after_market_df, stock)  # , column_const = '_AFTER')
                mean_stats = pd.concat([market_hours_stats, after_hours_stats]).groupby(level=0).mean()
                ratio_stats_cols = [col + '_RATIO' for col in market_hours_stats.columns]
                ratio_stats = pd.DataFrame(data=[list(after_hours_stats.to_numpy().flatten() -
                                                      market_hours_stats.to_numpy().flatten())],
                                           columns=ratio_stats_cols)
                output.append(pd.concat([daily_performance, mean_stats, ratio_stats], axis=1))

        df = pd.concat(output).dropna().reset_index(drop=True)
        with pd.option_context('mode.use_inf_as_null', True):
            df = df.dropna().reset_index(drop=True)

        self.stats_df = df

    def get_stats_for_df(self):
        if self.stats_df is None:
            self.calculate_stats_for_df()

        return self.stats_df