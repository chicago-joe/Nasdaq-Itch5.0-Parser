###########################################################################
#       Parsing NASDAQ ITCH-5.0 Trade Data
#
#                       Created by Joseph Loss on 09/01/2019
#                           MS Financial Engineering
#                     University of Illinois, Urbana-Champaign
#
#                          contact: loss2@illinois.edu
#
###########################################################################
import Cython
import numpy
import pyximport
pyximport.install(setup_args={"script_args":["--compiler=mingw32"],
                              "include_dirs":numpy.get_include()}, reload_support=True,language_level = 3)
import gzip
import shutil
from pathlib import Path
from urllib.request import urlretrieve
import urllib.parse

from clint.textui import progress
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from struct import unpack
from collections import namedtuple, Counter
from datetime import timedelta
from time import time


# store data download in a subdirectory and convert the result to hdf format (for faster read/write)
data_path = Path('C://Users//jloss//PyCharmProjects//NASDAQ-ITCH-5.0-VWAP-PARSER//data')
itch_store=str(data_path/'itch.h5')
order_book_store = data_path/'order_book.h5'

# give FTP address, filename, and the date of the file you want to download:
FTP_URL = 'ftp://emi.nasdaq.com/ITCH/Nasdaq_ITCH/'
SOURCE_FILE = '01302019.NASDAQ_ITCH50.gz'

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

# put parser format strings (given in ITCH5.0 pdf) into format dictionaries within class:
class format_strings:
    event_codes = {'O': 'Start of Messages',
                   'S': 'Start of System Hours',
                   'Q': 'Start of Market Hours',
                   'M': 'End of Market Hours',
                   'E': 'End of System Hours',
                   'C': 'End of Messages'}
    encoding = {'primary_market_maker': {'Y': 1, 'N': 0},
                'printable'           : {'Y': 1, 'N': 0},
                'buy_sell_indicator'  : {'B': 1, 'S': -1},
                'cross_type'          : {'O': 0, 'C': 1, 'H': 2},
                'imbalance_direction' : {'B': 0, 'S': 1, 'N': 0, 'O': -1}}
    formats = {
        ('integer', 2): 'H',
        ('integer', 4): 'I',
        ('integer', 6): '6s',
        ('integer', 8): 'Q',
        ('alpha', 1)  : 's',
        ('alpha', 2)  : '2s',
        ('alpha', 4)  : '4s',
        ('alpha', 8)  : '8s',
        ('price_4', 4): 'I',
        ('price_8', 8): 'Q',
    }

# the message_types.xlxs contains type specifications as laid out by ITCH documentation:
message_data = (pd.read_excel('C://Users//jloss//PyCharmProjects//NASDAQ-ITCH-5.0-VWAP-PARSER//src//message_types.xlsx',
                              sheet_name='messages',
                              encoding='latin1').sort_values('id').drop('id', axis=1))

# basic string cleaning:
def clean_message_types(df):
    df.columns = [col.lower().strip() for col in df.columns]
    df.value = df.value.str.strip()
    df.name = (df.name
               .str.strip() # remove whitespace
               .str.lower()
               .str.replace(' ', '_')
               .str.replace('-', '_')
               .str.replace('/', '_'))
    df.notes = df.notes.str.strip()
    df['message_type'] = df.loc[df.name == 'message_type', 'value']
    return df

# read message types from xlsx and run string cleaning function
message_types = clean_message_types(message_data)

# extract message type codes/names to make results more readable
message_labels = (message_types.loc[:, ['message_type', 'notes']]
                  .dropna()
                  .rename(columns={'notes': 'name'}))
message_labels.name = (message_labels.name
                       .str.lower()
                       .str.replace('message', '')
                       .str.replace('.', '')
                       .str.strip().str.replace(' ', '_'))
print(message_labels.head())

# finalize msg specs: offset, length, and value type to be used by struct
message_types.message_type = message_types.message_type.ffill()
message_types = message_types[message_types.name != 'message_type']
message_types.value = (message_types.value
                       .str.lower()
                       .str.replace(' ', '_')
                       .str.replace('(', '')
                       .str.replace(')', ''))

print('\n\n')
print(message_types.info())
print('\n\n')
print(message_types.head(1))

# persist from file:
message_types.to_csv('message_types.csv', index=False)
message_types = pd.read_csv('message_types.csv')

# parser translates the message specs into format strings and tuples that capture msg content
# so we create formatting tuples from ITCH specifications:
message_types.loc[:, 'formats'] = (message_types[['value', 'length']].apply(tuple, axis=1).map(format_strings.formats))

# and extract formatting details for alphanumeric fields
alpha_fields = message_types[message_types.value == 'alpha'].set_index('name')
alpha_msgs = alpha_fields.groupby('message_type')
alpha_formats = { k:v.to_dict() for k, v in alpha_msgs.formats }  # use formats dictionary for fields marked "alpha"
alpha_length = { k:v.add(5).to_dict() for k, v in alpha_msgs.length }

# generate msg classes as named tuples and format strings
message_fields, fstring = { }, { }
for t, message in message_types.groupby('message_type'):
    message_fields[t] = namedtuple(typename = t, field_names = message.name.tolist())
    fstring[t] = '>' + ''.join(message.formats.tolist())


# post-processing for alphanumeric fields
def format_alphanumeric(mtype, data):
    # process byte strings of alpha type
    for col in alpha_formats.get(mtype).keys():
        if mtype != 'R' and col == 'stock':     # save memory, we already have stock locator so we can delete the stock ticker name
            data = data.drop(col, axis=1)
            continue
        data.loc[:, col] = data.loc[:, col].str.decode("utf-8").str.strip()
        if format_strings.encoding.get(col):
            data.loc[:, col] = data.loc[:, col].map(format_strings.encoding.get(col))
    return data

# handles occasional storing of messages
def store_messages(m):
    """Handle occasional storing of all messages"""
    with pd.HDFStore(itch_store) as store:
        for mtype, data in m.items():
            # convert to DataFrame
            data = pd.DataFrame(data)

            # parse timestamp info
            data.timestamp = data.timestamp.apply(int.from_bytes, byteorder='big')
            data.timestamp = pd.to_timedelta(data.timestamp)

            # apply alpha formatting
            if mtype in alpha_formats.keys():
                data = format_alphanumeric(mtype, data)

            s = alpha_length.get(mtype)
            if s:
                s = {c: s.get(c) for c in data.columns}
            dc = ['stock_locate']
            if m == 'R':
                dc.append('stock')
            store.put(mtype, data,
                      format = 't',
                      min_itemsize = s,
                      data_columns = dc)

messages = {}
message_count = 0
message_type_counter = Counter()


# simplified version to process the bin file and produce the parsed orders by msg type:
start = time()
with file_name.open('rb') as data:
    while True:
        # determine msg size in bytes
        message_size = int.from_bytes(data.read(2), byteorder = 'big', signed = False)

        # get msg type by reading first byte
        message_type = data.read(1).decode('ascii')

        # create data structure to capture result
        if not messages.get(message_type):
            messages[message_type] = []

        message_type_counter.update([message_type])

        # read & store msg
        record = data.read(message_size - 1)
        message = message_fields[message_type]._make(unpack(fstring[message_type], record))
        messages[message_type].append(message)

        # system event handler:
        if message_type == 'S':
            timestamp = int.from_bytes(message.timestamp, byteorder = 'big')
            print(format_strings.event_codes.get(message.event_code.decode('ascii'), 'Error'))
            print('\t{0}\t{1:,.0f}'.format(timedelta(seconds=timestamp * 1e-9), message_count))
            if message.event_code.decode('ascii') == 'C':
                store_messages(messages)
                break

        message_count += 1
        if message_count % 2.5e7 == 0:
            timestamp = int.from_bytes(message.timestamp, byteorder = 'big')
            print('\t{0}\t{1:,.0f}\t{2}'.format(timedelta(seconds = timestamp * 1e-9),
                                                message_count,
                                                timedelta(seconds = time() - start)))
            store_messages(messages)
            messages = {}


print(timedelta(seconds=time() - start))

