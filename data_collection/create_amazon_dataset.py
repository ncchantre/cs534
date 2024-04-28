# Loop through full Reddit CSVs and extract posts containing a key word (e.g., Amazon)
# Process data in chunks due to the large size of the CSVs
import pandas as pd

key_word = 'amazon'
months = ['01','02','03','04','05','06','07','08','09','10','11','12']

for month in months:
    df = pd.read_csv('D:/reddit/RS_2022-' + month + '.csv', chunksize=1000000)
    for chunk in df:
        chunk = chunk[chunk['text'].str.contains(key_word, na=False, case = False)]
        chunk.to_csv('D:/reddit/RS_2022_' + key_word + '.csv', mode='a', header = False)