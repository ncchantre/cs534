from fastai.text.all import *
import pandas as pd
from multiprocessing import Process, freeze_support, set_start_method

def ulmfit_train():
    df = pd.read_csv('D:/reddit/RS_2022_random_sample_labels.csv',
                     encoding = "ISO-8859-1",
                     names = ['id', 'title', 'body', 'sentiment'])
    dls = TextDataLoaders.from_df(df, valid_pct=0.2, text_col='body', label_col='sentiment')
    learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
    learn.fine_tune(1, 0.001)
    learn.validate()
    #learn.save('ULMFiT_fine_tuned')

if __name__ == '__main__':
    freeze_support()
    set_start_method('spawn')
    p = Process(target=ulmfit_train)
    p.start()
