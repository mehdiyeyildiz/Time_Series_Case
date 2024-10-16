"""
transaction_date: Satış verilerinin tarihi
merchant_id: Üye iş yerlerinin id'leri (Her bir Üye iş yeri için eşsiz numara)
Total_Transaction: İşlem sayısı
Category: Üye iş yerlerinin kategorileri(veride yok)
Total_Paid: Ödeme miktarı

Görev 1 : Veri Setinin Keşfi
Adım 1: Iyzico_data.csv dosyasını okutunuz. transaction_date değişkeninin tipini date'e çeviriniz.
Adım 2: Veri setinin başlangıc ve bitiş tarihleri nedir?
Adım 3: Kategoriler(merchant_id kastediliyor) nelerdir?
Adım 4: Her kategoride kaç iş yeri var?
Adım 5: Her kategorideki toplam işlem sayısı kaçtır?
Adım 6: Her kategorideki toplam ödeme miktarı kaçtır?
Adım 7: Kategorilerin her bir yıl içerisindeki transaction count grafiklerini gözlemleyiniz.

Görev 2 : Feature Engineering tekniklerini uygulayanız. Yeni feature'lar türetiniz.
• Date Features
• Lag/Shifted Features
• Rolling Mean Features
• Exponentially Weighted Mean Features
• Özel günler, döviz kuru vb.

Görev 3 : Modellemeye Hazırlık ve Modelleme
Adım 1: One-hot encoding yapınız.
Adım 2: Custom Cost Function'ları tanımlayınız.
Adım 3: Veri setini train ve validation olarak ayırınız.
Adım 4: LightGBM Modelini oluşturunuz ve SMAPE ile hata değerini gözlemleyiniz.
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
warnings.filterwarnings('ignore')


df = pd.read_csv(r"C:\Users\Mehdiye\Desktop\study case\Time Series\iyzico_data.csv")
df.head()


for x in df.columns:
    print(x,"column")
    print(df[x].value_counts())

for x in df.columns:
    print(x,"column")
    print(df[x].nunique())

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

df.drop(labels="Unnamed: 0", axis=1, inplace=True)
df["transaction_date"] = pd.to_datetime(df["transaction_date"])
df["transaction_date"].min(), df["transaction_date"].max()

df["merchant_id"].unique()

df["merchant_id"].value_counts()

df.groupby(["merchant_id"]).agg({"Total_Transaction": "sum"})
df.groupby(["merchant_id"]).agg({"Total_Paid": "sum"})


#Adım 7: Kategorilerin her bir yıl içerisindeki transaction count grafiklerini gözlemleyiniz.


for id in df.merchant_id.unique():
    plt.figure(figsize=(15, 15))
    plt.subplot(3, 1, 1, title=str(id)+"2018-2019 Transaction Count")
    df[(df.merchant_id == id) & (df.transaction_date >= "2018-01-01") & (df.transaction_date < "2019-01-01")]["Total_Transaction"].plot()
    plt.xlabel("")
    plt.subplot(3, 1, 2, title=str(id) + "2019-2020 Transaction Count")
    df[(df.merchant_id == id) & (df.transaction_date >="2019-01-01") & (df.transaction_date <"2020-01-01")]["Total_Transaction"].plot()
    plt.xlabel("")
    plt.show(block=True)



#Görev 2 : Feature Engineering tekniklerini uygulayanız. Yeni feature'lar türetiniz.
#• Date Features
def create_date_features(df, date_column):
    df['month'] = df[date_column].dt.month
    df['day_of_month'] = df[date_column].dt.day
    df['day_of_year'] = df[date_column].dt.dayofyear
    df['week_of_year'] = df[date_column].dt.weekofyear
    df['day_of_week'] = df[date_column].dt.dayofweek
    df['year'] = df[date_column].dt.year
    df["is_wknd"] = df[date_column].dt.weekday // 4
    df['is_month_start'] =df[date_column].dt.is_month_start.astype(int)
    df['is_month_end'] = df[date_column].dt.is_month_end.astype(int)
    df['quarter'] = df[date_column].dt.quarter
    df['is_quarter_start'] = df[date_column].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df[date_column].dt.is_quarter_end.astype(int)
    df['is_year_start'] = df[date_column].dt.is_year_start.astype(int)
    df['is_year_end'] = df[date_column].dt.is_year_end.astype(int)
    return df
df = create_date_features(df, "transaction_date")

df.groupby(["merchant_id", "year", "month"]).agg({"Total_Transaction": ["sum", "mean", "median"]})
df.groupby(["merchant_id", "year", "month"]).agg({"Total_Paid": ["sum", "mean", "median"]})


#• Lag/Shifted Features
def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))

#çözümde kullanmamış
#df.sort_values(by=["merchant_id", "transaction_date"], axis=0).head()

def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['Total_Transaction_lag_' + str(lag)] = dataframe.groupby(["merchant_id"])['Total_Transaction'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

lags = [90, 91, 92, 179, 180, 181, 182, 183, 184, 85, 186, 365, 540, 541, 542, 543, 544]

df = lag_features(df, lags)

#• Rolling Mean Features

def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['Total_Transaction_roll_mean_' + str(window)] = dataframe.groupby(["merchant_id"])['Total_Transaction']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe

df = roll_mean_features(df, [92, 182, 365, 546])



#• Exponentially Weighted Mean Features

def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['Total_Transaction_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["merchant_id"])['Total_Transaction'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]

df = ewm_features(df, alphas, lags)

#• Özel günler, döviz kuru vb.

df["is_black_friday"] = 0
df.loc[df["transaction_date"].isin(["2018-11-22","2018-11-23","2019-11-29","2019-11-30"]), "is_black_friday"] = 1

df["is_summer_solstice"] = 0
df.loc[df["transaction_date"].isin(["2018-06-19","2018-06-20","2018-06-21","2018-06-22",
                                    "2019-06-19","2019-06-20","2019-06-21","2019-06-22",]), "is_summer_solstice"] = 1


df["valentines_day"] = 0
df.loc[df["transaction_date"].isin(["2018-02-14", "2019-02-14"]), "valentines_day"] = 1

df["mothers_day"] = 0
df.loc[df["transaction_date"].isin(["2018-05-13", "2019-05-12"]), "mothers_day"] = 1

df["fathers_day"] = 0
df.loc[df["transaction_date"].isin(["2018-06-17", "2019-06-16"]), "fathers_day"] = 1

df.head()

#Görev 3 : Modellemeye Hazırlık ve Modelleme
#Adım 1: One-hot encoding yapınız.

df = pd.get_dummies(df, columns=['merchant_id','day_of_week', 'month'])
df['Total_Transaction'] = np.log1p(df["Total_Transaction"].values)


#Adım 2: Custom Cost Function'ları tanımlayınız.

def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val

def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False

#Adım 3: Veri setini train ve validation olarak ayırınız.

train = df.loc[(df["transaction_date"]< "2020-10-01")]
val = df.loc[(df["transaction_date"] >= "2020-10-01")]

cols = [col for col in train.columns if col not in ['transaction_date', 'id', "Total_Transaction","Total_Paid", "year" ]]

X_train = train[cols]
y_train =train["Total_Transaction"]

X_val= val[cols]
y_val = val["Total_Transaction"]


y_train.shape, X_train.shape, y_val.shape, X_val.shape

#Adım 4: LightGBM Modelini oluşturunuz ve SMAPE ile hata değerini gözlemleyiniz.

lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 1000,
              'early_stopping_rounds': 200,
              'nthread': -1}


lgbtrain = lgb.Dataset(data=X_train, label=y_train, feature_name=cols)
lgbval = lgb.Dataset(data=X_val, label=y_val, feature_name=cols, reference=lgbtrain)

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  feval=lgbm_smape,
                  verbose_eval=100)

y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

smape(np.expm1(y_pred_val), np.expm1(y_val))



#değişken önem düzeyi

def plot_lgb_importances(model, plot=False, num=10):

    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))


plot_lgb_importances(model, num=30, plot=True)


