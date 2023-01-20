import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn import cluster
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
pd.set_option('display.max_columns', 20)
data = pd.read_csv(open('D:\聚类分析\mobike.csv'), index_col=0)
print(data.info())


# 1.数据清洗
# 单位特殊字符清洗
data.replace(to_replace=r'^\s+', value=np.nan, regex=True, inplace=True)
data['tripduration'] = data['tripduration'].apply(lambda x: x.replace(',', ''))
# 时间格式转化
data['start_time'] = pd.to_datetime(data['start_time'])
data['end_time'] = pd.to_datetime(data['end_time'])
# 数据类型转化
data['age'] = data['age'].astype(float)
data['tripduration'] = data['tripduration'].astype(float)
# 去除重复样本
data.drop_duplicates(subset=None, keep='first', inplace=True)


# 2.数据预处理
# 缺失值处理,缺失率最大只有7%，考虑删除
def missing_value(df):
    missing = data.isnull().sum()/len(data)
    missing = missing[missing > 0]
    plt.title('missing')
    label = missing.index
    plt.xticks(range(len(missing)), label)  # 设置刻度和标签
    plt.bar(range(len(missing)), missing)
    plt.show()


missing_value(data)
data.dropna(inplace=True)

# 异常值处理,3sigma原则


def find_outer(df, fea):
    data_std = np.std(df[fea])
    data_mean = np.mean(df[fea])
    outliers_cut_off = data_std * 3
    lower_rule = data_mean - outliers_cut_off
    upper_rule = data_mean + outliers_cut_off
    df[fea+'_outliers'] = df[fea].apply(lambda x: str('异常值') if x > upper_rule or x < lower_rule else '正常值')
    return df


numerical_feature = ['age', 'tripduration', 'timeduration', 'birthyear']
for fea in numerical_feature:
    data = find_outer(data, fea)
    print(data[fea+'_outliers'].value_counts(normalize=True))
    print('*'*10)
    data = data[data[fea+'_outliers'] == '正常值']
    data.drop(fea+'_outliers', axis=1, inplace=True)
    data = data.reset_index(drop=True)

# 3.特征工程
# 特征编码
data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})
data['usertype'] = data['usertype'].map({'Subscriber': 1, 'Customer': 0})
data["start_time"] = data["start_time"].apply(lambda x: x.hour)
data["end_time"] = data["end_time"].apply(lambda x: x.hour)
# 特征筛选
# 删除高基数类特征，一般来说高基数类特征没有什么信息价值
data = data.drop(['user_id', 'bikeid', 'from_station_id', 'to_station_id',
                  'from_station_name', 'to_station_name'], axis=1)
# 共线性筛选，开始和结束时间之差可以得到骑行时长，删除；年龄和出生年份，骑行时长和骑行距离高度相关，各删除一个


def corr_data(df):
    data_corr = df[list(data.select_dtypes(exclude='object').columns)].corr()
    plt.figure(figsize=(8, 8), dpi=100)
    data_corr[abs(data_corr) <= 0.7] = 0
    sns.heatmap(data_corr)  # 热力图
    plt.show()


corr_data(data)
data = data.drop(['start_time', 'end_time', 'birthyear', 'tripduration'], axis=1)
# 标准化，标准差标准化
x = preprocessing.scale(data)
print(data.info())

# 4.训练与调参
# 手肘法寻找最佳聚类个数
wcss = []  # 所有点到对应簇中心的距离平方和
for i in range(1, 11):
    best_km = cluster.KMeans(n_clusters=i, random_state=2023)
    best_km.fit(x)
    wcss.append(best_km.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('the elbow method')
plt.xlabel('number of n_cluster')
plt.ylabel('wcss')
plt.show()
# 训练
model = cluster.KMeans(n_clusters=5, init="k-means++")
model.fit(x)

# 5.评估
y_pred = model.fit_predict(x)
score = silhouette_score(x, y_pred)
print('轮廓系数', score)

# 6.结果分析与可视化
data_km = data.copy()
data_km['label'] = model.labels_  # 添加标签
data_clu = pd.DataFrame(model.cluster_centers_)  # 聚类中心
# 不同分群样本量和占比
count = data_km.groupby('label')['gender'].count()
rate = count/data_km.shape[0]
r1 = pd.concat((count, rate), axis=1)
r1.columns = ['count', 'rate']
plt.pie(r1.rate,
        autopct="%3.1f%%",
        labels=r1.index)
plt.title('不同分群样本量和占比')
plt.show()

# 性别分群占比
r2 = pd.crosstab(data_km['label'], data_km['gender'])
r2.columns = ['女性', '男性']
for i in r2.index:
    plt.pie(r2.iloc[i, :], autopct="%3.1f%%", labels=r2.columns)
    plt.title('labels = %s' % i)
    plt.show()
# 用户类型分群占比
r3 = pd.crosstab(data_km['label'], data_km['usertype'])
r3.columns = ['Customer', 'Subscriber']
for i in r3.index:
    plt.pie(r3.iloc[i, :], autopct="%3.1f%%", labels=r3.columns)
    plt.title('labels = %s' % i)
    plt.show()
# 骑行时长和距离的均值情况
r4 = data_km.groupby('label')[['timeduration', 'age']].mean()
print(r4)

'''
结论：
聚类结果分为5类，轮廓系数0.47，聚类效果较好；
0类占19.6%，客户大多是女性，会员，骑行时间平均在9分钟，年龄平均在34岁
1类占12.9%，客户大多是男性，会员，骑行时间平均在22分钟，年龄平均在35岁，
2类占3.1%，客户大多是男性，普通用户，时间平均在15分钟，年龄平均在33岁，
3类占18.3%，客户大多是男性，会员，骑行时间平均在8分钟，年龄平均在53岁，
4类占46.1%，客户大多是男性，会员，骑行时间平均在7分钟，年龄平均在31岁，是摩拜的核心用户群，猜测是刚毕业人群
'''