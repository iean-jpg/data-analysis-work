import pandas as pd
from sklearn import model_selection
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVC

train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')

# 解决空值
train_df.Embarked.fillna(train_df.Embarked.mode()[0], inplace=True)
train_df.Cabin.fillna('UnKnow', inplace=True)
# train_df.Age.fillna(train_df.Age.median(), inplace=True)

# 特征工程
# 提取船舱位置信息
train_df.loc[:, 'Cabin'] = train_df.Cabin.apply(lambda x: x[0])
# 船票号获取相同船票号的数量
tc = dict(train_df.Ticket.value_counts())
train_df['ticket_group'] = train_df.Ticket.apply(lambda x: tc[x])
# Name名字拆分提取三部分内容
train_df['f_name'] = train_df.Name.apply(lambda x: x[:x.find(',')])
train_df['title'] = train_df.Name.apply(lambda x: x[x.find(',') + 1:x.find('.')])
train_df['l_name'] = train_df.Name.apply(lambda x: x[x.find('.') + 1:])
# 增加家庭成员数量
train_df['f_size'] = train_df.Parch + train_df.SibSp + 1
# 对家庭成员进行分段
train_df['f_size_type'] = 'a'
train_df.loc[(train_df.f_size >= 4) & (train_df.f_size <= 5), 'f_size_type'] = 'b'
train_df.loc[train_df.f_size > 5, 'f_size_type'] = 'c'
# 对Age分段
train_df['age_type'] = 'a'
train_df.loc[(train_df.Age > 16) & (train_df.Age <= 30), 'age_type'] = 'b'
train_df.loc[(train_df.Age > 30) & (train_df.Age <= 40), 'age_type'] = 'c'
train_df.loc[(train_df.Age > 40), 'age_type'] = 'd'
# Age年龄，缺失值用众数代替
titles = train_df.title.unique()
for title in titles:
    train_df.loc[(train_df.title == title) & (train_df.Age.isnull()), 'Age'] = train_df.Age[
        train_df.title == title].median()
# 增加儿童标识
train_df['is_child'] = 0
train_df.loc[train_df.Age < 18, 'is_child'] = 1
# 名字长度
train_df['name_length'] = train_df.Name.apply(lambda x: len(x))
# 家族中是否有人遇难或获救
train_df['family_survived'] = 0.5
for idx, rows in train_df.groupby(['f_name', 'Fare']):
    if len(rows) == 1:
        continue
    for idx2, row in rows.iterrows():
        survived = rows.drop(idx2).Survived.max()
        train_df.loc[(train_df.PassengerId == row.PassengerId), 'family_survived'] = 1
# Ticket 船票号，提取相同号的数量
for idx, rows in train_df.groupby(['Ticket']):
    if len(rows) == 1:
        continue
    for idx2, row in rows.iterrows():
        survived = rows.drop(idx2).Survived.max()
        train_df.loc[(train_df.PassengerId == row.PassengerId), 'family_survived'] = 1

train_df = pd.get_dummies(train_df)
X = train_df.iloc[:, 2:]
y = train_df.iloc[:, 1]
X = SelectKBest(k=1370).fit_transform(X, y)
clf = SVC(kernel='linear', C=0.31)
sc = model_selection.cross_val_score(clf, X, y, scoring='accuracy')
clf.fit(X, y)
print(sc.mean())
