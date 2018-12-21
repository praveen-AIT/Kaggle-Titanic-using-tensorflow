import pandas as pd
import numpy as np
import random as rnd
from sklearn.model_selection import train_test_split

# Machine learning
import tensorflow as tf

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

# A pattern with one group will return a DataFrame with one column if expand=True
# A pattern with one group will return a Series if expand=False.

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
	
	
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)\
	

guess_ages = np.zeros((2,3))

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']

train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
	
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]


for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass
	
	
freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
	

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

y = train_df['Survived'].values
y = y.astype(float).reshape(-1, 1)

X_train, X_dev, y_train, y_dev = train_test_split(train_df, y, test_size=0.1, random_state=0)

#X_train = train_df.drop("Survived", axis=1)
#Y_train = train_df["Survived"]
X_test  = test_df.copy()

print(X_train.shape, y_train.shape)
print(type(X_train))

t1 = tf.convert_to_tensor(X_train)
t2 = tf.convert_to_tensor(y_train)
t3 = tf.convert_to_tensor(X_test)

print(t1)


seed = 7 # for reproducible purpose
input_size = X_train.shape[1] # number of features
learning_rate = 0.001 # most common value for Adam
epochs = 815 # I've tested previously that this is the best epochs to avoid overfitting

graph = tf.Graph()
with graph.as_default():
    tf.set_random_seed(seed)
    np.random.seed(seed)

    X_input = tf.placeholder(dtype=tf.float32, shape=[None, input_size], name='X_input')
    y_input = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y_input')
    
    W1 = tf.Variable(tf.random_normal(shape=[input_size, 1], seed=seed), name='W1')
    b1 = tf.Variable(tf.random_normal(shape=[1], seed=seed), name='b1')
    sigm = tf.nn.sigmoid(tf.add(tf.matmul(X_input, W1), b1), name='pred')
    
    # Computes sigmoid cross entropy given logits.
    # sigmoid function returns A Tensor of the same shape as logits with the componentwise logistic losses.
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_input, logits=sigm, name='loss'))
    train_steps = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    pred = tf.cast(tf.greater_equal(sigm, 0.5), tf.float32, name='pred') # 1 if >= 0.5
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, y_input), tf.float32), name='acc')
    
    init_var = tf.global_variables_initializer()
	

train_feed_dict = {X_input: X_train, y_input: y_train}
#dev_feed_dict = {X_input: X_dev, y_input: y_dev}
test_feed_dict = {X_input: X_test} # no y_input since the goal is to predict it

sess = tf.Session(graph=graph)
sess.run(init_var)

cur_loss = sess.run(loss, feed_dict=train_feed_dict)
train_acc = sess.run(acc, feed_dict=train_feed_dict)
#test_acc = sess.run(acc, feed_dict=dev_feed_dict)

print(X_test)
print('step 0: loss {0:.5f}, train_acc {1:.2f}%'.format(cur_loss, 100*train_acc))
for step in range(1, epochs+1):
    sess.run(train_steps, feed_dict=train_feed_dict)
    cur_loss = sess.run(loss, feed_dict=train_feed_dict)
    train_acc = sess.run(acc, feed_dict=train_feed_dict)
    #test_acc = sess.run(acc, feed_dict=dev_feed_dict)
    if step%100 != 0: # print result every 100 steps
        continue
    #print('step {3}: loss {0:.5f}, train_acc {1:.2f}%'.format(cur_loss, 100*train_acc))

train_df = pd.read_csv("train.csv")
id_test = train_df['PassengerId'].values
y_pred = sess.run(pred, feed_dict=test_feed_dict).astype(int)
#prediction = pd.DataFrame(np.concatenate([id_test, y_pred]), columns=['PassengerId', 'Survived'])

#print(y_pred)
	
y = list()
y = y_pred.tolist()
print(y)

file = open("sub2.csv", 'w')
for i in range(418):
	file.write(str(test_df["PassengerId"][i]) + "," + str(y[i][0]) + "\n")

file.close()	
	
	
	
	
	
	
	






