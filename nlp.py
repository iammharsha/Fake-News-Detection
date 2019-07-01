import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
import seaborn as sb
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cross_validation import KFold


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


#imp.reload(sys)
#sys.setdefaultencoding('utf8')

train = pd.read_csv('train.csv')


def data_obs():
    print("training dataset size:")
    print(train.shape)
    print(train.head(10))

    

data_obs()


def create_distribution(dataFile):
    
    return sb.countplot(x='Label', data=dataFile, palette='hls')

def data_qualityCheck():
    
    print("Checking data qualitites...")
    train.isnull().sum()
    train.info()
        
    print("check finished.")
    

create_distribution(train)
plt.show()
data_qualityCheck()


def feature_selection(dataset):
	corpus = []
	for i in range(0, len(dataset['Statement'])):
	    review = re.sub('[^a-zA-Z]', ' ', dataset['Statement'][i])
	    review = review.lower()
	    review = review.split()
	    ps = SnowballStemmer('english')
	    stopword_set = set(stopwords.words('english'))
	    review = [ps.stem(word) for word in review if not word in stopword_set]
	    review = ' '.join(review)
	    corpus.append(review)
	    #print corpus
	print corpus[:30]
	cv = CountVectorizer()
	X = cv.fit_transform(corpus).toarray()
	y = dataset.iloc[:, 1].values
	#kf = KFold()
	#kf.get_n_splits(X)
	tfidfV = TfidfTransformer()
	X = tfidfV.fit_transform(X,y)
	#print X
	#print(cv.vocabulary_)
	return X,y

X,y=feature_selection(train)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)
classifier = RandomForestClassifier(n_jobs=3)
classifier.fit(X_train, y_train)
#print classifier
#X_test,y_test=feature_selection(test)

y_pred = classifier.predict(X_test)
print y_pred
cm = confusion_matrix(y_test, y_pred)
print cm
tn, fp, fn, tp = cm.ravel()
#print tn,tp,fn,fp
precision=(float)(tp)/(tp+fp)
recall=(float)(tp)/(tp+fn)
print "Accuracy: "+str(((float)(tn+tp)/(tn+tp+fn+fp))*100)
print "Precision: "+ str(precision)
print "Recall: "+ str(recall)
print "F-Score: "+ str((float)(2*(precision*recall)/(precision+recall)))
