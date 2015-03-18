import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


full_answers = pd.read_csv('../data/covtype.data')
submission = pd.read_csv('output.csv')

full_answers = full_answers.ix[len(full_answers.index) - len(submission.index):, len(full_answers.columns) - 1:]


print(accuracy_score(full_answers, submission['Cover_Type']))