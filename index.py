import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dates = []
prices= []

def get_data(filename):
    with open(filename, 'r') as csvFile:
        csvFileReader = csv.reader(csvFile)
        next(csvFileReader)
        for row in csvFileReader:
            dates.append(int(row[0].split('-')[0]))
            prices.append(float(row[1]))
    return

def predict_prices(dates,prices,x):
    dates = np.reshape(dates, (len(dates),1))

    svm_len = SVR(kernel='linear', C=1e3)
    svm_polly = SVR(kernel ='poly', C=1e3, degree= 2)
    svm_rbf = SVR(kernel = 'rbf', C=1e3, gamma=0.1 )

    svm_len.fit(dates,prices)
    svm_polly.fit(dates,prices)
    svm_rbf.fit(dates,prices)

    plt.scatter(dates,prices,color='black',label='Data')
    plt.plot(dates,svm_rbf.predict(dates), color='red', label='RBF Mode')
    plt.plot(dates,svm_len.predict(dates), color='green', label='Linear Mode')
    plt.plot(dates,svm_polly.predict(dates), color='blue', label='Polynomial Mode')

    plt.xlabel('prices')
    plt.ylabel('dates')
    plt.title('Support Vectore Machine')
    plt.legend()
    plt.show()

    return svm_rbf.predict(x)[0],svm_len.predict(x)[0],svm_polly.predict(x)[0];

get_data('AAPL-2.csv')
predicted_prices = predict_prices(dates,prices,29)
print(predicted_price);

    
