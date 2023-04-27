from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
from get_names import names
# from convert_crime_data import names 


# import data
df = pd.read_csv('new_filename.csv')


# def run_regression(y_name):

y_var_name='racepctblack numeric'
y_vec = df[[y_var_name]].values

subset=df.iloc[:,:128] # for some reason if I don't do it this way I get an error
x_mat = subset.drop([y_var_name,'communityname string'], axis=1).values # drop y_vec and the string values

X_train, X_test, Y_train, Y_test = train_test_split(x_mat, y_vec, test_size=0.2)

model = linear_model.LinearRegression() # I think this just gives us a shorthand
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

print(f"Coefficients: {model.coef_} \n Intercept: {model.intercept_}")
print('Mean Squared Error: %.2f' % mean_squared_error(Y_test, Y_pred))
print('Coefficient of determination : %.2f' % r2_score(Y_test, Y_pred))

results={
    'weights': model.coef_,
    'indices':[i for i in range(len(model.coef_))],
    'names': names
}

weights=model.coef_[0]
data = [] # lord forgive me, finally found a use for pointers
for i in range(len(weights)):
    data.append((weights[i], i, names[i]))

sorted_data = sorted(data, key=lambda x: x[0])

plt.scatter(Y_test, Y_pred)
plt.title(f"Adam's Model of predicting the {y_var_name}")
plt.xlabel(f"Actual {y_var_name}")
plt.ylabel(f"Predicted {y_var_name}")
plt.show()