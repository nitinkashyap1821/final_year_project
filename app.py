import json
import os
import warnings

warnings.filterwarnings('ignore')
# -------------------------------------------------model_code------------------------------------------------------------
import sqlite3

import plotly
import plotly.express as px

conn = sqlite3.connect('rainfall_database')
cur = conn.cursor()
try:
    cur.execute('''CREATE TABLE user (
    user_id NUMBER PRIMARY KEY AUTOINCREMENT,
     name varchar(20) DEFAULT NULL,
      email varchar(50) DEFAULT NULL,
     password varchar(20) DEFAULT NULL,
     gender varchar(10) DEFAULT NULL,
     age int(11) DEFAULT NULL
   )''')

except:
    pass
# !/usr/bin/env python
# coding: utf-8


# include packages
import pandas as pd

# import matplotlib.pyplot as plt
# import seaborn as sns

# reading the dataset
dataset = pd.read_csv("Daily Rainfall dataset.csv")
dataset.head()

from sklearn.model_selection import train_test_split

predictors = dataset.drop(["year", "Rainfall"], axis=1)
target = dataset["Rainfall"]

X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=0)

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

lr = LinearRegression()

lab_enc = preprocessing.LabelEncoder()
Y_train = lab_enc.fit_transform(Y_train)

lr.fit(X_train, Y_train)

Y_pred_lr = lr.predict(X_test)

score_lr = lr.score(X_test, Y_test)
print("The accuracy score achieved using Logistic regression is: " + str(score_lr) + " %")

data1 = pd.read_csv('rainfall dataset india 1901-2017.csv', index_col=[0])  # used for flood prediction

data = pd.read_csv('rainfall dataset india 1901-2017.csv', index_col=[0])  # used for rainfall_analysis
data.drop(['Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec', 'YEAR', 'ANNUAL', 'Flood'], axis=1,
          inplace=True)  # used for rainfall_analysis

# cleaning data for flood_prediction
data1.drop(['Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec', 'YEAR', 'ANNUAL'], axis=1,
           inplace=True)  # used for flood prediction
data1['SUBDIVISION'] = data1['SUBDIVISION'].str.upper()
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
SUBDIVISION = le.fit_transform(data1.SUBDIVISION)
data1['SUBDIVISION'] = SUBDIVISION
data1.dropna(inplace=True, axis=0)
data1['Flood'].replace(['YES', 'NO'], [1, 0], inplace=True)
x1 = data1.iloc[:, 0:13]
y1 = data1.iloc[:, -1]
from sklearn.model_selection import train_test_split

x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.2)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(x1_train, y1_train)
y_pred_lr = lr.predict(x1_test)
from sklearn.metrics import accuracy_score

print("\naccuracy score:%f" % (accuracy_score(y1_test, y_pred_lr) * 100))
# from sklearn import preprocessing
#
# le = preprocessing.LabelEncoder()
#
# SUBDIVISION = le.fit_transform(data_1.SUBDIVISION)
# data_1['SUBDIVISION'] = SUBDIVISION
#
# data_1.dropna(inplace=True)
#
# data_1['Flood'].replace(['YES', 'NO'], [1, 0], inplace=True)
# x1 = data_1.iloc[:, 0:14]
#
# y1 = data_1.iloc[:, -1]
#
# from sklearn.model_selection import train_test_split
#
# x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.2)
#
# from sklearn.naive_bayes import GaussianNB
#
# clf_NB = GaussianNB()
# clf_NB.fit(x1_train, y1_train)
# y_pred_NB = clf_NB.predict(x1_test)
#
# from sklearn.metrics import accuracy_score
#
# print(accuracy_score(y_pred_NB, y1_test))
# score_nb = accuracy_score(y_pred_NB, y1_test)

from flask import Flask, render_template, url_for, request, flash, redirect, session

app = Flask(__name__)
app.config['SECRET_KEY'] = '881e69e15e7a528830975467b9d87a98'


# -------------------------------------home_page-------------------------------------------------------------------------

@app.route('/')
@app.route('/home')
def home():
    if not session.get('logged_in'):
        return render_template('home.html')
    else:
        return redirect(url_for('user_account'))


# -------------------------------------about_page-------------------------------------------------------------------------
@app.route("/about")
def about():
    return render_template('about.html')


# -------------------------------------about_page-------------------------------------------------------------------------


# --------------------------------------help_page-------------------------------------------------------------------------
@app.route("/help")
def helping():
    return render_template('help.html')


# --------------------------------------help_page-------------------------------------------------------------------------
# --------------------------------------helpafterlogin_page-------------------------------------------------------------------------
@app.route("/helpafterlogin")
def helpafterlogin():
    return render_template('helpafterlogin.html')


# --------------------------------------helpafterloginpage------------------------------------------------------------------------

# -------------------------------------user_login_page-------------------------------------------------------------------------
@app.route('/user_login', methods=['POST', 'GET'])
def user_login():
    conn = sqlite3.connect('rainfall_database')
    print("a")
    cur = conn.cursor()
    print("b")
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['psw']
        print('asd')
        count = cur.execute('SELECT * FROM user WHERE email = "%s" AND password = "%s"' % (email, password))
        print(count)
        # conn.commit()
        # cur.close()

        l = len(cur.fetchall())
        if l > 0:
            flash(f'Successfully Logged in')
            session['uname']=email
            session['psw']=password
            return render_template('index.html')
        else:
            print('hello')
            flash(f'Invalid Email and Password!')
    return render_template('user_login.html')


# -------------------------------------user_login_page-----------------------------------------------------------------
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/aboutafterlogin')
def aboutafterlogin():
    return render_template('aboutafterlogin.html')

# -------------------------------------user_register_page-------------------------------------------------------------------------

@app.route('/user_register', methods=['POST', 'GET'])
def user_register():
    conn = sqlite3.connect('rainfall_database')
    cur = conn.cursor()
    if request.method == 'POST':
        name = request.form['uname']
        email = request.form['email']
        password = request.form['psw']
        gender = request.form['gender']
        age = request.form['age']
        print('before')
        cur.execute("insert into user(name,email,password,gender,age) values ('%s','%s','%s','%s','%s')" % (
            name, email, password, gender, age))
        conn.commit()
        # cur.close()
        print('data inserted')
        flash(f'Successfully Registered')
        return redirect(url_for('user_login'))

    return render_template('user_register.html')


# -------------------------------------user_register_page-------------------------------------------------------------------------


# --------------------------------------rainfall_analysis_prediction----------------------------------------------------------------
@app.route('/rainfall_analysis_predict', methods=['POST', 'GET'])
def rainfall_analysis_predict():
    if request.method == 'POST':
        text = request.form['subdivision']
        # return redirect(url_for("rainfall_graph",text=text))
        subDivision = data.loc[data['SUBDIVISION'] == text]
        subDivision = subDivision.mean()

        fig = px.bar(subDivision, x=subDivision.index, y=subDivision.values)
        fig.update_layout(
        title=text,
        xaxis_title="Months",
        yaxis_title="Rainfall (in mm)",
        legend_title="Legend Title",
        )
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return render_template('rainfall_graph.html', graphJSON=graphJSON)
        
    else:
        return render_template("rainfall_analysis_predict.html")

    # day = request.form['day']
    # visibilityHigh = request.form['visibilityHigh']
    # visibilityAvg = request.form['visibilityAvg']
    # month = request.form['month']
    # tempHigh = request.form['tempHigh']
    # tempAvg = request.form['tempAvg']
    # visibilityLow = request.form['visibilityLow']
    # tempLow = request.form['tempLow']
    # windAvg = request.form['windAvg']
    # DPLow = request.form['DPLow']
    # DPHigh = request.form['DPHigh']
    # DPAvg = request.form['DPAvg']
    # humidityHigh = request.form['humidityHigh']
    # SLPHigh = request.form['SLPHigh']
    # SLPLow = request.form['SLPLow']
    # SLPAvg = request.form['SLPAvg']
    # humidityAvg = request.form['humidityAvg']
    # humidityLow = request.form['humidityLow']
    # global lr
    # if request.method == 'POST':
    #     out = lr.predict([[float(month), float(day), float(tempHigh), float(tempAvg), float(tempLow), float(DPHigh),
    #                        float(DPAvg), float(DPLow), float(humidityHigh), float(humidityAvg), float(humidityLow),
    #                        float(SLPHigh), float(SLPAvg), float(SLPLow), float(visibilityHigh), float(visibilityAvg),
    #                        float(visibilityLow), float(windAvg)]])
    #     out1 = float("%.2f" % out)
    #     # out(0, abs(out[0]))
    #     if out1 <= 0:
    #         flash(f'The Rainfall  is 0mm')
    #     else:
    #         flash(f'The Rainfall  is {out1}mm')
    #     # return out
    #     # output.delete(0, END)
    #     # output.insert(0, abs(out[0]))
    #     return render_template('user_account.html')


# --------------------------------------rainfall_analysis_prediction------------------------------

# --------------------------------------rainfall_analysis_graph-----------------------------
# @app.route("/rainfall_graph/<txt>")
# def rainfall_graph(text):
#     subDivision = data.loc[data['SUBDIVISION'] == text]
#     subDivision = subDivision.mean()
#
#     fig = px.bar(subDivision, x=subDivision.index, y=subDivision.values)
#     fig.update_layout(
#         title="Plot Title",
#         xaxis_title="X Axis Title",
#         yaxis_title="X Axis Title",
#         legend_title="Legend Title",
#     )
#     graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
#     return render_template('rainfall_graph.html', graphJSON=graphJSON)


# --------------------------------------rainfall_analysis_graph------------------------------

# ------------------------------------predict_page-----------------------------------------------------------------

@app.route("/flood")
def flood():
    return render_template('flood.html')

@app.route("/noflood")
def noflood():
    return render_template('noflood.html')
# -------------------------------------

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        SUBDIVISION = request.form['subdivision']
        JAN = request.form['jan']
        FEB = request.form['feb']
        MAR = request.form['Mar']
        APR = request.form['Apr']
        MAY = request.form['May']
        JUN = request.form['Jun']
        JUL = request.form['Jul']
        AUG = request.form['Aug']
        SEP = request.form['Sep']
        OCT = request.form['Oct']
        NOV = request.form['Nov']
        DEC = request.form['Dec']

        sd= {'ANDAMAN & NICOBAR ISLANDS': 0,
         'ARUNACHAL PRADESH': 1,
         'ASSAM & MEGHALAYA': 2,
         'NAGA MANI MIZO TRIPURA': 21,
         'SUB HIMALAYAN WEST BENGAL & SIKKIM': 28,
         'GANGETIC WEST BENGAL': 10,
         'ORISSA': 23,
         'JHARKHAND': 15,
         'BIHAR': 3,
         'EAST UTTAR PRADESH': 9,
         'WEST UTTAR PRADESH': 35,
         'UTTARAKHAND': 31,
         'HARYANA DELHI & CHANDIGARH': 12,
         'PUNJAB': 24,
         'HIMACHAL PRADESH': 13,
         'JAMMU & KASHMIR': 14,
         'WEST RAJASTHAN': 34,
         'EAST RAJASTHAN': 8,
         'WEST MADHYA PRADESH': 33,
         'EAST MADHYA PRADESH': 7,
         'GUJARAT REGION': 11,
         'SAURASHTRA & KUTCH': 26,
         'KONKAN & GOA': 17,
         'MADHYA MAHARASHTRA': 19,
         'MATATHWADA': 20,
         'VIDARBHA': 32,
         'CHHATTISGARH': 4,
         'COASTAL ANDHRA PRADESH': 5,
         'TELANGANA': 30,
         'RAYALSEEMA': 25,
         'TAMIL NADU': 29,
         'COASTAL KARNATAKA': 6,
         'NORTH INTERIOR KARNATAKA': 22,
         'SOUTH INTERIOR KARNATAKA': 27,
         'KERALA': 16,
         'LAKSHADWEEP': 18}
        print(SUBDIVISION)

        out = lr.predict([[sd[SUBDIVISION], float(JAN), float(FEB), float(MAR), float(APR), float(MAY), float(JUN), float(JUL),
                           float(AUG), float(SEP), float(OCT), float(NOV), float(DEC)]])
        if out[0] == 0:
            return redirect(url_for('noflood'))
        else:
            return redirect(url_for('flood'))
    return render_template("predict.html")
    # SUBDIVISION = request.form['SUBDIVISION']
    # YEAR = request.form['YEAR']
    # JAN = request.form['JAN']
    # FEB = request.form['FEB']
    # MAR = request.form['MAR']
    # APR = request.form['APR']
    # MAY = request.form['MAY']
    # JUN = request.form['JUN']
    # JUL = request.form['JUL']
    # AUG = request.form['AUG']
    # SEP = request.form['SEP']
    # OCT = request.form['OCT']
    # NOV = request.form['NOV']
    # DEC = request.form['DEC']
    # out = clf_NB.predict([[float(SUBDIVISION), float(YEAR), float(JAN), float(FEB), float(MAR), float(APR),
    #                        float(MAY), float(JUN), float(JUL), float(AUG), float(SEP), float(OCT),
    #                        float(NOV), float(DEC)]])
    # print(out)
    # if out[0] == 1:
    #     # s = print('Yes floods in {}')
    #     # print(s.format(SUBDIVISION))
    #     flash(f'Yes')
    #     return render_template('index.html')
    #
    # else:
    #     print('No')
    #     flash(f'No')
    #     return render_template('noFlood.html')
    #
    # return render_template('flood.html')


@app.route("/user_account")
def user_account():
    conn = sqlite3.connect('rainfall_database')
    print("a")
    cur = conn.cursor()
    print("b")
    email=session['uname']
    password=session['psw']
    count = cur.execute('SELECT * FROM user WHERE email = "%s" AND password = "%s"' % (email, password))
    userdetails=cur.fetchall()[0]
    d=dict()
    d['name']=userdetails[0]
    d['email']=userdetails[1]
    d['gender']=userdetails[3]
    d['age']=userdetails[4]
    return render_template('user_account.html',details=d)


@app.route("/logout")
def logout():
    session['logged_in'] = False
    return render_template('home.html')

if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    app.run(debug=True)
