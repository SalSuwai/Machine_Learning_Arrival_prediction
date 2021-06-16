from flask import Flask, render_template, request
import pickle 
import numpy as np

model = pickle.load(open('model_GNB.pkl', 'rb') )

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def man():
    data1 = int(request.form['one'])
    data2 = int(request.form['two'])
    data3 = int(request.form['three'])
    data4 = int(request.form['four'])
    data5 = int(request.form['five'])
    data6 = int(request.form['six'])
    data7 = int(request.form['seven'])
    data8 = int(request.form['eight']) 
    data9 = int(request.form['nine'])
    data10 = int(request.form['ten'])
    data11 = int(request.form['A'])
    data12 = int(request.form['B'])
    data13 = int(request.form['C'])
    data14 = int(request.form['D'])
    data15 = int(request.form['E'])

    Arr=np.array([[data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13,data14,data15]])
    
    pred = model.predict(Arr)
    return render_template('after.html',data=pred)
    
    if __name__ == '__main__':
        app.run(debug=True)

