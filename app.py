from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

pipe = pickle.load(open('pipe.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')
    


@app.route('/predict', methods=['POST'])
def predict():
    temp_array = list()
    if request.method == 'POST':
        batting_team = request.form['batting-team']
        bowling_team = request.form['bowling-team']
        target = int(request.form['target'])
        score = int(request.form['score'])
        overs = float(request.form['overs'])
        wickets = float(request.form['wickets'])
        runs_left = target - score
        ov = str(overs)
        index = ov.index('.')
        o = ov[0 : index]
        b = ov[index : ]
        if float(o) != 0.0:
            z = int(float(o) * 6 + float(b))
        else:
            z = float(float(o) * 6 + float(b))
        balls_left = 120 - z
        wickets = 10 - wickets
        crr = score / z
        rrr = (runs_left*6)/balls_left
    
    input_df = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'runs_left':[runs_left],'balls_left':[balls_left],'wickets':[wickets],'total_runs_x':[target],'crr':[crr],'rrr':[rrr]})
    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    win = str(round(win*100, 2))
    loss = str(round(loss*100, 2))
    
    return render_template('result.html', win = win, batting_team = batting_team, loss = loss, bowling_team = bowling_team)
    
    








if __name__ == '__main__':
	app.run(debug=True)