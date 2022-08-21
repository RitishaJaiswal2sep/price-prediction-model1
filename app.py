import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
enc = pickle.load(open('enc.pkl', 'rb'))
labelencoder = pickle.load(open('labelencoder.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [str(x) for x in request.form.values()]
    #final_features = [np.array(features, dtype=object)]

    
    df_nb = pd.DataFrame({'neighbourhood': features[0]}, index=[0])
    enc_df = pd.DataFrame(enc.transform(df_nb).toarray())
    
    df_bul = pd.DataFrame({'building': features[4]}, index=[0])
    labelencoder_bul = pd.DataFrame(labelencoder.transform(df_bul))
    
    final_df = pd.DataFrame({'size':int(features[1]),'bedrooms':int(features[2]),'bathrooms':int(features[3]),'neighbourhood_0':enc_df[0],'neighbourhood_1':enc_df[1],'neighbourhood_2':enc_df[2],'building_lbl': labelencoder_bul[0] })
 
    prediction = model.predict(final_df)

    output = prediction

    return render_template('index.html', prediction_text='The price of the apartment you are looking for is around $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)