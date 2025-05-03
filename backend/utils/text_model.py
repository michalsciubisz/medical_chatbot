from joblib import load
from config.constant import COLUMNS_TO_FILL_PATIENT

import pandas as pd
import random

def create_df_from_response(response):
    
        height = int(response["What's your height? [cm]"])
        weight = int(response["What's your weight? [kg]"])

        bmi_calc = weight / (height/100)**2

        patient = {
            'Gender': random.choice(['woman', 'man']),
            'Age': response["How old are you?"],
            'Weight': weight,
            'Height': height,
            'BMI': bmi_calc,
            'Work type': response["What type of work do you have (physical/headwork/mixed/do not work)?"],
            **{col: 0 for col in COLUMNS_TO_FILL_PATIENT if 'Diagnosis' in col or 'Cause' in col or 'Ailments' in col},
            'Family history with osteoarthisis': 'no',
            'Moving independently': 'yes',
            'Use of assistive equipment': 'no',
            'Performing body toilet, bathing': int(response["Rate difficulty while getting in/out of bath:"]),
            'Dressing': int(response["Rate difficulty while putting on socks:"]),
            'Cleaning': int(response["Rate difficulty while light domestic duties:"]),
            'Lifting or carrying heavy items': int(response["Rate difficulty while heavy domestic duties:"]),
            'Climbing stairs up to one floor': int(response["Rate difficulty while ascending stairs:"]),
            'Climbing stairs up to second floor': int(response["Rate difficulty while ascending stairs:"]),
            'Going down the stairs': int(response["Rate difficulty while descending stairs:"]),
            'Bending or crouching': int(response["Rate difficulty while bending to floor:"]),
            'Walk about 100 meters': int(response["Rate difficulty while walking on flat surface:"]),
            'Walk about 500 meters': int(response["Rate difficulty while walking on flat surface:"]),
            'Walk above 1000 meters': int(response["Rate difficulty while walking on flat surface:"]),
            'Cycling': int(response["Rate difficulty while cycling:"]),
            'Pursuing your own interests': int(response["Rate your joint pain during rest:"]),
            'How often did pain occur in last month': int(response["How often did pain occured in last month? (0 not at all - 4 very often)"]),
            'Pain disturbed everyday life': int(response["Did pain disturbed your everydaylife? (0 not at all - 3 very much)"]),
            'Does pain forced to use of painkillers': 0,
            'Does osteoarthritis affected your professional work': 0,
            'Did you practice any sports or recreational activities before your illness': 'recreational activities',
            'Forced to give up': 'not affected',
            'Emotional disturbances': 'not present',
            'Therapy - pharmacological': 0.0,
            'Therapy - non-pharmacological': 0.0,
            'Therapy - surgical': 0.0,
            'Therapy - diet': 0.0
        }

        return pd.DataFrame([patient])

def predict_class(response):
    
    # loading the original data is necessary to match columns that model was trained on
    df = pd.read_csv("D:/semestr_10/master_thesis/medical_chatbot/data/text_data.csv")
    df = df.drop(['severity_score','severity_grade'], axis=1) # those data were ommited during training

    # creating patient df based on response
    patient_df = create_df_from_response(response)

    # one hot encoding on both data
    patient_hot_encoded = pd.get_dummies(patient_df)
    df_what = pd.get_dummies(df)

    # adding missing columns with zeros
    patient_hot_encoded = patient_hot_encoded.reindex(columns=df_what.columns, fill_value=0)

    # loading the model
    model = load("D:/semestr_10/master_thesis/medical_chatbot/notebooks/models/best_model_nlp.pkl")

    # getting the propabilities
    prediction_prob = model.predict_proba(patient_hot_encoded)

    return prediction_prob