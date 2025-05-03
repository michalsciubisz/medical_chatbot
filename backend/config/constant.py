IMAGE_MAPPER = {
    0 : "healthy/doubtful",
    1 : "moderate",
    2 : "severe"
}

QUESTIONS = {
    'womac' : [
        "How old are you?",
        "What's your height? [cm]",
        "What's your weight? [kg]",
        "What type of work do you have (physical/headwork/mixed/do not work)?",
        "Next questions will be on scale 0-4 \n Do you feel any pain while walking?", # pain
        "Do you feel any pain while climbing stairs?", # pain
        "Rate your nocturnal joint pain:", # pain
        "Rate your joint pain during rest:", # pain
        "Rate your joint pain during weight bearing:", # pain
        "Rate your morning stiffness:", # stiffness
        "Rate your stiffness later during the day:", # stiffness
        "Rate difficulty while descending stairs:", # physical function
        "Rate difficulty while ascending stairs:", # physical function
        "Rate difficulty while rising from sitting:", # physical function
        "Rate difficulty while standing:", # physical function
        "Rate difficulty while bending to floor:", # physical function
        "Rate difficulty while walking on flat surface:", # physical function
        "Rate difficulty while getting in or out of the car:", # physical function
        "Rate difficulty while going shopping:", # physical function
        "Rate difficulty while putting on socks:", # physical function
        "Rate difficulty while lying on bed:", # physical function
        "Rate difficulty while taking off socks:", # physical function
        "Rate difficulty while rising from bed:", # physical function
        "Rate difficulty while getting in/out of bath:", # physical function
        "Rate difficulty while sitting:", # physical function
        "Rate difficulty while getting on/off toilet:", # physical function
        "Rate difficulty while heavy domestic duties:", # physical function
        "Rate difficulty while light domestic duties:", # physical function
        "Rate difficulty while cycling:",
        "How often did pain occured in last month? (0 not at all - 4 very often)",
        "Did pain disturbed your everydaylife? (0 not at all - 3 very much)",
        ]
}

WOMAC_COLUMNS = ["Next questions will be on scale 0-4 \n Do you feel any pain while walking?", # pain
        "Do you feel any pain while climbing stairs?", # pain
        "Rate your nocturnal joint pain:", # pain
        "Rate your joint pain during rest:", # pain
        "Rate your joint pain during weight bearing:", # pain
        "Rate your morning stiffness:", # stiffness
        "Rate your stiffness later during the day:", # stiffness
        "Rate difficulty while descending stairs:", # physical function
        "Rate difficulty while ascending stairs:", # physical function
        "Rate difficulty while rising from sitting:", # physical function
        "Rate difficulty while standing:", # physical function
        "Rate difficulty while bending to floor:", # physical function
        "Rate difficulty while walking on flat surface:", # physical function
        "Rate difficulty while getting in or out of the car:", # physical function
        "Rate difficulty while going shopping:", # physical function
        "Rate difficulty while putting on socks:", # physical function
        "Rate difficulty while lying on bed:", # physical function
        "Rate difficulty while taking off socks:", # physical function
        "Rate difficulty while rising from bed:", # physical function
        "Rate difficulty while getting in/out of bath:", # physical function
        "Rate difficulty while sitting:", # physical function
        "Rate difficulty while getting on/off toilet:", # physical function
        "Rate difficulty while heavy domestic duties:", # physical function
        "Rate difficulty while light domestic duties:", # physical function
        ]

WOMAC_CLASS_MAPPING = {
    "No functional issues": 0,
    "Mild functional limitation": 1,
    "Moderate functional limitation": 2,
    "Severe functional limitation": 3,
    "Extreme functional limitation": 4
}

COLUMNS_TO_FILL_PATIENT = ["Diagnosis - medical examination and interview","Diagnosis - USG","Diagnosis - RTG","Diagnosis - MRI","Diagnosis - CT","Diagnosis - fluid from the joint","Cause - age","Cause - genetics","Cause - physical work","Cause - sedentary lifestyle","Cause - little physical activity","Cause - obesity","Cause - injuries","Cause - competitive sports","Cause - joint instability","Cause - joint inflammation","Cause - knock knees","Cause - spinal curvature","Cause - comorbidities","Ailments - pain","Ailments - limitation of mobility","Ailments - impairment of rotation and abduction in the joint","Ailments - stiffness in the joint","Ailments - difficulty bending and squatting","Ailments - pelvic drop on the side opposite to the affected side","Ailments - joint instability","Ailments - tenderness","Ailments - joint crepitus","Ailments - widening and distortion of the outlines of the joints","Ailments - joint swelling","Ailments - difficulty performing precise activities with your fingers","Ailments - limping", "Ailments - deterioration in the quality of gait"]
