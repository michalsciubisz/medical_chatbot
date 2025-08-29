from joblib import load

MODEL_PATH = "D:/semestr_10/master_thesis/medical_chatbot/notebooks/models/best_model_nlp.pkl"
_model = load(MODEL_PATH)
MODEL_CLASSES = list(_model.classes_)  

TEXT_DATA_CSV_PATH = r"D:/semestr_10/master_thesis/medical_chatbot/data/text_data.csv"

MODEL_CLASSES_FALLBACK = ["S", "T", "H"]   # lub ['Grade 0 - none', ..., 'Grade 4 - severe']

ID_TO_LEGACY = {
    "age": "How old are you?",
    "height_cm": "What's your height? [cm]",
    "weight_kg": "What's your weight? [kg]",
    "work_type": "What type of work do you have (physical/headwork/mixed/do not work)?",

    "pain_rest": "Rate your joint pain during rest:",
    "stiff_morning": "Rate your morning stiffness:",
    "stiff_day": "Rate your stiffness later during the day:",
    "pain_walk": "Do you feel any pain while walking?",
    "pain_stairs": "Do you feel any pain while climbing stairs?",
    "pain_night": "Rate your nocturnal joint pain:",
    "pain_weight": "Rate your joint pain during weight bearing:",
    "phys_descend": "Rate difficulty while descending stairs:",
    "phys_ascend": "Rate difficulty while ascending stairs:",
    "phys_rise_sit": "Rate difficulty while rising from sitting:",
    "phys_standing": "Rate difficulty while standing:",
    "phys_bending": "Rate difficulty while bending to floor:",
    "phys_walk_flat": "Rate difficulty while walking on flat surface:",
    "phys_car": "Rate difficulty while getting in or out of the car:",
    "phys_shopping": "Rate difficulty while going shopping:",
    "phys_put_socks": "Rate difficulty while putting on socks:",
    "phys_lying": "Rate difficulty while lying on bed:",
    "phys_take_socks": "Rate difficulty while taking off socks:",
    "phys_rise_bed": "Rate difficulty while rising from bed:",
    "phys_bath": "Rate difficulty while getting in/out of bath:",
    "phys_sitting": "Rate difficulty while sitting:",
    "phys_toilet": "Rate difficulty while getting on/off toilet:",
    "phys_heavy": "Rate difficulty while heavy domestic duties:",
    "phys_light": "Rate difficulty while light domestic duties:",
    "phys_cycling": "Rate difficulty while cycling:",
    "freq_last_month": "How often did pain occured in last month? (0 not at all - 4 very often)",
    "disturb_daily": "Did pain disturbed your everydaylife? (0 not at all - 3 very much)",
}

LEGACY_TO_ID = {v: k for k, v in ID_TO_LEGACY.items()}

WORK_TYPE_MAP = {
    "physical": "physical",
    "headwork": "headwork",
    "mixed": "mixed",
    "do_not_work": "do not work",
    "do not work": "do not work",
}

YN_MAP = {
    "yes": "tak", "tak": "tak", "y": "tak", True: "tak",
    "no": "nie", "nie": "nie", "n": "nie", False: "nie",
    "i don't know": "nie wiem", "nie wiem": "nie wiem", None: "nie wiem"
}

DEFAULT_GENDER = "woman"        # ustaw deterministycznie (nie losuj)
DEFAULT_WORK_IF_NONE = "do not work"

IMAGE_MAPPER = {
    0 : "healthy/doubtful",
    1 : "moderate",
    2 : "severe"
}

MIN_ANSWERS = 10 

QUESTIONS = {
    "womac": [
        {"id":"age","text":"How old are you?","type":"number","min":18,"max":110},
        {"id":"height_cm","text":"What's your height? [cm]","type":"number","min":120,"max":230},
        {"id":"weight_kg","text":"What's your weight? [kg]","type":"number","min":30,"max":300},
        {"id":"work_type","text":"What type of work do you have?","type":"choice","choices":["physical","headwork","mixed","do_not_work"]},
        {"id":"pain_walk","text":"Do you feel any pain while walking?","type":"scale","min":0,"max":4,"labels":["None","Slight","Moderate","Very","Extremely"]},
        {"id":"pain_stairs","text":"Do you feel any pain while climbing stairs?","type":"scale","min":0,"max":4,"labels":["None","Slight","Moderate","Very","Extremely"]},
        {"id":"pain_night","text":"Rate your nocturnal joint pain:","type":"scale","min":0,"max":4,"labels":["None","Slight","Moderate","Very","Extremely"]},
        {"id":"pain_rest","text":"Rate your joint pain during rest:","type":"scale","min":0,"max":4,"labels":["None","Slight","Moderate","Very","Extremely"]},
        {"id":"pain_weight","text":"Rate your joint pain during weight bearing:","type":"scale","min":0,"max":4,"labels":["None","Slight","Moderate","Very","Extremely"]},
        {"id":"stiff_morning","text":"Rate your morning stiffness:","type":"scale","min":0,"max":4,"labels":["None","Slight","Moderate","Very","Extremely"]},
        {"id":"stiff_day","text":"Rate your stiffness later during the day:","type":"scale","min":0,"max":4,"labels":["None","Slight","Moderate","Very","Extremely"]},
        {"id":"desc_stair","text":"Rate difficulty while descending stairs:","type":"scale","min":0,"max":4,"labels":["None","Slight","Moderate","Very","Extremely"]},
        {"id":"asc_stair","text":"Rate difficulty while ascending stairs:","type":"scale","min":0,"max":4,"labels":["None","Slight","Moderate","Very","Extremely"]},
        {"id":"rising_sit","text":"Rate difficulty while rising from sitting:","type":"scale","min":0,"max":4,"labels":["None","Slight","Moderate","Very","Extremely"]},
        {"id":"standing","text":"Rate difficulty while standing:","type":"scale","min":0,"max":4,"labels":["None","Slight","Moderate","Very","Extremely"]},
        {"id":"bending","text":"Rate difficulty while bending to floor:","type":"scale","min":0,"max":4,"labels":["None","Slight","Moderate","Very","Extremely"]},
        {"id":"walking","text":"Rate difficulty while walking on flat surface:","type":"scale","min":0,"max":4,"labels":["None","Slight","Moderate","Very","Extremely"]},
        {"id":"getting_car","text":"Rate difficulty while getting in or out of the car:","type":"scale","min":0,"max":4,"labels":["None","Slight","Moderate","Very","Extremely"]},
        {"id":"shopping","text":"Rate difficulty while going shopping:","type":"scale","min":0,"max":4,"labels":["None","Slight","Moderate","Very","Extremely"]},
        {"id":"socks","text":"Rate difficulty while putting on socks:","type":"scale","min":0,"max":4,"labels":["None","Slight","Moderate","Very","Extremely"]},
        {"id":"rising_bed","text":"Rate difficulty while rising from bed:","type":"scale","min":0,"max":4,"labels":["None","Slight","Moderate","Very","Extremely"]},
        {"id":"getting_bath","text":"Rate difficulty while getting in/out of bath:","type":"scale","min":0,"max":4,"labels":["None","Slight","Moderate","Very","Extremely"]},
        {"id":"sitting","text":"Rate difficulty while sitting:","type":"scale","min":0,"max":4,"labels":["None","Slight","Moderate","Very","Extremely"]},
        {"id":"getting_toilet","text":"Rate difficulty while getting on/off toilet:","type":"scale","min":0,"max":4,"labels":["None","Slight","Moderate","Very","Extremely"]},
        {"id":"heavy_d","text":"Rate difficulty while heavy domestic duties:","type":"scale","min":0,"max":4,"labels":["None","Slight","Moderate","Very","Extremely"]},
        {"id":"light_d","text":"Rate difficulty while light domestic duties:","type":"scale","min":0,"max":4,"labels":["None","Slight","Moderate","Very","Extremely"]},
        {"id":"cycling","text":"Rate difficulty while cycling:","type":"scale","min":0,"max":4,"labels":["None","Slight","Moderate","Very","Extremely"]},
        {"id":"freq_last_month","text":"How often did pain occur in last month?","type":"scale","min":0,"max":4,"labels":["Never", "Very rarely", "Rarely","Often","Very often"]},
        {"id":"disturb_daily","text":"Did pain disturb your everyday life? (0 not at all - 3 very much)","type":"scale","min":0,"max":3,"labels":["Not at all","A little","Average","Very"]},
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


COLUMNS_TO_FILL_PATIENT = ["Diagnosis - medical examination and interview","Diagnosis - USG","Diagnosis - RTG","Diagnosis - MRI","Diagnosis - CT","Diagnosis - fluid from the joint","Cause - age","Cause - genetics","Cause - physical work","Cause - sedentary lifestyle","Cause - little physical activity","Cause - obesity","Cause - injuries","Cause - competitive sports","Cause - joint instability","Cause - joint inflammation","Cause - knock knees","Cause - spinal curvature","Cause - comorbidities","Ailments - pain","Ailments - limitation of mobility","Ailments - impairment of rotation and abduction in the joint","Ailments - stiffness in the joint","Ailments - difficulty bending and squatting","Ailments - pelvic drop on the side opposite to the affected side","Ailments - joint instability","Ailments - tenderness","Ailments - joint crepitus","Ailments - widening and distortion of the outlines of the joints","Ailments - joint swelling","Ailments - difficulty performing precise activities with your fingers","Ailments - limping", "Ailments - deterioration in the quality of gait"]

# Kanoniczny porządek klas modelu (zgodny z debugiem: ['Grade 0 - none', ..., 'Grade 4 - severe'])
GRADE_RANK = [
    "Grade 0 - none",
    "Grade 1 - doubtful",
    "Grade 2 - minimal",
    "Grade 3 - moderate",
    "Grade 4 - severe",
]

# Kanoniczne etykiety WOMAC (po normalizacji do %)
WOMAC_LABELS = [
    "No functional issues",
    "Dobtful functional limitation",
    "Minimal functional limitation",
    "Moderate functional limitation",
    "Severe functional limitation",
]

# Progi nasilenia dla etykiet WOMAC; każdy próg to maksymalna wartość w % dla danej etykiety
# Interpretacja: <= threshold -> label
WOMAC_SEVERITY_THRESHOLDS = [
    (10,  WOMAC_LABELS[0]),  # 0..10%
    (30,  WOMAC_LABELS[1]),  # 10..25%
    (50,  WOMAC_LABELS[2]),  # 25..45%
    (70,  WOMAC_LABELS[3]),  # 45..75%
    (100, WOMAC_LABELS[4]),  # 65..100%
]

# Mapowanie etykiet WOMAC -> kanoniczne klasy (Grade 0..4)
LABEL_TO_GRADE = {
    WOMAC_LABELS[0]: GRADE_RANK[0],  # No functional issues -> Grade 0 - none
    WOMAC_LABELS[1]: GRADE_RANK[1],  # Doubtful -> Grade 1
    WOMAC_LABELS[2]: GRADE_RANK[2],  # Minimal -> Grade 2
    WOMAC_LABELS[3]: GRADE_RANK[3],  # Moderate -> Grade 3
    WOMAC_LABELS[4]: GRADE_RANK[4],  # Severe -> Grade 4
}

# Odwrotne mapowanie
GRADE_TO_LABEL = {v: k for k, v in LABEL_TO_GRADE.items()}

# Współczynnik łączenia (WOMAC vs. model tekstowy)
DEFAULT_WOMAC_WEIGHT = 0.7

# Domyślny maks na pytanie typu "scale"
DEFAULT_SCALE_MAX = 4