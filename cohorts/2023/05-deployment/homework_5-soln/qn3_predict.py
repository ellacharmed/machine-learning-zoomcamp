import pickle

def load(input_file):
    with open(input_file, 'rb') as f_in: 
        return pickle.load(f_in)

dv = load(f'dv.bin')
model = load(f'model1.bin')
a_client = {"job": "retired", "duration": 445, "poutcome": "success"}

X = dv.transform([a_client])
y_pred = model.predict_proba(X)[0, 1]

print(a_client)
print(y_pred)

# with own 'model1.bin', as md5sum failed. value is different in g-form, not among MCQ choices
# {'job': 'retired', 'duration': 445, 'poutcome': 'success'}
# 0.863849776245835