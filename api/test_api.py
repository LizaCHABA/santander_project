import pandas as pd
import requests
import json

#Charger le dataset

df = pd.read_csv("../data/train.csv")

#Enlever les colonnes non utilisées
X = df.drop(columns=["target", "ID_code"])

#Prendre 1 exemple réel
sample = X.sample(1, random_state=1)

#Convertir en dictionnaire JSON
payload = sample.to_dict(orient="records")[0]

print("\n====== Données envoyées à l’API ======\n")
print(json.dumps(payload, indent=4))


#Appler Api
url = "http://127.0.0.1:5000/predict"

response = requests.post(url, json=payload)


#Affichage de la réponse 
print("\n====== Réponse de l’API ======\n")
print(response.json())
