import pandas as pd
import re
import unicodedata

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder




data = pd.read_csv("data/examples.csv") #Recupere le tableaux


# Enleve les accent avec unicode, enleve les majuscules avec lower et applique grace a apply
textFinal = data["text"].apply(
    lambda x: "".join(
        c for c in unicodedata.normalize("NFD", re.sub(r"[^\w\s]", "", x.lower()))
        if unicodedata.category(c) != "Mn"
    )
)
#Tokenizer 
x= textFinal
y = data["label"]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(x) #Construit le vocabulaire et les indices correpondants a chaque mot
sequences = tokenizer.texts_to_sequences(x) #Transforme chaque phrase en une liste d'indice numérique

longueur_max = max(len(seq) for seq in sequences); #Recupere la phrase la plus longue
X_padded = pad_sequences(sequences, maxlen= longueur_max);#Ajoute des 0 jusqu a la phrase la plus longue car en ML on a besoin d avoir la meme taile de caractere


print(y);
