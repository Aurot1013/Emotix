import pandas as pd
import re
import unicodedata

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

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
X_padded = pad_sequences(sequences, maxlen= longueur_max); #Ajoute des 0 jusqu a la phrase la plus longue car en ML on a besoin d avoir la meme taile de caractere

#Encoder les labels
label_encoder = LabelEncoder()
y_encoder = label_encoder.fit_transform(y);

#Construction du modele
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=longueur_max)) # La phrase devient une matrice  de taille longueur_max,128
model.add(LSTM(64)) #regarde la sequence de mot pour comprendre le contexte, la sortie du LSTM est un vecteur de taille 64
model.add(Dense(1, activation='sigmoid'))  #transforme la sortie du LSTM en une probabilité pour chaque classe et sigmoid assure que la somme des probabilité est égale a 1

#Compilation du modele
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Entrainement du modele
model.fit(
    X_padded,      # séquences de mots
    y_encoder,     # labels encodés
    epochs=10,     # nombre de passages sur tout le dataset + il est grand plus le modele apprendre mais risque de surapprendre
    batch_size=2   # combien de phrases à la fois petites valeur maj frequente mais plus lent
)


phrase = input("Tape ta phrase à analyser : ")

# Préparation de la phrase
phrase_prep = "".join(
    c for c in unicodedata.normalize("NFD", re.sub(r"[^\w\s]", "", phrase.lower()))
    if unicodedata.category(c) != "Mn"
)

# Tokenization
sequence = tokenizer.texts_to_sequences([phrase_prep])

# Padding pour correspondre à longueur_max
sequence_pad = pad_sequences(sequence, maxlen=longueur_max)

# Prédiction
prediction = model.predict(sequence_pad)[0][0]  # récupère la valeur du sigmoid

# Seuil de 0.5 pour décider si c'est positif ou négatif
if prediction >= 0.5:
    resultat = 'positif'
else:
    resultat = 'negatif'

print("Sentiment prédit :", resultat)
