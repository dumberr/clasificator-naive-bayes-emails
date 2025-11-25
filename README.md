# clasificator-naive-bayes-emails
# Naive Bayes Spam Classifier
Clasificator Naive Bayes Multinomial pentru emailuri spam/ham.  

## Descriere
Acest proiect implementeaza un clasificator Naive Bayes scris de la zero in Python, fara scikit-learn sau alte librarii ML.
Modelul clasifica emailuri in doua categorii: spam si ham.

## Setul de date
Fisierul folosit este `emails.csv`, creat manual din emailuri reale (anonimizate).

Format:
text,label
"Text email 1",ham
"Text email 2",spam

powershell
Copy code

Tokenizarea elimina punctuatia si transforma textul in cuvinte.

## Modelul Naive Bayes

Formula Bayes:
P(c | text) = [ P(text | c) * P(c) ] / P(text)

yaml
Copy code

Independenta cuvintelor:
P(text | c) = Produs_i P(w_i | c)

yaml
Copy code

Laplace smoothing:
P(w | c) = ( count(w,c) + 1 ) / ( total_cuvinte_c + |V| )

yaml
Copy code

Forma logaritmica folosita:
log P(c) + Suma_i log P(w_i | c)

diff
Copy code

## Structura codului
- tokenize(): curata textul si il imparte in cuvinte
- load_dataset(): citeste CSV-ul
- train_test_split(): imparte datele in train/test
- NaiveBayesMultinomial: implementarea modelului
- accuracy_score(): calculeaza acuratetea
- main(): pipeline complet

## Rulare

Varianta standard:
python bayes.py

yaml
Copy code

Cu alt dataset:
python bayes.py alt_dataset.csv

makefile
Copy code

Exemplu:
Felicitari, ai castigat un premiu garantat!
prediction: spam

markdown
Copy code

## Rezultate
Acuratete obtinuta pe emailurile reale: aproximativ 75% - 90%.

## Bibliografie
- Wikipedia — Naive Bayes classifier
