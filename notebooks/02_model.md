1. Wczytanie danych tak jak w 01_DA.ipynb, reshape, labels.
2. Preprocessing: standaryzacja dla wszystkich próbek per kanał per epoka czasowa,
   wywalić label == -1, wydzielić dodatkowo train.csv na 80/20, by mieć zbiór walidacyjny, zbiór
   testowy ju mamy w test.csv.
3. Trenowanie modelu, na poczatek moze jakies proste z scikit-learn, by miec punkt odniesienia, a potem
   Deep ML, Sieci konwolucyjne, cross entropy loss
4. Ewaluacja: accuracy (klasy są zbalansowane), F1, confusion matrix
5. Analiza błędów
6. tSNE na embeddingach, dla deep ML wyciągnąć output dla danych testowych z warstwy przed softmaxem,
   zwizualizować tSNE, czy sieć nauczyła się separować klasy
