MNE filtracja nie wplywa na wynik klasyfikacji
bez: acc=0.2075 f1=0.1934, z: acc=0.2043 f1=0.1950

Train loss spada z 2.30 do 2.10. Krzywe się zbiegają. Róznica między trian-val acc nie jest
katastrofalna - 3% róznicy - nie ma duzego przeuczenia. To jest to co przewidziała EDA - słaby
sygnał, małe róznice między klasami. Bottleneck to dane, nie model.

Model radzi sobie na cyfrach 0, 1, 7 (wysoki recall: 0.39 / 0.29 / 0.37, najlepsze F1), a pada na
2, 3, 8, 9 (niski recall, F1~0.13). W EDA cyfra '1' odstawała na O1, a 7 w średniej po kanałach.
Klasyfikator znajduje sygnał tam, gdzie EDA go wskazała.
