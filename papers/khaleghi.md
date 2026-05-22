Sieć konwolucyjna na surowym sygnale, po preprocessingu: standaryzacja z-score.

Warstwa 1 - depth-wise conv (po czasie, per kanał)
np. 4 kernele per kanał, kernel_size=5, konwolucja 1D po 256 próbkach.
(B, 14, 256) -> (B, 56, ~256)
Wyłapuje lokalne wzorce czasowe na kazdym kanale osobno.

Warstwa 2 - pointwise conv (kernel_size=1, łączenie kanałów)
Dla kazdej pozycji czasowej łączy informacje z 56 filtrów w nową reprezentację.
(B, 56, ~256) -> (B, N, ~256)
Uczy się relacji między kanałami, np. co O1 i AF3 razem mówią o cyfrze.

Warstwa 3 (opcjonalna) - zwykła Conv1D
Wyciąga wzorce wyzszego rzedu z juz istniejacych reprezentacji.

Global Average Pooling (GAP) - Średnia po osi czasu (bez parametrów)
(B, N, ~256) -> (B, N)

Dense(128) -> Dense(10) + CrossEntropyLoss
