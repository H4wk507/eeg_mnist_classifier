Sygnał EEG - napięcie elektryczne wytwarzane przez aktywność zbiorowiska neuronów, mierzone za pomocą
elektrod umieszczanych na skórze głowy.

Functional Connectivity - statystyczna zaleznosc między sygnałami z róznych obszarów mózgu.
Miedzy kanałami EEG u nas.
Jeśli kanał O1 i O2 wykazują skoordynowaną aktywność, to mówimy, ze są funkcjonalnie połączone.
Najprostszym sposobem oszacowania FC jest policzenie korelacji Pearsona między wszystkimi parami kanałów.
Otrzymujemy wtedy macierz 14x14, gdzie kazda wartość opisuje podobieństwo aktywności dwóch kanałów.
Korelacja ma jednak istotne ograniczenie: jest podatne na 'przewodzenie objętościowe', to znaczy sytuację,
w której jedno źródło aktywności neuronalnej jest jednocześnie rejestrowane przez kilka elektrod. Wtedy
dwa kanały mogą wyglądać na silnie połączone, mimo ze korelacja wynika głównie ze wspólnego źródła
sygnału albo bliskości elektrod.

Spectral Connectivity - Functional Connectivity liczone osobno dla wybranych pasm częstotliwości, np.
theta `4-8 Hz`, alpha `8-13 Hz`. W projekcie uzywamy `spectral_connectivity_epochs`, zeby policzyć
zalezności między parami kanałów EEG w danym paśmie. Wynik to macierz 14x14, gdzie kazda komórka opisuje
siłę połączenia między dwoma kanałami.

wPLI (weighted Phase Lag Index) - miara spectral connectivity oparta na przesunięciu fazowym między
sygnałami. Jest mniej podatna na przewodzenie objętościowe niz zwykła korelacja, bo skupia się głównie
na stabilnych, niezerowych opóźnieniach fazowych między kanałami.
Wartość bliska 0 oznacza słabą zalezność między kanałami, a 1 silniejszą zalezność fazową.
wPLI to po prostu bardziej EEG-friendly connectivity w paśmie częstotliwości.

W projekcie liczymy macierze wPLI w pasmach theta i alpha:

- osobno dla każdej klasy cyfry, aby sprawdzić, czy connectivity różni się między cyframi,
- oraz dla całego zbioru danych, aby zobaczyć ogólny wzorzec połączeń między kanałami.

`spread` to różnica między największą i najmniejszą wartością connectivity dla danej pary kanałów
między klasami cyfr. Mały `spread` oznacza, że connectivity prawie nie różnicuje klas.

DTW - to algorytm mierzący podobieństwo między dwoma szeregami czasowymi, który toleruje przesunięcia i
rozciągnięcia w czasie. Ta sama reakcja mózgu na cyfrę '3' raz wystąpi 200ms po bodźcu, raz 250ms.
DTW to dopasowywuje oraz zwraca koszt dopasowania. Im nizszy koszt, tym bardziej podobne sygnały.

W projekcie DTW słuzy do porównywania średnich przebiegów EEG dla róznych cyfr. Dla kazdej cyfry
najpierw liczymy średni sygnał w danym kanale, a potem sprawdzamy, jak bardzo średnie przebiegi cyfr
róznią się od siebie. Wynikiem jest macierz odległości między cyframi. Niska wartość DTW oznacza podobne
przebiegi, a wysoka, ze średnie sygnały dwóch cyfr są bardziej rózne.
DTW liczymy osobno dla kanałów O1 i O2 oraz jako średnią po wszystkich kanałach. Dzięki temu mozna sprawdzić,
czy róznice między cyframi są widoczne głównie w kanałach wzrokowych, czy raczej w innych obszarach.

MNE - biblioteka uzywana w projekcie do preprocessingu sygnału EEG.
W naszym pipeline MNE słuzy do zamiany surowych wartości ADC na wolty, usunięcia offsetu DC, filtrowania
sygnału oraz odrzucenia epok z duzymi artefaktami.
Stosujemy filtr notch 50 Hz, zeby usunąć zakłócenia sieciowe, filtr pasmowy 1-45 Hz, zeby zostawić główny
zakres EEG, oraz próg 150 µV, zeby odrzucić epoki o zbyt duzej amplitudzie.
