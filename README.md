# Dynamika zawodników defensywnych — symulacja i optymalizacja

Projekt modeluje zachowanie obrońców w sytuacji ataku na bramkę i szuka optymalnych parametrów sił działających na zawodników metodą **gradient ascent**. Implementacja znajduje się w notebooku Jupyter [`Marchwiak_Myszka.ipynb`](Marchwiak_Myszka.ipynb).

## Opis

Model opisuje ruch obrońców jako sumę trzech składników:

| Siła | Znaczenie | Wzór (uproszczony) |
|------|-----------|-------------------|
| \(F_{\text{pos}}\) | Dążenie do pozycji wyjściowej (prawo Hooke'a) | \(-k_{\text{pos}} \cdot (\mathbf{r}_i - \mathbf{r}_{\text{pos}})\) |
| \(F_{\text{opp}}\) | Reakcja na napastników (przyciąganie \(1/r^2\)) | \(k_{\text{opp}} \cdot (\mathbf{r}_{\text{opp}} - \mathbf{r}_i) / (\|\mathbf{r}\|^2 + \varepsilon)\) |
| \(F_{\text{team}}\) | Odpychanie od kolegów ( Coulomb ) | \(\sum -k_{\text{team}} \cdot (\mathbf{r}_j - \mathbf{r}_i) / (\|\mathbf{r}\|^2 + \varepsilon)\) |

Symulacja obejmuje:

- **7 obrońców** i **6 napastników** na boisku 68×105 m,
- ruch piłki, podania i próby odbioru,
- animację meczu w matplotlib,
- optymalizację parametrów \(k_{\text{pos}}, k_{\text{opp}}, k_{\text{team}}\) w celu maksymalizacji skuteczności obrony.

## Wymagania

- Python 3.10+
- Jupyter Notebook / JupyterLab

Zależności (patrz [`requirements.txt`](requirements.txt)):

```
numpy
matplotlib
plotly
```

`plotly` jest opcjonalne, ale potrzebne do **interaktywnego wykresu 3D** (obracanie, zoom, podpowiedzi).

## Instalacja

```bash
git clone <url-repozytorium>
cd rrwt
pip install -r requirements.txt
```

## Uruchomienie

1. Uruchom Jupyter w katalogu projektu:

```bash
jupyter notebook
```

2. Otwórz `Marchwiak_Myszka.ipynb`.
3. Wykonaj wszystkie komórki (**Run All**).

### Kolejność komórek

| Komórka | Zawartość |
|---------|-----------|
| 1 | Opis modelu (markdown) |
| 2 | Symulacja, klasy zawodników, animacja |
| 3 | Opis optymalizacji (markdown) |
| 4 | Gradient ascent, wykresy 2D i 3D |
| 5 | Wnioski |

### Interaktywny wykres 3D

Jeśli wykres nie obraca się po uruchomieniu, w komórce z optymalizacją odkomentuj:

```python
%pip install plotly
```

i uruchom komórkę ponownie.

## Parametry symulacji

| Parametr | Domyślna wartość | Opis |
|----------|------------------|------|
| `k_pos` | 1.0 | Siła dążenia do pozycji |
| `k_opp` | 5.0 | Siła reakcji na napastników |
| `k_team` | 1.0 | Siła odpychania od kolegów |
| `DELTA_T` | 0.3 | Krok czasowy symulacji |
| `ANIMATION_STEPS` | 2000 | Maks. kroków animacji |
| `OPTIMIZATION_STEPS` | 800 | Maks. kroków w pojedynczej symulacji (optymalizacja) |

Optymalizacja domyślnie: 50 iteracji, 50 symulacji na ocenę wartości funkcji celu.

## Przykładowe wyniki

Po optymalizacji (wyniki z ostatniego uruchomienia — mogą się nieco różnić ze względu na losowość):

| Metryka | Wartość |
|---------|---------|
| Skuteczność obrońców | **96,0%** |
| \(k_{\text{pos}}\) | 0,91 |
| \(k_{\text{opp}}\) | 6,81 |
| \(k_{\text{team}}\) | 0,42 |

Wnioski: skuteczniejsza jest **silniejsza presja na piłkę** i **słabsze odpychanie** między obrońcami niż przy parametrach startowych.

## Struktura repozytorium

```
rrwt/
├── Marchwiak_Myszka.ipynb   # główny notebook
├── requirements.txt         # zależności Python
└── README.md                # ten plik
```

## Źródła

- Jonathan Wilson — *Odwrócona piramida. Historia taktyki piłkarskiej*
- [Prawo Hooke'a](https://pl.wikipedia.org/wiki/Prawo_Hooke’a)
- [Prawo powszechnego ciążenia](https://pl.wikipedia.org/wiki/Prawo_powszechnego_ciążenia)
- [Prawo Coulomba](https://pl.wikipedia.org/wiki/Prawo_Coulomba)
- [Gradient descent / ascent](https://en.wikipedia.org/wiki/Gradient_descent)
- [Algorytm Boids](https://en.wikipedia.org/wiki/Boids)
