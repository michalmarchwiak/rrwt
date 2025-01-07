import numpy as np

def calculate_total_force(
    r_i, r_goal_i, r_opp, teammates, k_goal, k_opp, k_team, epsilon=1e-6
):
    """
    Oblicza całkowitą siłę działającą na zawodnika i.

    Parametry:
        r_i (np.ndarray): Pozycja zawodnika i (np. [x, y]).
        r_goal_i (np.ndarray): Wyznaczona pozycja zawodnika i (np. [x, y]).
        r_opp (np.ndarray): Pozycja przeciwnika (np. [x, y]).
        teammates (list of np.ndarray): Lista pozycji pozostałych kolegów z drużyny.
        k_goal (float): Współczynnik siły dążenia do celu.
        k_opp (float): Współczynnik siły przyciągania do przeciwnika.
        k_team (float): Współczynnik siły odpychania między zawodnikami.
        epsilon (float): Mała wartość zapobiegająca dzieleniu przez zero (domyślnie 1e-6).

    Zwraca:
        np.ndarray: Wektor całkowitej siły działającej na zawodnika i (np. [F_x, F_y]).
    """
    # Siła dążenia do wyznaczonej pozycji (celu)
    F_goal = -k_goal * (r_i - r_goal_i)

    # Siła przyciągania do przeciwnika
    r_diff_opp = r_opp - r_i
    distance_opp = np.linalg.norm(r_diff_opp)
    F_opp = (k_opp * r_diff_opp) / (distance_opp**2 + epsilon)

    # Siła odpychania od kolegów z drużyny
    F_team = np.zeros_like(r_i)
    for r_j in teammates:
        r_diff_team = r_j - r_i
        distance_team = np.linalg.norm(r_diff_team)
        if distance_team > 0:  # Unikamy dzielenia przez zero
            F_team += (-k_team * r_diff_team) / (distance_team**2 + epsilon)

    # Łączna siła
    F_total = F_goal + F_opp + F_team

    return F_total

# Przykład użycia
if __name__ == "__main__":
    # Pozycja zawodnika i
    r_i = np.array([10, 20])

    # Wyznaczona pozycja celu
    r_goal_i = np.array([15, 25])

    # Pozycja przeciwnika
    r_opp = np.array([30, 35])

    # Pozycje pozostałych kolegów z drużyny
    teammates = [
        np.array([12, 22]),
        np.array([14, 19]),
        np.array([11, 18]),
    ]

    # Współczynniki sił
    k_goal = 1.0  # siła dążenia do celu
    k_opp = 2.0   # siła przyciągania do przeciwnika
    k_team = 0.5  # siła odpychania od kolegów z drużyny

    # Obliczenie całkowitej siły
    F_total = calculate_total_force(r_i, r_goal_i, r_opp, teammates, k_goal, k_opp, k_team)

    # Wyświetlenie wyniku
    print("Całkowita siła działająca na zawodnika i:", F_total)

