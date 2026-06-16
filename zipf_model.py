import numpy as np
from collections import Counter
from typing import cast
from scipy.stats import linregress


def compute_frequencies(words_list):
    """
    Recebe uma lista de palavras e retorna duas listas (rankings e frequências).
    A contagem é ordenada da palavra mais frequente para a menos frequente.
    """
    # Conta a ocorrência de cada palavra
    word_counts = Counter(words_list)

    # Ordena as frequências em ordem decrescente
    freqs = sorted(word_counts.values(), reverse=True)

    # Cria o array de rankings (1, 2, 3, ..., N)
    ranks = np.arange(1, len(freqs) + 1)

    return ranks, np.array(freqs)


def compute_zipf_alpha(ranks, freqs):
    """
    Aplica a regressão linear no espaço log-log para encontrar o coeficiente alpha.
    """
    # Converte os rankings e frequências para a escala logarítmica
    log_ranks = np.log(ranks)
    log_freqs = np.log(freqs)

    # Aplica Regressão Linear: log(f) = -alpha * log(r) + log(C)
    result = linregress(log_ranks, log_freqs)

    # O alpha é o valor absoluto da inclinação (slope) da reta
    slope = cast(float, result[0])
    intercept = cast(float, result[1])
    r_value = cast(float, result[2])
    alpha = abs(slope)

    return alpha, intercept, r_value**2
