import numpy as np
import matplotlib.pyplot as plt
from typing import cast
from scipy import stats


def plot_zipf_comparison(
    ranks_cn, freqs_cn, alpha_cn, ranks_ca, freqs_ca, alpha_ca, book_id
):
    """
    Plota o gráfico Log-Log comparando um texto CN com seu gêmeo CA.
    """
    plt.figure(figsize=(10, 6))

    # CN - Scatter e Linha de Tendência
    log_r_cn = np.log(ranks_cn)
    log_f_cn = np.log(freqs_cn)
    plt.scatter(
        log_r_cn,
        log_f_cn,
        alpha=0.5,
        label=f"Humano (CN) - α={alpha_cn:.4f}",
        color="blue",
        s=10,
    )
    m_cn, c_cn = np.polyfit(log_r_cn, log_f_cn, 1)
    plt.plot(log_r_cn, m_cn * log_r_cn + c_cn, color="darkblue", linestyle="--")

    # CA - Scatter e Linha de Tendência
    log_r_ca = np.log(ranks_ca)
    log_f_ca = np.log(freqs_ca)
    plt.scatter(
        log_r_ca,
        log_f_ca,
        alpha=0.5,
        label=f"IA (CA) - α={alpha_ca:.4f}",
        color="red",
        s=10,
    )
    m_ca, c_ca = np.polyfit(log_r_ca, log_f_ca, 1)
    plt.plot(log_r_ca, m_ca * log_r_ca + c_ca, color="darkred", linestyle="--")

    plt.title(f"Lei de Zipf: Humano vs IA (Livro ID: {book_id})")
    plt.xlabel("Log(Rank)")
    plt.ylabel("Log(Frequência)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Salva o gráfico
    plt.savefig(f"zipf_comparison_id_{book_id}.png")
    print(f"Gráfico log-log salvo como zipf_comparison_id_{book_id}.png")
    plt.close()


def plot_alpha_histogram(alphas_cn, alphas_ca):
    """
    Plota um histograma comparando a distribuição dos coeficientes alpha.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(
        alphas_cn,
        bins=15,
        alpha=0.6,
        label="Humanos (CN)",
        color="blue",
        edgecolor="black",
    )
    plt.hist(
        alphas_ca, bins=15, alpha=0.6, label="IA (CA)", color="red", edgecolor="black"
    )

    plt.axvline(
        np.mean(alphas_cn),
        color="darkblue",
        linestyle="dashed",
        linewidth=2,
        label=f"Média CN: {np.mean(alphas_cn):.2f}",
    )
    plt.axvline(
        np.mean(alphas_ca),
        color="darkred",
        linestyle="dashed",
        linewidth=2,
        label=f"Média CA: {np.mean(alphas_ca):.2f}",
    )

    plt.title("Distribuição dos Coeficientes Alpha de Zipf")
    plt.xlabel("Valor de Alpha (α)")
    plt.ylabel("Frequência (Quantidade de Livros)")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    plt.savefig("alphas_histogram.png")
    print("Histograma salvo como alphas_histogram.png")
    plt.close()


def perform_ks_test(alphas_cn, alphas_ca):
    """
    Aplica o teste Kolmogorov-Smirnov para 2 amostras.
    """
    result = stats.ks_2samp(alphas_cn, alphas_ca)
    stat = cast(float, result[0])
    p_value = cast(float, result[1])

    print("\n" + "=" * 40)
    print("RESULTADO DO TESTE KOLMOGOROV-SMIRNOV")
    print("=" * 40)
    print(f"Estatística KS: {stat:.4f}")
    print(f"P-value: {p_value:.4e}")

    if p_value < 0.05:
        print(
            "Conclusão: Rejeitamos a hipótese nula. As distribuições de CN e CA são ESTATISTICAMENTE DIFERENTES."
        )
    else:
        print(
            "Conclusão: Não podemos rejeitar a hipótese nula. As distribuições são estatisticamente semelhantes."
        )
    print("=" * 40)
