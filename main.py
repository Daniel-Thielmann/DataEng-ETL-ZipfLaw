import pandas as pd
from pathlib import Path
from pre_processing import download_gutenberg, clean_gutenberg_text, extract_n_words
from llm_generation import generate_artificial_text
from zipf_model import compute_frequencies, compute_zipf_alpha
from statistical_analysis import (
    plot_zipf_comparison,
    plot_alpha_histogram,
    perform_ks_test,
)


def main():
    try:
        csv_path = Path(__file__).with_name("gutenberg_ids.csv")
        df_ids = pd.read_csv(csv_path)
        if "book_id" in df_ids.columns:
            GUTENBERG_IDS = df_ids["book_id"].tolist()
        elif "id" in df_ids.columns:
            GUTENBERG_IDS = df_ids["id"].tolist()
        else:
            raise KeyError("book_id")
    except FileNotFoundError:
        print("Arquivo gutenberg_ids.csv não encontrado.")
        return
    except KeyError:
        print("O arquivo gutenberg_ids.csv precisa ter a coluna 'book_id' ou 'id'.")
        return

    alphas_cn = []
    alphas_ca = []

    # Vamos guardar os dados de 1 livro específico para plotar o gráfico Log-Log (ex: o primeiro)
    sample_book_id = GUTENBERG_IDS[0]
    sample_data = {}

    print("=== INICIANDO PIPELINE DE MODELAGEM ===")

    for i, b_id in enumerate(GUTENBERG_IDS, start=1):
        print(f"\n--- Processando ID: {b_id} ({i}/100) ---")

        raw_text = download_gutenberg(b_id)
        if not raw_text:
            continue

        cleaned_text = clean_gutenberg_text(raw_text)
        human_words = extract_n_words(cleaned_text, n=25000)

        if human_words:
            artificial_words = generate_artificial_text(b_id, human_words)

            ranks_cn, freqs_cn = compute_frequencies(human_words)
            alpha_cn, _, _ = compute_zipf_alpha(ranks_cn, freqs_cn)
            alphas_cn.append(alpha_cn)

            ranks_ca, freqs_ca = compute_frequencies(artificial_words)
            alpha_ca, _, _ = compute_zipf_alpha(ranks_ca, freqs_ca)
            alphas_ca.append(alpha_ca)

            print(f"Alpha Humano (CN): {alpha_cn:.4f} | Alpha IA (CA): {alpha_ca:.4f}")

            # Salva os arrays do primeiro livro para gerar o gráfico log-log depois
            if b_id == sample_book_id:
                sample_data = {
                    "r_cn": ranks_cn,
                    "f_cn": freqs_cn,
                    "a_cn": alpha_cn,
                    "r_ca": ranks_ca,
                    "f_ca": freqs_ca,
                    "a_ca": alpha_ca,
                }

    print(f"\n=== FASE 4: ANÁLISE ESTATÍSTICA E VISUALIZAÇÃO ===")

    # 1. Gráfico Log-Log de um Livro de Amostra
    if sample_data:
        plot_zipf_comparison(
            sample_data["r_cn"],
            sample_data["f_cn"],
            sample_data["a_cn"],
            sample_data["r_ca"],
            sample_data["f_ca"],
            sample_data["a_ca"],
            sample_book_id,
        )

    # 2. Histograma Geral
    plot_alpha_histogram(alphas_cn, alphas_ca)

    # 3. Teste KS
    perform_ks_test(alphas_cn, alphas_ca)

    print("\nPipeline 100% Concluído com Sucesso!")


if __name__ == "__main__":
    main()
