from pathlib import Path
import typer
from loguru import logger
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

from reconnaissance_demotion.config import FIGURES_DIR, PROCESSED_DATA_DIR, emotion_initials
from reconnaissance_demotion.modeling.predict import predict_top3

app = typer.Typer()

# ------------------ Interface graphique ------------------
def launch_gui(models_dict, tfidf_vectorizer):
    root = tk.Tk()
    root.title("Reconnaissance d'émotions - ML")

    # Cadre graphique
    frame_graph = tk.Frame(root)
    frame_graph.pack(side=tk.LEFT, padx=10, pady=10)

    fig, ax = plt.subplots(figsize=(5, 4))
    canvas_graph = FigureCanvasTkAgg(fig, master=frame_graph)
    canvas_graph.get_tk_widget().pack()

    # Cadre input
    frame_input = tk.Frame(root)
    frame_input.pack(side=tk.RIGHT, padx=10, pady=10)

    label_scoring = tk.Label(
        frame_input,
        text="Scoring Model",
        font=("Helvetica", 18, "bold"),
        fg="black",
        bg="white",
        bd=2,
        relief="solid",
        padx=15, pady=10
    )
    label_scoring.pack(pady=(5, 10))

    label_emotions = tk.Label(
        frame_input,
        text="Émotions possibles :\nJoie | Colère | Tristesse | Dégoût | Neutre | Peur | Surprise",
        font=("Helvetica", 12, "bold"),
        fg="black",
        justify=tk.CENTER,
        wraplength=300
    )
    label_emotions.pack(pady=10)

    label_instruction = tk.Label(
        frame_input,
        text="Entrez votre texte et appuyez sur Entrée pour prédire :",
        font=("Arial", 12),
        fg="black",
        justify=tk.LEFT
    )
    label_instruction.pack(pady=(0, 5))

    entry_input = tk.Entry(
        frame_input, width=50, font=("Arial", 12),
        bd=5, relief="groove", highlightthickness=2,
        highlightbackground="gray"
    )
    entry_input.pack(pady=5)
    entry_input.focus()

    label_predictions = tk.Label(frame_input, text="", justify=tk.LEFT, font=("Arial", 12))
    label_predictions.pack(pady=10)

    # Fonction On enter appuyé
    def on_enter(event):
        phrase = entry_input.get()
        if phrase.strip() == "":
            return
        entry_input.delete(0, tk.END)

        results = {}
        for model_name, model in models_dict.items():
            results[model_name] = predict_top3(model, tfidf_vectorizer, phrase)

        # Affichage texte Top3
        text_display = ""
        for model_name, top3_list in results.items():
            line = f"{model_name} : " + " | ".join(
                [f"{emotion_initials.get(emo, emo[0]).upper()} ({score:.2f}%)"
                 for emo, score in top3_list]
            )
            text_display += line + "\n"

        label_predictions.config(text=text_display)

        # Graphique
        ax.clear()
        model_names = list(results.keys())
        top1_confidences = [results[m][0][1] for m in model_names]
        top1_labels = [results[m][0][0] for m in model_names]

        bars = ax.bar(model_names, top1_confidences, color=["red", "blue", "green"])
        ax.set_ylim(0, 100)

        for bar, lbl in zip(bars, top1_labels):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1,
                    lbl, ha='center')

        ax.set_ylabel("Confiance (%)")

        ax_top = ax.secondary_xaxis('top')
        ax_top.set_xticks(range(len(top1_labels)))
        ax_top.set_xticklabels(top1_labels, rotation=0, fontsize=10)
        ax_top.set_xlabel("Émotions prédites")

        canvas_graph.draw()

        if not hasattr(on_enter, "legend_label"):
            on_enter.legend_label = tk.Label(
                frame_input,
                text="Légende :\nJ = Joie | C = Colère | T = Tristesse | D = Dégoût\nN = Neutre | P = Peur | S = Surprise",
                font=("Arial", 10),
                justify=tk.CENTER
            )
            on_enter.legend_label.pack(pady=(10, 0))

    entry_input.bind("<Return>", on_enter)

    def on_closing():
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

# ------------------ Commande CLI ------------------
@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
):
    logger.info("Interface graphique prête à être lancée.")
    logger.info("Utilisez launch_gui(models, tfidf) depuis main.py pour démarrer.")
    logger.success("Module plots chargé.")

if __name__ == "__main__":
    app()
