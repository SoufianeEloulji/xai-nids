import json
from collections import Counter
import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

from src.storage.db import fetch_recent, fetch_stats

st.set_page_config(page_title="Détection d'intrusion réseau — Live", layout="wide")

MAX_ROWS = 200

# Bouton d'actualisation en direct
col_title, col_toggle = st.columns([3, 1])
with col_title:
    st.title("Détection d'intrusion réseau")
with col_toggle:
    st.write("") # Espace
    live_mode = st.toggle("Actualisation en direct", value=True, help="Désactivez pour cliquer sur une ligne sans être interrompu")

# Le rafraîchissement ne tourne que si le toggle est activé
if live_mode:
    st_autorefresh(interval=2000, key="refresh")

rows = fetch_recent(limit=MAX_ROWS)
stats = fetch_stats()

if not rows:
    st.info("En attente des premières prédictions... lancez simulator/simulate_traffic.py pour générer du trafic.")
    st.stop()

df = pd.DataFrame(rows)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

n_total = stats["total"]
n_attacks = sum(r["c"] for r in stats["by_label"] if r["label"] != "BENIGN")
attack_rate = (n_attacks / n_total * 100) if n_total else 0.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Flux analysés (total)", n_total)
col2.metric("Attaques détectées (total)", n_attacks)
col3.metric("Taux d'attaque", f"{attack_rate:.1f}%")
col4.metric("Confiance moyenne", f"{df['confidence'].mean() * 100:.1f}%")

st.divider()

left, right = st.columns([2, 1])

with left:
    st.subheader("Flux récents")
    display_df = df[["timestamp", "label", "confidence"]].copy()
    display_df["confidence"] = (display_df["confidence"] * 100).round(1).astype(str) + "%"

    def highlight_attacks(row):
        color = "background-color: #ffe5e5" if row["label"] != "BENIGN" else ""
        return [color] * len(row)

    sorted_display_df = display_df.sort_values("timestamp", ascending=False)

    # L'ajout de la "key" règle le problème des clics multiples
    event = st.dataframe(
        sorted_display_df.style.apply(highlight_attacks, axis=1),
        width="stretch",
        height=300,
        on_select="rerun",
        selection_mode="single-row",
        key="traffic_table" 
    )

    # --- LOGIQUE D'AFFICHAGE SHAP ---
    # Si une ligne est sélectionnée, on affiche son explication
    if len(event.selection.rows) > 0:
        st.error(" **Analyse de la ligne sélectionnée (SHAP)** ")
        selected_iloc = event.selection.rows[0]
        original_index = sorted_display_df.index[selected_iloc]
        selected_record = df.loc[original_index]
    # Sinon, on affiche toujours l'explication de la toute dernière ligne
    else:
        st.info(" **Dernière prédiction (SHAP)** - *Mettez sur pause et cliquez sur une ligne pour l'analyser* ")
        selected_record = df.iloc[-1]
        
    st.write(f"**Label prédit :** {selected_record['label']} | **Confiance :** {selected_record['confidence'] * 100:.1f}% | **Heure :** {selected_record['timestamp']}")
    
    top_features = json.loads(selected_record["top_features_json"])
    if top_features:
        feat_df = pd.DataFrame(top_features).sort_values("shap_value")
        feat_df["label_display"] = feat_df.apply(
            lambda r: f"{r['feature']} = {r['value']:.3g}" if r.get("value") is not None else r["feature"], 
            axis=1
        )

        fig_shap = px.bar(
            feat_df, x="shap_value", y="label_display", orientation="h",
            color="shap_value", color_continuous_scale=["#3b82f6", "#ef4444"]
        )
        fig_shap.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0), coloraxis_showscale=False, yaxis_title=None)
        
        # Ajout d'une clé ici aussi pour éviter le rechargement brutal du graphique
        st.plotly_chart(fig_shap, width="stretch", key="shap_chart") 
    else:
        st.caption("Pas d'explication SHAP disponible.")

with right:
    st.subheader("Répartition par label")
    label_counts = Counter(df["label"])
    label_df = pd.DataFrame(label_counts.items(), columns=["label", "count"]).sort_values("count", ascending=False)
    fig_pie = px.pie(label_df, names="label", values="count", hole=0.4)
    fig_pie.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_pie, width="stretch", key="pie_chart")

    st.subheader("Volume dans le temps")
    volume = df.set_index("timestamp").resample("5s").size().reset_index(name="count")
    fig_volume = px.line(volume, x="timestamp", y="count", markers=True)
    fig_volume.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_volume, width="stretch", key="volume_chart")