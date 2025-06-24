import streamlit as st
from RAG import answer_question

st.set_page_config(page_title="Tindle RAG Chat", page_icon="💬", layout="centered")

st.title("💬 Tindle RAG Chat")
st.markdown("Posez une question sur le droit fiscal, l'assistant RAG vous répondra en s'appuyant sur les sources indexées.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Affichage de l'historique du chat
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])

# Saisie utilisateur
if prompt := st.chat_input("Votre question..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.spinner("Recherche de la réponse..."):
        try:
            response = answer_question(prompt)
        except Exception as e:
            response = f"Erreur lors de l'appel au RAG : {e}"
    st.session_state["messages"].append({"role": "assistant", "content": response})
    st.rerun()
