"""
app.py
Streamlit dashboard — 🌱 IBM Granite - DIP Diet RAG Agent
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DIP Diet RAG Agent",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2e7d32;
        text-align: center;
        padding-bottom: 0.25rem;
    }
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f1f8e9;
        border-left: 4px solid #66bb6a;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
    }
    .meal-card {
        background: #ffffff;
        border: 1px solid #c8e6c9;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 14px;
    }
    .meal-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1b5e20;
        margin-bottom: 8px;
    }
    .rule-badge {
        background: #e8f5e9;
        color: #2e7d32;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Load vector store (cached) — uses Chroma's built-in fast embedder ────────
@st.cache_resource(show_spinner="Loading DIP knowledge base…")
def load_collection():
    import chromadb
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    client = chromadb.PersistentClient(path="./chroma_db")
    ef = DefaultEmbeddingFunction()
    return client.get_or_create_collection(name="dip_diet", embedding_function=ef)


def get_context(query: str, collection) -> str:
    results = collection.query(query_texts=[query], n_results=5)
    docs = results.get("documents", [[]])[0]
    return "\n\n".join(docs)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/51/IBM_logo.svg/320px-IBM_logo.svg.png",
        width=90,
    )
    st.markdown("## ⚙️ Your Profile")

    weight = st.number_input(
        "🏋️ Body Weight (kg)",
        min_value=30.0,
        max_value=200.0,
        value=70.0,
        step=0.5,
        help="Used to compute your personalised DIP food quantities.",
    )

    conditions = st.text_area(
        "🩺 Health Conditions",
        placeholder="e.g. Type 2 Diabetes, Hypertension…",
        height=80,
    )

    fruit_grams  = weight * 10
    veggie_grams = weight * 5

    st.markdown("---")
    st.markdown("### 📊 Your Daily Targets")

    st.markdown(
        f"""
        <div class="metric-card">
            🍎 <strong>Morning Fruits</strong><br/>
            <span style="font-size:1.5rem;font-weight:700;color:#388e3c;">{fruit_grams:.0f} g</span>
            <span style="color:#888;font-size:0.85rem;"> ({weight:.1f} kg × 10)</span>
        </div>
        <div class="metric-card">
            🥗 <strong>Raw Veggies / Meal (Plate 1)</strong><br/>
            <span style="font-size:1.5rem;font-weight:700;color:#388e3c;">{veggie_grams:.0f} g</span>
            <span style="color:#888;font-size:0.85rem;"> ({weight:.1f} kg × 5)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        "<small>⚡ Powered by **IBM Granite** + **Groq** · RAG via ChromaDB</small>",
        unsafe_allow_html=True,
    )


# ── Main header ───────────────────────────────────────────────────────────────
st.markdown(
    '<div class="main-title">🌱 IBM Granite — DIP Diet RAG Agent</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Disciplined & Intelligent Plan · Powered by IBM watsonx + Groq · RAG with ChromaDB</div>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["💬 Conversational Assistant", "📅 Visual Meal Plan"])


# ════════════════════════════════════════════════════════════════════════════
#  TAB 1 — Conversational Assistant
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Ask anything about the DIP Diet")
    st.markdown(
        "The agent retrieves relevant DIP rules, runs them through **IBM Granite** for precise "
        "nutritional math, then uses **Groq (gpt-oss-120b)** to craft a clear, personalised answer."
    )

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input row
    with st.form("query_form", clear_on_submit=True):
        user_input = st.text_input(
            "💬 Your question",
            placeholder="e.g. What should I eat for lunch as a 75 kg diabetic?",
        )
        submitted = st.form_submit_button("🚀 Submit", use_container_width=True)

    if submitted and user_input.strip():
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Retrieving context → 🧠 IBM Granite → ✨ Groq…"):
                try:
                    from agent import run_rag_agent

                    collection = load_collection()
                    context    = get_context(user_input, collection)
                    answer    = run_rag_agent(
                        user_query=user_input,
                        weight=weight,
                        conditions=conditions,
                        context=context,
                    )
                except Exception as exc:
                    answer = f"⚠️ **Error:** {exc}"

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

    elif submitted:
        st.warning("Please type a question before submitting.")


# ════════════════════════════════════════════════════════════════════════════
#  TAB 2 — Visual Meal Plan
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📅 Your Personalised DIP Meal Plan")
    st.markdown(
        f"Quantities are auto-calculated for **{weight:.1f} kg** body weight using DIP formulas."
    )

    # ── Morning ──────────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="meal-card">
            <div class="meal-header">🌅 Breakfast — Until 12:00 PM</div>
            <span class="rule-badge">FRUITS ONLY</span>&nbsp;
            <span class="rule-badge">NO JUICE</span>&nbsp;
            <span class="rule-badge">BEFORE NOON</span>
            <br/><br/>
            <strong>Target: {fruit_grams:.0f} g of fresh seasonal fruits</strong>
            <br/>Formula: {weight:.1f} kg × 10 = {fruit_grams:.0f} g
            <br/><br/>
            <strong>Sample combinations reaching {fruit_grams:.0f} g:</strong>
            <ul>
                <li>🍉 Watermelon {fruit_grams*0.4:.0f} g &nbsp;+&nbsp; 🍌 Banana {fruit_grams*0.3:.0f} g &nbsp;+&nbsp; 🍎 Apple {fruit_grams*0.3:.0f} g</li>
                <li>🥭 Mango {fruit_grams*0.4:.0f} g &nbsp;+&nbsp; 🍐 Pear {fruit_grams*0.3:.0f} g &nbsp;+&nbsp; 🍊 Orange {fruit_grams*0.3:.0f} g</li>
                <li>🍈 Papaya {fruit_grams*0.5:.0f} g &nbsp;+&nbsp; 🍇 Grapes {fruit_grams*0.25:.0f} g &nbsp;+&nbsp; 🥝 Kiwi {fruit_grams*0.25:.0f} g</li>
            </ul>
            <small style="color:#888;">✅ Drink 2–3 glasses of plain water before eating. Mix fruits freely to hit your target.</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Lunch ─────────────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="meal-card">
            <div class="meal-header">☀️ Lunch — 12:00 PM to 4:00 PM</div>
            <span class="rule-badge">TWO-PLATE SYSTEM</span>&nbsp;
            <span class="rule-badge">PLATE 1 FIRST</span>&nbsp;
            <span class="rule-badge">PLANT-BASED</span>
            <br/><br/>
            <strong>🥗 Plate 1 — Raw Salad ({veggie_grams:.0f} g — MUST finish before Plate 2)</strong>
            <br/>Formula: {weight:.1f} kg × 5 = {veggie_grams:.0f} g raw vegetables
            <ul>
                <li>🥒 Cucumber {veggie_grams*0.3:.0f} g</li>
                <li>🍅 Tomato {veggie_grams*0.3:.0f} g</li>
                <li>🥕 Carrot {veggie_grams*0.2:.0f} g</li>
                <li>🌱 Radish (Mooli) {veggie_grams*0.2:.0f} g</li>
            </ul>
            <strong>🍽️ Plate 2 — Cooked Plant-Based Meal (ONLY after finishing Plate 1)</strong>
            <ul>
                <li>2 whole wheat rotis / 1 cup brown rice / 1 cup millets</li>
                <li>1 cup cooked dal (lentils) or beans</li>
                <li>1 cup sabzi (cooked vegetables, minimal oil &amp; salt)</li>
            </ul>
            <small style="color:#888;">❌ No dairy, no animal products, minimal oil (&lt;1 tsp), minimal salt.</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Evening Snack ─────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="meal-card">
            <div class="meal-header">🌤️ Evening Snack — 4:00 PM to 5:30 PM (Optional)</div>
            <span class="rule-badge">LIGHT</span>&nbsp;
            <span class="rule-badge">WHOLE FOODS</span>
            <br/><br/>
            <ul>
                <li>🌰 4–5 soaked almonds + 2 walnuts</li>
                <li>🥥 Coconut water (1 glass)</li>
                <li>🍎 Small serving of fresh fruit</li>
                <li>🌿 Herbal tea (no milk, no sugar)</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Dinner ────────────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="meal-card">
            <div class="meal-header">🌙 Dinner — STRICTLY Before 8:00 PM</div>
            <span class="rule-badge">BEFORE 8 PM</span>&nbsp;
            <span class="rule-badge">TWO-PLATE SYSTEM</span>&nbsp;
            <span class="rule-badge">LIGHT MEAL</span>
            <br/><br/>
            <strong>🥗 Plate 1 — Raw Salad ({veggie_grams:.0f} g — same as lunch)</strong>
            <ul>
                <li>🥒 Cucumber + 🍅 Tomato + 🥕 Carrot + 🌱 Radish = {veggie_grams:.0f} g total</li>
            </ul>
            <strong>🍽️ Plate 2 — Light Cooked Meal (ONLY after finishing Plate 1)</strong>
            <ul>
                <li>Moong dal khichdi with vegetables</li>
                <li>Vegetable soup + 1 millet roti</li>
                <li>Steamed vegetables + small portion of brown rice</li>
            </ul>
            <small style="color:#f44336;font-weight:600;">⚠️ CRITICAL: Finish dinner before 8:00 PM. Eating after 8 PM is NOT allowed on DIP Diet.</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Foods to Avoid ────────────────────────────────────────────────────────
    with st.expander("🚫 Foods to Avoid Completely on DIP Diet"):
        avoid_cols = st.columns(2)
        avoid_list_1 = [
            "❌ All packed / processed foods",
            "❌ Refined sugar & sweets",
            "❌ Dairy (milk, paneer, curd, ghee)",
            "❌ Meat, fish, eggs",
            "❌ White rice & maida products",
        ]
        avoid_list_2 = [
            "❌ Cold drinks & packaged juices",
            "❌ Alcohol & tobacco",
            "❌ Deep-fried foods",
            "❌ Artificial sweeteners",
            "❌ Fast food & restaurant junk",
        ]
        with avoid_cols[0]:
            for item in avoid_list_1:
                st.markdown(item)
        with avoid_cols[1]:
            for item in avoid_list_2:
                st.markdown(item)

    # ── Healthy Substitutions ─────────────────────────────────────────────────
    with st.expander("✅ Healthy Substitutions"):
        subs = {
            "Cow/buffalo milk": "Almond milk or coconut milk",
            "Refined white sugar": "Dates, jaggery, or ripe banana",
            "White rice": "Brown rice, millets (jowar/bajra/ragi)",
            "Maida (all-purpose flour)": "Whole wheat flour or ragi flour",
            "Butter / ghee": "Cold-pressed coconut oil (minimal)",
            "Packaged snacks": "Fresh fruits, soaked nuts, sprouts",
            "Tea/coffee with milk & sugar": "Herbal teas, lemon water, coconut water",
        }
        for original, substitute in subs.items():
            st.markdown(f"**{original}** → {substitute}")
