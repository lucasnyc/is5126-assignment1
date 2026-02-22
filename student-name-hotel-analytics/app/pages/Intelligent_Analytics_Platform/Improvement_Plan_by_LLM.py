import streamlit as st
import pandas as pd
import sqlite3
from pathlib import Path

# -- DB connection ------------------------------------------------------------
if "DB_PATH" in st.session_state:
    DB_PATH = Path(st.session_state["DB_PATH"])
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    DB_PATH = PROJECT_ROOT / "data" / "reviews_sample.db"

conn = sqlite3.connect(str(DB_PATH))

# -- Constants ----------------------------------------------------------------
ASPECT_COLS = [
    "service", "cleanliness", "value",
    "location_rating", "sleep_quality", "rooms",
]
ASPECT_LABELS = {
    "service":         "Service",
    "cleanliness":     "Cleanliness",
    "value":           "Value",
    "location_rating": "Location",
    "sleep_quality":   "Sleep Quality",
    "rooms":           "Rooms",
}

# Fixed model
HF_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# -- Page header --------------------------------------------------------------
st.title("Hotel Improvement Plan")
st.caption(
    "Enter a Hotel ID to identify its weakest service aspect and generate "
    "a targeted improvement plan using a local open-source language model."
)
st.markdown("---")

# -- Sidebar / Config ---------------------------------------------------------
with st.sidebar:
    st.header("Model Configuration")
    st.markdown(f"**Model:** `{HF_MODEL}`")
    st.caption("TinyLlama 1.1 B causal language model. Downloaded from Hugging Face on first use.")
    max_new_tokens = st.slider("Max new tokens", 128, 512, 256, step=32)
    n_reviews = st.slider("Number of weak reviews to analyse", 3, 10, 5)

# -- Model loader (cached across reruns) --------------------------------------
@st.cache_resource(show_spinner="Loading TinyLlama — this may take a minute on first use …")
def load_pipeline():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    model = AutoModelForCausalLM.from_pretrained(HF_MODEL)
    return tokenizer, model

# -- Filter -------------------------------------------------------------------
hotel_id_input = st.text_input(
    "Offering ID (Hotel ID)",
    value="",
    placeholder="e.g. 93466",
    key="hotel_id_main",
)

if not hotel_id_input.strip():
    st.info("Enter a Hotel ID above to begin.")
    st.stop()

try:
    hotel_id = int(hotel_id_input.strip())
except ValueError:
    st.error("Please enter a valid numeric Offering ID.")
    st.stop()

# -- Load data ----------------------------------------------------------------
@st.cache_data
def load_hotel_reviews(_conn, hid: int):
    return pd.read_sql(
        """
        SELECT overall, service, cleanliness, value, location_rating,
               sleep_quality, rooms, title, text, review_date
        FROM reviews
        WHERE offering_id = ?
          AND overall BETWEEN 1 AND 5
        ORDER BY review_date
        """,
        _conn,
        params=[hid],
    )

hotel_df = load_hotel_reviews(conn, hotel_id)

if hotel_df.empty:
    st.warning(f"No reviews found for Hotel ID {hotel_id}.")
    st.stop()

st.success(f"Hotel {hotel_id}  {len(hotel_df):,} reviews loaded.")

# -- Identify weakest aspect --------------------------------------------------
aspect_means = hotel_df[ASPECT_COLS].mean()
worst_col    = aspect_means.idxmin()
worst_label  = ASPECT_LABELS[worst_col]
worst_score  = aspect_means[worst_col]

st.markdown("---")
st.subheader("Aspect Performance")

cols = st.columns(len(ASPECT_COLS))
for col_widget, asp_col in zip(cols, ASPECT_COLS):
    score = aspect_means[asp_col]
    delta_color = "inverse" if asp_col == worst_col else "normal"
    col_widget.metric(
        ASPECT_LABELS[asp_col],
        f"{score:.2f}",
        delta=f"{'WEAKEST' if asp_col == worst_col else ''}",
        delta_color=delta_color,
    )

st.error(
    f"Weakest aspect: **{worst_label}** (avg {worst_score:.2f} / 5.0). "
    "The improvement plan below targets this area."
)

# -- Lowest-scoring reviews for the weakest aspect ----------------------------
st.markdown("---")
st.subheader(f"Lowest-Rated Reviews for: {worst_label}")

weak_reviews = (
    hotel_df[hotel_df[worst_col].notna() & hotel_df["text"].notna()]
    .nsmallest(n_reviews, worst_col)
    .reset_index(drop=True)
)

if weak_reviews.empty:
    st.warning("No reviews with text found for this aspect.")
    st.stop()

for i, row in weak_reviews.iterrows():
    with st.expander(
        f"Review {i + 1}  {worst_label} score: {row[worst_col]:.0f}/5"
        f"  |  Overall: {row['overall']:.0f}/5"
        f"  |  {str(row['review_date'])[:10]}"
    ):
        if pd.notna(row.get("title")):
            st.markdown(f"**{row['title']}**")
        st.write(row["text"])

# -- Build prompt -------------------------------------------------------------
# Keep snippets short to stay within model token limits (esp. flan-t5-base)
MAX_SNIPPET_CHARS = 300

review_snippets = []
for i, row in weak_reviews.iterrows():
    snippet = str(row["text"])
    if len(snippet) > MAX_SNIPPET_CHARS:
        snippet = snippet[:MAX_SNIPPET_CHARS] + "..."
    review_snippets.append(
        f"Review {i + 1} ({worst_label} score {row[worst_col]:.0f}/5): {snippet}"
    )

reviews_block = "\n".join(review_snippets)

prompt = (
    f"Task: Write a hotel improvement plan with exactly four sections: "
    f"1. Root Cause Summary, "
    f"2. Quick Wins (within 1 month), "
    f"3. Medium-Term Improvements (1-6 months), "
    f"4. Success Metrics (KPIs). "
    f"Be concise and base every point on the guest reviews below.\n\n"
    f"Context: You are a hospitality consultant reviewing {len(weak_reviews)} "
    f"low-scoring guest reviews for the '{worst_label}' aspect of Hotel {hotel_id}.\n\n"
    f"{reviews_block}\n\n"
    f"Now write the four-section improvement plan:"
)

# -- Generate improvement plan ------------------------------------------------
st.markdown("---")
st.subheader("AI-Generated Improvement Plan")

generate_btn = st.button("Generate Improvement Plan", type="primary")

if generate_btn:
    tokenizer, model = load_pipeline()

    with st.spinner("Running TinyLlama locally …"):
        try:
            import torch

            # TinyLlama is a chat model — apply the chat template
            # so the model receives proper <|system|>/<|user|>/<|assistant|> tokens
            system_msg = (
                "You are a hospitality consultant. "
                "Your task is to write a structured hotel improvement plan. "
                "Do NOT summarise the reviews or answer questions. "
                "Output ONLY the four-section plan."
            )
            user_msg = (
                f"Here are {len(weak_reviews)} low-scoring guest reviews for the "
                f"'{worst_label}' aspect of Hotel {hotel_id}:\n\n"
                f"{reviews_block}\n\n"
                f"Write an improvement plan with exactly these four sections:\n"
                f"1. Root Cause Summary\n"
                f"2. Quick Wins (within 1 month)\n"
                f"3. Medium-Term Improvements (1-6 months)\n"
                f"4. Success Metrics (KPIs)\n\n"
                f"Be concise. Base every point on the reviews above."
            )
            messages = [
                {"role": "system",  "content": system_msg},
                {"role": "user",    "content": user_msg},
            ]
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(
                formatted,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
            plan_text = tokenizer.decode(
                new_tokens, skip_special_tokens=True
            ).strip()
            prompt_used = formatted

            if plan_text:
                st.session_state["improvement_plan"] = plan_text
                st.session_state["plan_hotel"]       = hotel_id
                st.session_state["plan_aspect"]      = worst_label
                st.session_state["plan_prompt"]      = prompt_used
            else:
                st.error("The model returned an empty response. Try increasing Max new tokens.")

        except Exception as exc:
            st.error(f"Model inference failed: {exc}")

# -- Display cached plan ------------------------------------------------------
if (
    "improvement_plan" in st.session_state
    and st.session_state.get("plan_hotel") == hotel_id
    and st.session_state.get("plan_aspect") == worst_label
):
    st.markdown(f"#### Improvement Plan for Hotel {hotel_id}  {worst_label}")
    st.markdown(st.session_state["improvement_plan"])

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            label="Download Plan (txt)",
            data=st.session_state["improvement_plan"],
            file_name=f"improvement_plan_hotel_{hotel_id}_{worst_label.replace(' ', '_')}.txt",
            mime="text/plain",
        )
    with col_dl2:
        st.download_button(
            label="Download Reviews Used (txt)",
            data="\n\n---\n\n".join(review_snippets),
            file_name=f"weak_reviews_hotel_{hotel_id}_{worst_label.replace(' ', '_')}.txt",
            mime="text/plain",
        )

    with st.expander("View prompt sent to model"):
        st.code(st.session_state.get("plan_prompt", prompt), language="text")
else:
    st.info("Click **Generate Improvement Plan** to run the local model.")