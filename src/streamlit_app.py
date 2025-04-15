# import streamlit as st
# from dotenv import load_dotenv
# import asyncio
# import os
# from rag_agent import run_agent, run_and_verify_agent, run_image_agent

# # --------------------------------------------------------------------------
# # Setup and configuration
# # --------------------------------------------------------------------------
# # Load environment variables from .env file.
# load_dotenv()

# # Set the Streamlit page configuration.
# st.set_page_config(page_title="Clinical Assistant Chat", layout="wide")

# # --------------------------------------------------------------------------
# # Sidebar: Display a sticky image.
# # --------------------------------------------------------------------------
# st.sidebar.image("data/present/clinical_tiger.png", caption="Clinical Tiger Assistant", width=250)

# # --------------------------------------------------------------------------
# # Main Title and Instructions
# # --------------------------------------------------------------------------
# st.title("🐅 Clinical Tiger Chat")
# st.markdown("""
# **How to Use This Chat:**

# - Enter your clinical question below.
# - Click **Generate Answer** for a fast response.
# - Click **Generate + Verify Answer** to double-check via reasoning.
# - Click **Retrieve Image** to get the associated image location and caption.

# **Disclaimer:** This is for general guidance only. Please consult professionals for medical advice.
# """)

# # --------------------------------------------------------------------------
# # Initialize chat history in session state.
# # --------------------------------------------------------------------------
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # --------------------------------------------------------------------------
# # Chat form for textual queries
# # --------------------------------------------------------------------------
# with st.form("chat_form", clear_on_submit=True):
#     user_input = st.text_area("Enter your question:")
#     # Create three columns for separate buttons.
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         baseline_submit = st.form_submit_button("Generate Answer")
#     with col2:
#         verify_submit = st.form_submit_button("Generate + Verify Answer")
#     with col3:
#         image_submit = st.form_submit_button("Retrieve Image")

# # --------------------------------------------------------------------------
# # Process text-based queries: Generate Answer / Generate + Verify Answer.
# # --------------------------------------------------------------------------
# if (baseline_submit or verify_submit) and user_input.strip():
#     st.session_state.chat_history.append({"role": "User", "text": user_input.strip()})
#     with st.spinner("Thinking..."):
#         try:
#             if verify_submit:
#                 # Run full chain-of-verification workflow.
#                 answer, references, eval_ratio = run_and_verify_agent(user_input.strip())
#                 answer += f" (Correctness: {eval_ratio})"
#             else:
#                 # Run the basic agent workflow.
#                 answer, references = run_agent(user_input.strip())
#             # Append the result to the conversation history.
#             st.session_state.chat_history.append({
#                 "role": "Clinical Tiger",
#                 "text": answer,
#                 "references": references
#             })
#         except Exception as e:
#             st.session_state.chat_history.append({
#                 "role": "Clinical Tiger",
#                 "text": f"An error occurred: {e}",
#                 "references": ""
#             })

# # --------------------------------------------------------------------------
# # Process image-based queries: Retrieve Image.
# # --------------------------------------------------------------------------
# if image_submit and user_input.strip():
#     with st.spinner("Retrieving image..."):
#         try:
#             # Run the image agent workflow.
#             image_path, caption = run_image_agent(user_input.strip())

#             # Log the image result in the conversation history.
#             st.session_state.chat_history.append({
#                 "role": "Clinical Tiger",
#                 "text": caption,
#                 "image_path": image_path
#             })
#         except Exception as e:
#             st.session_state.chat_history.append({
#                 "role": "Clinical Tiger",
#                 "text": f"An error occurred retrieving the image: {e}"
#             })

# # --------------------------------------------------------------------------
# # Display conversation history
# # --------------------------------------------------------------------------
# st.markdown("## Conversation")
# for msg in st.session_state.chat_history:
#     if msg["role"] == "User":
#         st.markdown(f"**User:** {msg['text']}")
#     elif msg["role"] == "Clinical Tiger":
#         st.markdown(f"**🐅:** {msg['text']}")
#         if msg.get("references"):
#             with st.expander("Show References"):
#                 refs = msg["references"].split("\n")
#                 st.markdown("\n".join(f"- {ref}" for ref in refs if ref.strip()))
#         # If an image was returned, display it below the text.
#         if msg.get("image_path"):
#             st.image(msg["image_path"], caption=msg.get("caption", ""), use_container_width=True)


import streamlit as st
from dotenv import load_dotenv
import os
from rag_agent import Agent, VerifiedAgent, ImageAgent

# --------------------------------------------------------------------------
# Setup and configuration
# --------------------------------------------------------------------------
load_dotenv()
st.set_page_config(page_title="Clinical Assistant Chat", layout="wide")

# --------------------------------------------------------------------------
# Sidebar: Display a sticky image.
# --------------------------------------------------------------------------
st.sidebar.image("data/present/clinical_tiger.png", caption="Clinical Tiger Assistant", width=250)

# --------------------------------------------------------------------------
# Main Title and Instructions
# --------------------------------------------------------------------------
st.title("🐅 Clinical Tiger Chat")
st.markdown("""
**How to Use This Chat:**

- Enter your clinical question below.
- Click **Generate Answer** for a fast response.
- Click **Generate + Verify Answer** to double-check via reasoning.
- Click **Retrieve Image** to get the associated image location and caption.

**Disclaimer:** This is for general guidance only. Please consult professionals for medical advice.
""")

# --------------------------------------------------------------------------
# Initialize chat history in session state.
# --------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --------------------------------------------------------------------------
# Chat form for textual queries
# --------------------------------------------------------------------------
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area("Enter your question:")
    col1, col2, col3 = st.columns(3)
    with col1:
        baseline_submit = st.form_submit_button("Generate Answer")
    with col2:
        verify_submit = st.form_submit_button("Generate + Verify Answer")
    with col3:
        image_submit = st.form_submit_button("Retrieve Image")

# --------------------------------------------------------------------------
# Process text-based queries: Generate Answer / Generate + Verify Answer.
# --------------------------------------------------------------------------
if (baseline_submit or verify_submit) and user_input.strip():
    st.session_state.chat_history.append({
        "role": "User",
        "text": user_input.strip()
    })
    with st.spinner("Thinking..."):
        try:
            if verify_submit:
                # Use the chain-of-verification workflow.
                verified_agent = VerifiedAgent()
                answer, references, eval_ratio = verified_agent.run(user_input.strip())
                answer += f" (Correctness: {eval_ratio})"
            else:
                # Use the basic agent workflow.
                basic_agent = Agent()
                answer, references = basic_agent.run(user_input.strip())
            st.session_state.chat_history.append({
                "role": "Clinical Tiger",
                "text": answer,
                "references": references
            })
        except Exception as e:
            st.session_state.chat_history.append({
                "role": "Clinical Tiger",
                "text": f"An error occurred: {e}",
                "references": ""
            })

# --------------------------------------------------------------------------
# Process image-based queries: Retrieve Image.
# --------------------------------------------------------------------------
if image_submit and user_input.strip():
    with st.spinner("Retrieving image..."):
        try:
            image_agent = ImageAgent()
            image_path, caption = image_agent.run(user_input.strip())
            st.session_state.chat_history.append({
                "role": "Clinical Tiger",
                "text": caption,
                "image_path": image_path
            })
        except Exception as e:
            st.session_state.chat_history.append({
                "role": "Clinical Tiger",
                "text": f"An error occurred retrieving the image: {e}"
            })

# --------------------------------------------------------------------------
# Display conversation history
# --------------------------------------------------------------------------
st.markdown("## Conversation")
for msg in st.session_state.chat_history:
    if msg["role"] == "User":
        st.markdown(f"**User:** {msg['text']}")
    elif msg["role"] == "Clinical Tiger":
        st.markdown(f"**🐅:** {msg['text']}")
        if msg.get("references"):
            with st.expander("Show References"):
                refs = msg["references"].split("\n")
                st.markdown("\n".join(f"- {ref}" for ref in refs if ref.strip()))
        if msg.get("image_path"):
            st.image(msg["image_path"], caption=msg.get("caption", ""), use_container_width=True)
