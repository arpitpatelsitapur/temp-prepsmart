import streamlit as st
from utils.topic_to_quiz import load_quiz_from_topic

def show_topic_page():
    st.title("📚 Generate Quiz by Topic")

    # Information Section
    with st.expander("ℹ️ How to Use, Technology & Cautions", expanded=False):
        st.markdown("""
        ### How to Use:
        1. Enter a topic (e.g., "Machine Learning", "Databases").
        2. Specify the number of questions you want.
        3. Click the **"Get Quizzes"** button to generate quiz questions.
        4. If successful, a **"Start Quiz"** button will appear.

        ### Technologies Used:
        - **Streamlit** for the UI.
        - **Deepseek R1** for questions generation.
        - **Pandas** for handling quiz data.

        ### Cautions:
        - The quiz quality depends on the AI-generated output.
        - Ensure an active internet connection for API calls.
        - If generation fails, try rephrasing the topic.
        """)

    # User Input Section
    topic = st.text_input("Enter topic for quiz generation:")
    no_of_questions = st.number_input("Number of questions:", min_value=1, value=3)

    if st.button("Get Quizzes"):
        with st.spinner("Generating quiz questions..."):
            st.session_state.generating = True
            try:
                df = load_quiz_from_topic(topic, no_of_questions)
                if df is not None and not df.empty:
                    st.session_state.quiz_df = df
                    st.success("Quiz generated successfully!")
                    st.button("Start Quiz", type="primary", 
                             on_click=lambda: st.session_state.update({"quiz_state": "in_progress"}))
                else:
                    st.error("Failed to generate quiz. Please try again.")
            except Exception as e:
                st.error(f"Error generating quiz: {str(e)}")
            finally:
                st.session_state.generating = False