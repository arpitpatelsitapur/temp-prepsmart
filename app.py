import streamlit as st
import time
import threading
from deepgram import (
    DeepgramClient,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)
from groq import Groq
from dotenv import load_dotenv
import requests
# from playsound import playsound
import logging 
from utils.topic_page import *
from utils.topic_to_quiz import *

# ✅ Load environment variables
DEEPGRAM_API_KEY = st.secrets["DEEPGRAM_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# ✅ Streamlit Page Configuration
st.set_page_config(
    page_title="Prep-Smart",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto"
)


##########################################################################################################
# part 1 : Topic to quiz
# LLM : Deepseek
##########################################################################################################

# Session state initialization
if 'page' not in st.session_state:
    st.session_state.update({
        'page': 'home',
        'quiz_state': 'not_started',
        'current_question': 0,
        'user_answers': {},
        'shuffled_options': {},
        'question_feedback': [],
        'quiz_duration': 0,
        'final_score': 0,
        'quiz_source': None,
        'quiz_df': None,
        'generating': False,
        'start_time': 0
    })

# 🔓 Audio unlock state
if "audio_unlocked" not in st.session_state:
    st.session_state.audio_unlocked = False

def display_question():
    idx = st.session_state.current_question
    row = st.session_state.quiz_df.iloc[idx]

    # Use the original order of options
    options = [row['option1'], row['option2'], row['option3'], row['option4']]

    with st.expander(f"Question {idx + 1}", expanded=True):  # Display 1-based
        st.markdown(f"#### {row['question']}")
        
        # Save with idx + 1 to align with display numbering
        st.session_state.user_answers[idx] = st.radio(
            "Select an answer:", options,
            index=options.index(st.session_state.user_answers.get(idx, options[0])),
            key=f"q{idx}"  
        )

##########################################################################################################
# part 2 : Voice to voice interaction
# STT : Deepgram
# LLM : GROQ
# TTS : Deepgram
##########################################################################################################

# ✅ Gradual Display Inside the Div Block
def gradual_display_inside_div(text, role, delay=0.15):
    """Display text word by word inside the div block."""
    role_icon = "👤" if role == "user" else "🤖"
    color = "#f1f1f1" if role == "user" else "#d1e7dd"

    placeholder = st.empty()
    current_text = ""
    words = text.split()

    for word in words:
        if current_text:
            current_text += " " + word
        else:
            current_text = word

        # ✅ Display the entire div block with gradually updating text inside
        placeholder.markdown(
            f"""
            <div style='background-color: {color}; border-radius: 10px; padding: 15px; margin-bottom: 10px;'>
                <strong>{role_icon}:</strong> {current_text}
            </div>
            """,
            unsafe_allow_html=True
        )
        time.sleep(delay)

# ✅ Function for Deepgram TTS (run in a separate thread)
def text_to_speech(text: str, filename: str = "response_audio.wav"):
    """Convert text to speech using Deepgram and stream via browser."""
    if not st.session_state.audio_unlocked:
        return  # 🔐 Respect browser rule

    url = "https://api.deepgram.com/v1/speak"
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"text": text}

    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            st.error("TTS failed")
            return

        with open(filename, "wb") as f:
            f.write(response.content)

        # ✅ Browser-based audio playback
        with open(filename, "rb") as audio_file:
            st.audio(audio_file.read(), format="audio/wav", autoplay=True)

    except Exception as e:
        st.error(f"TTS Error: {e}")

    except Exception as e:
        st.error(f"Error handling audio: {str(e)}")


# ✅ Parallel Execution Function
def parallel_display_and_speak(text, role, delay=0.15):
    """Run display and speech in parallel using multithreading."""
    
    # ✅ Only run TTS for LLM responses
    if role == "llm":
        tts_thread = threading.Thread(target=text_to_speech, args=(text,))
        tts_thread.start()
    
    # Gradually display the text
    gradual_display_inside_div(text, role, delay)

    # ✅ Wait for TTS to finish only if it's an LLM response
    if role == "llm":
        tts_thread.join()


# ✅ Function for Deepgram STT
final_utterance = ""
def speech_to_text():
    is_finals = []  # Define the list inside the function

    try:
        # Initialize Deepgram client
        deepgram = DeepgramClient(api_key=DEEPGRAM_API_KEY)

        # Create a websocket connection
        dg_connection = deepgram.listen.websocket.v("1")

        # Event handlers
        def on_open(self, open, **kwargs):
            pass

        def on_message(self, result, **kwargs):
            nonlocal is_finals  # ✅ Use 'nonlocal' to access the enclosing scope
            sentence = result.channel.alternatives[0].transcript
            
            if len(sentence) == 0:
                return

            if result.is_final:
                print(f"✅ Final: {sentence}")
                
                # ✅ Append each final sentence to the list
                is_finals.append(sentence)

                if result.speech_final:
                    final_utterance = " ".join(is_finals)  # ✅ Combine all final sentences
                    print(f"🗣️ Speech Final: {final_utterance}")

        def on_metadata(self, metadata, **kwargs):
            pass

        def on_speech_started(self, speech_started, **kwargs):
            pass

        def on_utterance_end(self, utterance_end, **kwargs):
            pass

        def on_close(self, close, **kwargs):
            pass

        def on_error(self, error, **kwargs):
            pass

        def on_unhandled(self, unhandled, **kwargs):
            pass

        # Register event handlers
        dg_connection.on(LiveTranscriptionEvents.Open, on_open)
        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        dg_connection.on(LiveTranscriptionEvents.Metadata, on_metadata)
        dg_connection.on(LiveTranscriptionEvents.SpeechStarted, on_speech_started)
        dg_connection.on(LiveTranscriptionEvents.UtteranceEnd, on_utterance_end)
        dg_connection.on(LiveTranscriptionEvents.Close, on_close)
        dg_connection.on(LiveTranscriptionEvents.Error, on_error)
        dg_connection.on(LiveTranscriptionEvents.Unhandled, on_unhandled)

        # Define live options
        options: LiveOptions = LiveOptions(
            model="nova-3",
            language="en-US",
            smart_format=True,
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            interim_results=True,
            utterance_end_ms="3000",
            vad_events=True,
            endpointing=1000,
        )

        # Addon settings
        addons = {"no_delay": "true"}

        if dg_connection.start(options, addons=addons) is False:
            print("❌ Failed to connect to Deepgram")
            return

        # print("\n🎤 Press Enter to stop recording...\n")
        
        # Open a microphone stream
        microphone = Microphone(dg_connection.send)
        microphone.start()
        
        # Simulate recording for a while (e.g., 10 seconds)
        time.sleep(100)  # Wait for 15 seconds to simulate recording

        # Stop the microphone and connection
        microphone.finish()
        dg_connection.finish()

        # ✅ Return the combined transcript
        final_transcript = " ".join(is_finals)
        
        # ✅ Clear the list after returning the result to avoid leftover data
        is_finals = []

        return final_transcript

    except Exception as e:
        print(f"❌ Error: {e}")
        return ""


# ✅ Generate LLM response using GROQ
conversation_history = [
    {
        "role": "system",
        "content": """
        You are a professional technical recruiter conducting interviews. 
        Ask only short and concise technical questions, similar to real-life interviews. 
        Each question should be direct, clear, and focused on a single concept or problem.
        Keep the questions to 1-2 sentences maximum.
        Before asking the next question, give feedback on the previous answer.
        """
    }
]

def generate_response(user_input, model="llama-3.3-70b-versatile", temperature=0.7):
    """Generates response using Groq LLM with conversation context."""
    try:
        # Append user message to the conversation history
        conversation_history.append({"role": "user", "content": user_input})

        client = Groq(api_key=GROQ_API_KEY)

        completion = client.chat.completions.create(
            model=model,
            messages=conversation_history,
            temperature=temperature,
        )

        # Extract the assistant's message and add it to the history
        assistant_response = completion.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": assistant_response})

        return assistant_response

    except Exception as e:
        logging.error(f"❌ LLM Error: {e}")
        return "Error generating LLM response."

##########################################################################################################
# APP UI code
##########################################################################################################

# Sidebar configuration
with st.sidebar:
    st.image("utils/app_logo.png")
    st.header("App Configuration")
    source = st.sidebar.radio("Select Page", ["Home", "Topic to Quiz", "Voice-to-Voice"])
       
if source == "Home":
    st.title("Prep-Smart: Smart Interview Preparation with AI")
    st.write("""
    All-in-one platform for **interview preparation** with interactive quizzes and **voice-to-voice AI conversation**. 
    Whether you're brushing up on technical concepts or practicing real-time interviews, this app offers a seamless and engaging experience.
    """)

    # ✅ What You Can Do Section
    st.subheader("💡 What You Can Do:")
    
    st.markdown("""
    **1️⃣ 📚 Topic to Quiz**
    - Instantly generate **customized quizzes** on any technical topic.
    - Powered by **Deepseek R1**, a state-of-the-art language model for accurate and contextually relevant questions.
    - Test your knowledge with **multiple-choice questions (MCQs)**.
    - Receive **detailed feedback and scores** to assess your strengths and weaknesses.
    
    **2️⃣ 🎙️ Voice-to-Voice Interview**
    - Experience **real-time, interactive interviews** with an AI-powered recruiter.
    - Speak your answers naturally, just like in an actual interview.
    - The AI will respond with **technical questions, feedback, and guidance**.
    - Get **instant feedback** on your answers, helping you improve your communication skills.
    """)

    # ✅ Technologies Section
    st.subheader("⚙️ Technologies Powering This App:")
    st.markdown("""
    - **Streamlit**: For a seamless and interactive user interface.
    - **Deepseek R1**: 
        - **State-of-the-art language model** for generating accurate, topic-specific quiz questions.
    - **Deepgram**: 
        - **Speech-to-Text (STT)** for real-time transcription.
        - **Text-to-Speech (TTS)** for lifelike AI responses.
    - **Groq LLM**: For generating realistic interview questions and providing feedback.
    - **Multithreading**: Ensures parallel processing for a smooth and responsive experience.
    """)

    # ✅ Why Use This App Section
    st.subheader("🚀 Why Use Prep-Smart?")
    st.markdown("""
    - **Real-time practice**: Simulate interview conditions with an interactive AI.
    - **Adaptive learning**: Generate topic-based quizzes to strengthen weak areas.
    - **Efficient preparation**: Improve both **technical knowledge** and **communication skills**.
    - **User-friendly**: Easy-to-use interface with clear feedback and performance metrics.
    """)

    # ✅ Get Started Section
    st.subheader("✅ Get Started!")
    st.markdown("""
    Use the **sidebar** to navigate between:
    - 🏠 **Home Page**: Learn about the app.
    - 📚 **Topic to Quiz**: Generate and take topic-based quizzes.
    - 🎙️ **Voice-to-Voice**: Practice live AI-powered interviews.

    🔥 **Boost your interview confidence with Prep-Smart!** 🚀
    """)

# ✅ Placeholder for other pages
elif source == "Topic to Quiz":
    st.title("📚 Topic to Quiz")
    st.write("🔹 This is where you can generate quizzes based on a topic.")
    st.write("🧠 Powered by **Deepseek R1** for high-quality, accurate questions.")
    if st.session_state.quiz_state == 'not_started':
        show_topic_page()
        st.session_state.shuffled_options = {}  # ✅ Reset options when new quiz is generated

    elif st.session_state.quiz_state == 'in_progress':
        if st.session_state.start_time == 0:
            st.session_state.start_time = time.time()

        st.title("📝 Quiz In Progress")
        # st.dataframe(st.session_state.quiz_df)   # for debugging
        current_q = st.session_state.current_question
        total_q = len(st.session_state.quiz_df)
        st.progress((current_q + 1) / total_q, text=f"Question {current_q + 1} of {total_q}")
    
        display_question() 
        col1, col2 = st.columns([8, 1.2])
        with col1:
            if current_q > 0:
                st.button("Previous", on_click=lambda: st.session_state.update({"current_question": current_q - 1}))
        with col2:    
            if current_q < total_q - 1:
                st.button("Next", on_click=lambda: st.session_state.update({"current_question": current_q + 1}))
            else:
                if st.button("Submit", type="primary"):
                    st.session_state.quiz_duration = time.time() - st.session_state.start_time
                    st.session_state.quiz_state = 'completed'
                    st.rerun()

    elif st.session_state.quiz_state == 'completed':
        st.balloons()
        st.title("📊 Quiz Results")

        # Calculate final score and feedback
        score = 0
        feedback = []

        for idx, row in st.session_state.quiz_df.iterrows():
            correct_option = str(row['correct_answer']).strip().lower()
            user_answer = str(st.session_state.user_answers.get(idx, "")).strip().lower()
            is_correct = user_answer == correct_option

            score += int(is_correct)
            feedback.append({
                "Question": row["question"],
                "Your Answer": st.session_state.user_answers.get(idx, ""),
                "Correct Answer": row["correct_answer"],
                "Feedback": "Correct" if is_correct else "Incorrect"
            })

        st.session_state.final_score = score
        st.session_state.question_feedback = feedback

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Final Score", f"{st.session_state.final_score}/{len(st.session_state.quiz_df)}")
        with col2:
            mins = int(st.session_state.quiz_duration // 60)
            secs = int(st.session_state.quiz_duration % 60)
            st.metric("Time Taken", f"{mins}m {secs}s")

        st.subheader("Detailed Feedback")
        feedback_df = pd.DataFrame(st.session_state.question_feedback)

         # Function to format feedback table
        def color_feedback(val):
            if val == "Correct":
                return "background-color: green; color: white"
            elif val == "Incorrect":
                return "background-color: red; color: white"
            return ""

        if not feedback_df.empty:
            styled_feedback = feedback_df.style.applymap(color_feedback, subset=["Feedback"])
            st.dataframe(styled_feedback)
        else:
            st.warning("No feedback available.")  

        if st.button("🔄 Take Another Quiz"):
            # Clear all quiz-related session state
            quiz_state_keys = [
                'page', 'quiz_state', 'quiz_df', 'start_time',
                'user_answers', 'current_question', 'shuffled_options'
            ]
            for key in quiz_state_keys:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()                

elif source == "Voice-to-Voice":
    st.title("🎙️ AI-Powered Interview Bot")
    st.write("Speak to the bot and get instant technical interview responses.")

    # 🔓 One-time unlock
    if not st.session_state.audio_unlocked:
        if st.button("🎤 Start Interview"):
            st.session_state.audio_unlocked = True
            st.success("Audio enabled. Interview starting...")
            st.rerun()
        st.stop()

    # ✅ Interview starts AFTER unlock
    if st.button("New Conversation"):
        welcome_message = (
            "Hello, I am an AI Model designed to help you with interview preparation. "
            "Tell me about yourself."
        )
        parallel_display_and_speak(welcome_message, "llm")

        while True:
            transcript = speech_to_text()

            if not transcript:
                st.error("❌ No speech detected.")
                continue

            parallel_display_and_speak(transcript, "user")

            llm_response = generate_response(transcript)
            parallel_display_and_speak(llm_response, "llm")

            if any(word in transcript.lower() for word in ["exit", "quit", "stop", "bye"]):
                break
                
    with st.expander("ℹ️ How to Use, Technologies & Cautions", expanded=False):
        st.markdown("""
    ### 💡 How to Use:
    1. Click the **"Start Conversation"** button.
    2. Speak clearly into your microphone(By default our app is using 15 seconds window for user speech).
    3. The bot will transcribe your speech and display it word by word.
    4. The LLM will generate a response, which will be displayed and spoken in real-time.
    5. Say **"exit"**, **"quit"**, **"stop"**, or **"bye"** to end the conversation.

    ### ⚙️ Technologies Used:
    - **Streamlit** for the UI.
    - **Deepgram** for real-time speech-to-text (STT) and text-to-speech (TTS).
    - **Groq LLM** for generating interview-style responses.
    - **Multithreading** for parallel TTS and gradual text display.

    ### ⚠️ Cautions:
    - Ensure your microphone is properly configured.
    - For accurate transcription, speak clearly and avoid background noise.
    - A stable internet connection is required for API calls.
    - Keep Attention in default 15 seconds window for speech. 
    - If the app hangs or crashes, try refreshing the page.
    """)

