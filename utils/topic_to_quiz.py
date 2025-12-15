import streamlit as st
import pandas as pd
from openai import OpenAI

DEEPSEEKR1_API_KEY = st.secrets["DEEPSEEKR1_API_KEY"]
base_url = "https://openrouter.ai/api/v1"

def load_quiz_from_topic(topic: str, n: int) -> pd.DataFrame:
    client = OpenAI(api_key=DEEPSEEKR1_API_KEY, base_url=base_url)
    
    chat = client.chat.completions.create(
        model="deepseek/deepseek-r1:free",
        messages=[{
            "role": "user",
            "content": f"Generate {n} quiz questions on the topic '{topic}'. "
                      f"Each question should be in the format: "
                      f'"question","option1","option2","option3","option4","correct_answer"'
        }]
    )
    # print(chat.choices[0].message.content) # Debugging: Print raw response

    quiz_data = []
    response_text = chat.choices[0].message.content.strip()

    for line in response_text.split("\n"):
        columns = [col.strip().strip('"') for col in line.split('","')]
        if len(columns) == 6:
            # Replace 'none' with 'None of these' in options and correct answer
            for i in range(1, 5):  # indices 1-4 are options
                if columns[i].lower() == 'none':
                    columns[i] = 'None of these'
            if columns[5].lower() == 'none':  # index 5 is correct_answer
                columns[5] = 'None of these'
            quiz_data.append(columns)

    df=pd.DataFrame(quiz_data, columns=["question", "option1", "option2", "option3", "option4", "correct_answer"])
 
    df = df[df['question'] != 'question']  # Remove header row
    return df

if __name__ == "__main__":
    topic = "genetic algorithms in ml"
    n = 5
    quiz_df = load_quiz_from_topic(topic, n)
    print(quiz_df)
