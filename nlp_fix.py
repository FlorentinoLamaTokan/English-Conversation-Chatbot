import re
import gradio as gr
from langchain_ollama import ChatOllama
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import time
import sys
import random
import json
import base64
import io
from IPython.display import Audio, display, HTML
from gtts import gTTS
import tempfile
import os
import urllib.parse
import threading
import speech_recognition as sr


from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3")  # atau "mistral", "codellama", dll

# Initialize LLM
try:
    llm
except NameError:
    sys.stderr.write("LLM not found, initializing ChatOllama model...\n")
    llm = ChatOllama(model="deepseek-r1:8b", temperature=0.7)

# RAG Setup (keeping your existing setup)
search = DuckDuckGoSearchRun()

def search_and_process(query: str):
    """Your existing search function - keeping as is"""
    sys.stderr.write(f"Attempting web search for: {query}\n")
    raw_search_results = ""
    urls = []
    try:
        raw_search_results = search.run(query)
        sys.stderr.write(f"Raw search results snippet: {raw_search_results[:500]}...\n")

        found_urls = []
        url_regex = re.compile(r'http[s]?://(?:[a-zA-Z0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        found_urls.extend(url_regex.findall(raw_search_results))

        line_pattern = re.compile(r'.*\((http[s]?://(?:[a-zA-Z0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)\)$')
        for line in raw_search_results.split('\n'):
            match = line_pattern.search(line)
            if match:
                found_urls.append(match.group(1))

        urls = list(dict.fromkeys(found_urls))[:3]

        if urls:
            sys.stderr.write(f"Found URLs: {urls}\n")
            try:
                loader = WebBaseLoader(urls)
                docs = loader.load()
                sys.stderr.write(f"Loaded {len(docs)} documents from URLs.\n")
                return docs
            except Exception as e:
                sys.stderr.write(f"Error loading documents from URLs: {e}\n")
                return raw_search_results
        else:
            sys.stderr.write("No URLs found in search results. Returning raw results text.\n")
            return raw_search_results

    except Exception as e:
        sys.stderr.write(f"Error during search: {e}\n")
        return ""

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

try:
    embeddings = GPT4AllEmbeddings()
    sys.stderr.write("GPT4AllEmbeddings initialized.\n")
except Exception as e:
    sys.stderr.write(f"Error initializing GPT4AllEmbeddings: {e}\n")
    embeddings = None

# Enhanced system prompts
base_system_prompt_content = (
    "You are W11, a friendly English tutor for beginners. "
    "Focus on daily topics like travel, school, daily routines, food, family, hobbies. "
    "Provide gentle grammar & pronunciation feedback. "
    "Keep responses conversational and encouraging. "
    "If asked about vocabulary, provide simple definitions with example sentences."
)

# NEW: Vocabulary generation prompt (needed for quiz)
vocab_system_prompt_content = (
    "You are a vocabulary generator for English beginners. "
    "Generate vocabulary lists based on the specific topic provided by the user. "
    "Create words that are practical and commonly used in real-life situations related to that topic. "
    "For each word, provide: word, simple definition, example sentence, difficulty level (1-3). "
    "Always respond in valid JSON format: {\"words\": [{\"word\": \"example\", \"definition\": \"simple meaning\", \"example\": \"Example sentence using the word\", \"level\": 1}]}"
)

# Prompts
base_prompt = ChatPromptTemplate.from_messages([
    ("system", base_system_prompt_content),
    ("human", "{question}"),
])

vocab_prompt = ChatPromptTemplate.from_messages([
    ("system", vocab_system_prompt_content),
    ("human", "Generate 10 vocabulary words for the topic: {topic}. Include beginner-friendly words related to daily life."),
])

# Your existing RAG prompt
rag_system_prompt_content = (
    "You are W11_F-AMG, a friendly English tutor for beginners. "
    "Provide gentle grammar & pronunciation feedback. "
    "Answer the user's question accurately using the following context. "
    "If the context doesn't contain enough information, politely state that you need more information. "
    "\n\nContext: {context}"
)
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", rag_system_prompt_content),
    ("human", "{question}"),
])

# Analysis prompt (keeping your existing one)
analysis_system_prompt_content = (
    "You are an AI assistant focused on analyzing user input. "
    "Analyze for: "
    "1. **Grammar:** Identify errors and suggest corrections. "
    "2. **Intent:** Determine the main purpose of the message. "
    "3. **Context:** Explain the main topic of the message. "
    "Format output:\n"
    "**Grammar Analysis:** [feedback]\n"
    "**Detected Intent:** [intent]\n"
    "**Contextual Relevance:** [context]"
)
analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", analysis_system_prompt_content),
    ("human", "Analyze: {user_message}"),
])

# NEW: Vocabulary and Quiz Functions (keeping only necessary ones)
def generate_vocabulary(topic):
    """Generate vocabulary for a given topic using the LLM"""
    sys.stderr.write(f"Generating vocabulary for topic: {topic}\n")
    try:
        vocab_messages = [
            SystemMessage(content=vocab_system_prompt_content),
            HumanMessage(content=f"Generate 10 vocabulary words for the topic: {topic}. Include beginner-friendly words related to daily life. Return in valid JSON format.")
        ]

        response = llm.invoke(vocab_messages)
        vocab_text = clean_response(response.content)
        sys.stderr.write(f"Vocabulary response: {vocab_text[:300]}...\n")

        try:
            json_match = re.search(r'\{.*\}', vocab_text, re.DOTALL)
            if json_match:
                vocab_data = json.loads(json_match.group())
            else:
                vocab_data = json.loads(vocab_text)

            words = vocab_data.get('words', [])
            sys.stderr.write(f"Successfully parsed JSON vocabulary: {len(words)} words\n")
            return words[:10]

        except json.JSONDecodeError:
            return parse_vocabulary_from_text(vocab_text, topic)

    except Exception as e:
        sys.stderr.write(f"Error generating vocabulary: {e}\n")
        return create_emergency_vocab(topic)

def parse_vocabulary_from_text(text, topic):
    """Parse vocabulary dari text response DeepSeek"""
    words = []
    lines = text.split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        word_patterns = [
            r'^\d+\.\s*([a-zA-Z]+)',
            r'^-\s*([a-zA-Z]+)',
            r'^\*\s*([a-zA-Z]+)',
            r'^([a-zA-Z]+)\s*[-:]'
        ]

        for pattern in word_patterns:
            match = re.match(pattern, line)
            if match:
                word = match.group(1).lower()
                words.append({
                    'word': word,
                    'definition': f'A word related to {topic}',
                    'example': f'I use {word} when talking about {topic}',
                    'level': 1
                })
                break

    return words[:10]

def create_emergency_vocab(topic):
    """Emergency fallback vocabulary"""
    basic_words = {
        'travel': ['plane', 'hotel', 'ticket', 'passport', 'luggage', 'map', 'tourist', 'trip', 'vacation', 'journey'],
        'food': ['breakfast', 'lunch', 'dinner', 'hungry', 'delicious', 'cook', 'restaurant', 'menu', 'taste', 'drink'],
        'school': ['student', 'teacher', 'book', 'study', 'learn', 'class', 'homework', 'exam', 'library', 'grade'],
        'work': ['job', 'office', 'meeting', 'computer', 'boss', 'colleague', 'salary', 'project', 'deadline', 'career'],
        'family': ['mother', 'father', 'sister', 'brother', 'children', 'parents', 'relative', 'love', 'home', 'together'],
        'daily life': ['wake', 'eat', 'work', 'sleep', 'home', 'time', 'day', 'night', 'morning', 'evening']
    }

    topic_words = basic_words.get(topic.lower(), basic_words['daily life'])
    return [{'word': word, 'definition': f'Related to {topic}', 'example': f'I use {word} daily', 'level': 1} for word in topic_words]


def create_vocabulary_quiz(words, topic="daily life"):
    """Create a quiz from vocabulary words"""
    if not words:
        sys.stderr.write("No vocabulary available, generating new vocabulary...\n")
        words = generate_vocabulary(topic)
        if not words:
            return "Sorry, I couldn't generate vocabulary for the quiz. Please try 'quiz me about [topic]' first."

    selected_words = random.sample(words, min(5, len(words)))
    quiz = f"🎯 **Vocabulary Quiz - {topic.title()}**\n\n"

    all_definitions = [word.get('definition', 'definition') for word in words]

    for i, word_data in enumerate(selected_words, 1):
        word = word_data.get('word', 'word')
        correct_definition = word_data.get('definition', 'definition')

        wrong_answers = [definition for definition in all_definitions if definition != correct_definition]
        if len(wrong_answers) < 3:
            wrong_answers.extend(['A type of color', 'A number', 'A musical instrument'])

        random.shuffle(wrong_answers)
        wrong_answers = wrong_answers[:3]

        all_options = [correct_definition] + wrong_answers
        random.shuffle(all_options)

        correct_position = all_options.index(correct_definition)
        correct_letter = ['a', 'b', 'c', 'd'][correct_position]

        quiz += f"{i}. What does '{word}' mean?\n"
        for j, option in enumerate(all_options):
            letter = ['a', 'b', 'c', 'd'][j]
            quiz += f"   {letter}) {option}\n"
        quiz += "\n"

    quiz += "Type your answers (like: 1a, 2c, 3b...) and I'll check them! 🤔"
    return quiz


def clean_response(text):
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()

def is_likely_update_query(query: str):
    update_patterns = re.compile(
        r"\b(terbaru|terkini|update|siapa presiden sekarang|hasil .*?\d{4}|juara .*?\d{4}|now|current|saat ini|sekarang|who is the|what is the current)\b",
        re.IGNORECASE
    )
    return bool(update_patterns.search(query))

def transcribe_and_respond(audio, state):
    recognizer = sr.Recognizer()

    # ✔️ Periksa jika audio dikirim sebagai dict
    if isinstance(audio, dict) and "name" in audio:
        audio = audio["name"]

    try:
        with sr.AudioFile(audio) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        text = "Sorry, I couldn't understand your speech."
    except sr.RequestError:
        text = "Speech recognition service is unavailable."
    except Exception as e:
        text = f"Error processing audio: {e}"

    return respond(text, state)



# Modified respond function
def respond(user_input, state):
    if state is None:
        messages = []
        ui_history = []
        current_vocab = []
        current_topic = "daily life"
    else:
        # Handle different state formats for backward compatibility
        if len(state) == 2:
            messages, ui_history = state
            current_vocab = []
            current_topic = "daily life"
        else:
            messages, ui_history, current_vocab, current_topic = state

    sys.stderr.write(f"User input: '{user_input}'\n")

    # Check for special commands
    user_lower = user_input.lower().strip()

    # Quiz command
    if user_lower.startswith(('quiz me about', 'quiz about', 'test me about')):
        if 'about' in user_input.lower():
            topic = user_input.lower().split('about')[-1].strip()
        else:
            topic = 'daily life'

        current_vocab = generate_vocabulary(topic)
        current_topic = topic

        quiz = create_vocabulary_quiz(current_vocab, current_topic)

        messages.append(HumanMessage(content=user_input))
        messages.append(AIMessage(content=quiz))
        ui_history.append({"role": "user", "content": user_input})
        ui_history.append({"role": "assistant", "content": quiz})

        chat_display = [(msg["role"].capitalize(), msg["content"]) for msg in ui_history]
        return chat_display, (messages, ui_history, current_vocab, current_topic)


    # Regular conversation (your existing logic)
    requires_update = is_likely_update_query(user_input)
    sys.stderr.write(f"User input likely requires update: {requires_update}\n")

    ai_resp = None
    bot_text = ""
    needs_rag = False
    used_rag = False

    if requires_update:
        sys.stderr.write("Question likely requires update, proceeding directly to RAG...\n")
        needs_rag = True
    else:
        sys.stderr.write("Question does not likely require update. Attempting to answer with LLM's internal knowledge first...\n")
        messages_for_base_llm = [SystemMessage(content=base_system_prompt_content)]
        messages_for_base_llm.extend(messages)
        messages_for_base_llm.append(HumanMessage(content=user_input))

        try:
            initial_llm_response = llm.invoke(messages_for_base_llm)
            initial_raw_text = initial_llm_response.content
            initial_bot_text = clean_response(initial_raw_text)
            sys.stderr.write(f"Initial LLM response (first 200 chars): {initial_bot_text[:200]}...\n")

            uncertainty_patterns = re.compile(r"(i don't know|i am not sure|i lack enough information|cannot answer with certainty|no specific information)", re.IGNORECASE)
            needs_rag = bool(uncertainty_patterns.search(initial_bot_text))
            sys.stderr.write(f"Initial response indicates need for RAG: {needs_rag}\n")

            if not needs_rag:
                sys.stderr.write("LLM appears to have an answer. Using initial response.\n")
                bot_text = initial_bot_text
                ai_resp = initial_llm_response
                used_rag = False
            else:
                sys.stderr.write("Initial LLM response indicates need for more info. Proceeding to RAG.\n")
                needs_rag = True

        except Exception as e:
            sys.stderr.write(f"Error during initial LLM call: {e}\n")
            needs_rag = True
            initial_llm_response = None

    # RAG workflow (your existing logic)
    if needs_rag:
        sys.stderr.write("Starting RAG workflow...\n")
        retrieved_data = search_and_process(user_input)

        context_text = ""
        if isinstance(retrieved_data, list) and retrieved_data:
            try:
                splits = text_splitter.split_documents(retrieved_data)
                if embeddings is not None:
                    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
                    retrieved_chunks = retriever.invoke(user_input)
                    context_text = "\n\n".join([chunk.page_content for chunk in retrieved_chunks])
                else:
                    context_text = "Error: Embeddings not available"
            except Exception as e:
                sys.stderr.write(f"Error during document processing: {e}\n")
                context_text = "Error processing retrieved documents."
        elif isinstance(retrieved_data, str) and retrieved_data:
            context_text = retrieved_data

        if context_text and embeddings is not None:
            messages_for_rag_llm = [SystemMessage(content=rag_system_prompt_content.format(context=context_text))]
            messages_for_rag_llm.extend(messages)
            messages_for_rag_llm.append(HumanMessage(content=user_input))

            try:
                ai_resp = llm.invoke(messages_for_rag_llm)
                raw = getattr(ai_resp, "content", str(ai_resp))
                bot_text = clean_response(raw)
                used_rag = True
            except Exception as e:
                sys.stderr.write(f"Error invoking LLM with RAG context: {e}\n")
                bot_text = "Sorry, there was an error processing your request with additional information."
                ai_resp = AIMessage(content=bot_text)

    # User input analysis (your existing logic)
    sys.stderr.write(f"Analyzing user input: '{user_input}'\n")
    analysis_messages = analysis_prompt.format_messages(user_message=user_input)
    analysis_results = ""
    try:
        analysis_response = llm.invoke(analysis_messages)
        analysis_results = clean_response(analysis_response.content)
    except Exception as e:
        sys.stderr.write(f"Error during analysis: {e}\n")
        analysis_results = "Analysis temporarily unavailable."

    # Update history
    messages.append(HumanMessage(content=user_input))
    ui_history.append({"role": "user", "content": user_input})

    if ai_resp is not None:
        messages.append(ai_resp)
        ui_history.append({"role": "assistant", "content": bot_text})
    else:
        fallback_msg = "Sorry, there was an issue generating a response."
        messages.append(AIMessage(content=fallback_msg))
        ui_history.append({"role": "assistant", "content": fallback_msg})

    # Add analysis
   # Jalankan analisis di thread terpisah
    # Langsung tampilkan hasil analisis di UI
    analysis_messages = analysis_prompt.format_messages(user_message=user_input)
    analysis_results = ""
    try:
        analysis_response = llm.invoke(analysis_messages)
        analysis_results = clean_response(analysis_response.content)

        analysis_message_content = f"📊 **Analysis of your message:**\n\n{analysis_results}"

        messages.append(AIMessage(content=analysis_message_content))
        ui_history.append({"role": "assistant", "content": analysis_message_content})

    except Exception as e:
        sys.stderr.write(f"Error during analysis: {e}\n")



    # Convert ui_history to format accepted by gr.Chatbot
    chat_display = [
        (msg["role"].capitalize(), msg["content"]) if isinstance(msg, dict) else msg
        for msg in ui_history
    ]

    return chat_display, (messages, ui_history, current_vocab, current_topic)


def run_analysis_async(user_input, chatbot, state):
    try:
        analysis_messages = analysis_prompt.format_messages(user_message=user_input)
        analysis_response = llm.invoke(analysis_messages)
        analysis_results = clean_response(analysis_response.content)

        analysis_message_content = f"📊 **Analysis of your message:**\n{analysis_results}"

        # Tambahkan ke history (bukan chatbot langsung)
        state[1].append({"role": "assistant", "content": analysis_message_content})

        # OPTIONAL: Kirim analisis ke UI (kalau kamu pakai polling / gr.update nanti)
        # chatbot.append(("Assistant", analysis_message_content))

    except Exception as e:
        sys.stderr.write(f"[!] Analysis error: {e}\n")




# Modified UI
with gr.Blocks(title="W11 - English Tutor with Quiz & RAG") as demo:
    gr.Markdown(""" # 🎓 W11 - Your Personal English Tutor with Quiz & RAG

    **Features:**
    - 💬 **Chat**: Normal conversation with grammar feedback and RAG for factual questions.
    - 🎯 **Quiz**: Type "quiz me about [topic]" (e.g., "quiz me about travel") to get a vocabulary quiz.

    **Topics**: travel, school, daily routines, food, family, hobbies, shopping, work, etc.
    """)

    chatbot = gr.Chatbot(type="messages", height=500)
    state = gr.State(None)

    with gr.Row():
        user_input = gr.Textbox(
            placeholder="Try: 'quiz me about travel' or just chat with me!",
            container=False,
            scale=4
        )
        send_btn = gr.Button("Send", scale=1)
    
    with gr.Row():
        mic_input = gr.Audio(source="microphone", type="filepath", label="🎤 Speak here", scale=4)
        mic_send = gr.Button("Send Voice", scale=1)

    # Quick action buttons (keeping only quiz)
    with gr.Row():
        quiz_btn = gr.Button("🎯 Quiz Me", size="sm")

    


    # Event handlers
    user_input.submit(
        fn=respond,
        inputs=[user_input, state],
        outputs=[chatbot, state]
    )

    send_btn.click(
        fn=respond,
        inputs=[user_input, state],
        outputs=[chatbot, state]
    )

    mic_send.click(
        fn=transcribe_and_respond,
        inputs=[mic_input, state],
        outputs=[chatbot, state]
    )

    mic_send.click(lambda: None, None, mic_input)

    # Quick button handler for quiz
    def show_quiz_instruction(state_value):
        if state_value is None:
            messages = []
            ui_history = []
            current_vocab = []
            current_topic = "daily life"
        else:
            if len(state_value) == 2:
                messages, ui_history = state_value
                current_vocab = []
                current_topic = "daily life"
            else:
                messages, ui_history, current_vocab, current_topic = state_value

        instruction = "🎯 **Quiz Instructions**\n\nPlease type: 'quiz me about [topic]'\n\nExample topics:\n• travel\n• food\n• daily life\n• school\n• family\n• work\n• shopping"

        # Add instruction to history
        ui_history.append({"role": "assistant", "content": instruction})
        messages.append(AIMessage(content=instruction))

        return ui_history, (messages, ui_history, current_vocab, current_topic)

    quiz_btn.click(
        fn=show_quiz_instruction,
        inputs=[state],
        outputs=[chatbot, state]
    )

    # Clear input after sending
    user_input.submit(lambda: "", None, user_input)
    send_btn.click(lambda: "", None, user_input)

demo.launch(debug=True, inline=True)