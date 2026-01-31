# TTS Dataset Manager

A Streamlit-based tool for managing, annotating, and enhancing Text-to-Speech (TTS) datasets.
This app allows you to load datasets, search & filter transcripts, visualize and record audio, translate transcripts, and export enriched datasets for training or analysis.

## Features  
- Dataset Upload  
    - Load .csv or .json files containing transcripts and audio paths.  
    - Automatically checks if audio files exist.  

- Search & Filter  
    - Search transcripts by keyword.  
    - Highlight missing audio files.  

- Audio Tools  
    - Play existing audio linked to transcripts.  
    - Visualize audio waveform using Librosa.  
    - Record and replace audio directly inside the app.  

- Translation (OpenAI)  
    - Translate transcripts into multiple languages (Croatian, English, French, German, Polish).  
    - Translate single entries or the entire dataset.  
    - Download the translated dataset as CSV.  

- Annotation  
    - Add metadata like speaker, emotion, and category.  

- Export  
    - Save the enriched dataset as CSV or JSON.  

## Instalation  
**1. Requirements**  
- Python 3.12+  
- Internet connection (for translation and package installation)  
- OpenAI API key (for translation)  
  
**2. Clone repository**  
```
git clone https://github.com/yourusername/tts-dataset-tool.git  
cd tts-dataset-tool  
```
  
**3. Set up Virtual Environment**  
```
python -m venv .venv 
source .venv/bin/activate   # Linux/Mac 
.venv\Scripts\activate      # Windows 
```
  
**4. Install dependencies**  
```
uv sync
```  
  
requirements.txt should contain: 
```
streamlit 
pandas 
pydub 
librosa 
matplotlib 
openai 
sounddevice 
numpy 
```
## Set up
**1. Configure secrets**  
Create a `.streamlit/secrets.toml` file:  
```
OPENAI_API_KEY = “your_api_key_here”
```  

**2. Make sure your dataset has at least these columns:**  
- `transcript` – text of the utterance
- `audio_path` – path to corresponding audio file

## Running the App 
```
streamlit run Steamlit_app.py
``` 
The app will launch in your browser at http://localhost:8501  

## Example Dataset Format
```
transcript,audio_path
"Hello, how are you?","audio/hello.wav"
"Good morning!","audio/good_morning.wav"
```

## Tech Stack

- Streamlit - Web UI
- Pandas - Data Handling
- Librosa & Matplotlib - Audio visualisation
- Pydub & Sounddevice - Audio proessing & recording
- OpenaI GPT Models - Translation