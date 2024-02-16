# Nepal Constitution QA Bot

This project introduces a Question and Answer (QA) Bot, specifically designed to provide insights and answers based on the Constitution of Nepal. Utilizing the advanced capabilities of the LLM MistralAI Mixtral-8x7B-Instruct-v0.1 model from Hugging Face, this application offers an interactive and informative way to explore the contents of the Constitution of Nepal.

## Features

- **Interactive QA Bot**: Users can ask questions related to the Constitution of Nepal, and the bot will provide answers based on the constitutional text.
- **Powered by MistralAI**: Utilizes the cutting-edge LLM technology from Hugging Face for accurate and context-aware responses.
- **User-Friendly Interface**: Built with Streamlit, the application boasts a simple and intuitive UI, making it accessible to everyone.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python (3.7 or later)
- Chainlit
- Hugging Face's Transformers library

### Installation

1. Clone this repository:

```bash
git https://github.com/Badal-Shrestha/llm_qa_nepal_constitution
cd nepal-constitution-qa-bot

pip install -r requirements.txt

```
### HugginFace api token
- Replace it with your hugginface api token in config.py file

### Running the Application
- **Execute the following command in your terminal:**
```bash
chainlit run app.py
```
### Usage
Upon launching the application, you will be greeted with a text input field. Here, you can type in your questions regarding the Constitution of Nepal. The QA Bot will process your query and provide a relevant answer based on the constitutional document.

### Contributing
Contributions are welcome! If you have suggestions for improving this application, feel free to fork the repository, make your changes, and submit a pull request.

### Acknowledgments
- **Constitution Source**: Law Commission of Nepal
- **Model Training and Technology**: Hugging Face
