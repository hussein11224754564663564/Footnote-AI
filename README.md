# Footnote AI

Streamlit app for SEC filing forensic screening with provider support for Gemini, Anthropic, AWS Bedrock, Azure OpenAI, Groq, Mistral, and Hugging Face.

## Local run

```bash
pip install -r requirements.txt
streamlit run FootnoteAI.py
```

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Make sure `FootnoteAI.py` is at the repository root.
3. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
4. Click `Create app`.
5. Select your repo, branch, and `FootnoteAI.py` as the entrypoint.
6. Add any provider API key in Streamlit Secrets, not in code.

## Required secret

Set one of these in Streamlit Cloud secrets:

```toml
GEMINI_API_KEY = "your-key-here"
ANTHROPIC_API_KEY = "your-key-here"
AWS_REGION = "us-east-1"
AWS_ACCESS_KEY_ID = "your-access-key-id"
AWS_SECRET_ACCESS_KEY = "your-secret-access-key"
AWS_SESSION_TOKEN = "your-session-token" # optional
BEDROCK_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0" # optional
AZURE_OPENAI_API_KEY = "your-key-here"
GROQ_API_KEY = "your-key-here"
MISTRAL_API_KEY = "your-key-here"
HF_TOKEN = "your-key-here"
```

AWS Bedrock can also use the default AWS credential chain if your deployment environment already has a role or credentials configured.
