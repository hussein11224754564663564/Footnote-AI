# Footnote AI

Streamlit app for SEC 10-K forensic screening.

## Local run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Make sure `streamlit_app.py` is at the repository root.
3. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
4. Click `Create app`.
5. Select your repo, branch, and `streamlit_app.py` as the entrypoint.
6. Add your Gemini API key in Streamlit Secrets, not in code.

## Required secret

Set this in Streamlit Cloud secrets:

```toml
GEMINI_API_KEY = "your-key-here"
```

If you want to use a different key name, update the app code to read that secret name.
