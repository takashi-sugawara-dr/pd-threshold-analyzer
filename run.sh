#!/bin/bash
cd "$(dirname "$0")"
PYTHONWARNINGS="ignore::UserWarning:scipy" /usr/bin/python3 -m streamlit run app.py
