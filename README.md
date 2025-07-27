PDF Heading Extractor

This script extracts headings from PDF files using PyMuPDF and spaCy NLP.
Setup Instructions

    Clone the repository (if not done already):

git clone <your-repo-url>
cd <your-repo-folder>

    Create and activate a virtual environment (recommended):

python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate

    Install required Python packages:

pip install -r requirements.txt

    Download spaCy English language model:

python -m spacy download en_core_web_sm

How to run

Place your PDF file (e.g., file02.pdf) in the project directory, then run:

python your_script_name.py

The output JSON will be saved as output1.json.
