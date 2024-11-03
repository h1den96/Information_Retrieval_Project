=================================================
Project Setup Guide
Follow these steps to run the project:

Clone or download this repository:

bashCopygit clone <repository-url>
cd <project-directory>

Create and activate a virtual environment:

Windows:
bashCopypython -m venv .venv
.venv\Scripts\activate
macOS/Linux:
bashCopypython -m venv .venv
source .venv/bin/activate

Install required packages:

bashCopypip install -r requirements.txt
Required Packages
The project uses the following Python packages:

nltk
numpy
matplotlib
requests
beautifulsoup4

Alternatively, you can run setup.py which will automatically create the virtual environment and install all dependencies:
bashCopypython setup.py
==============================================================