conda create -n lfqa python=3.10
conda activate lfqa

python3 -m pip install torch
python3 -m pip install -r requirements.txt
python3 -m pip uninstall faiss-cpu
python3 -m pip uninstall faiss-gpu
python3 -m pip install faiss-gpu

python3 -m spacy download en_core_web_sm
python3 -m nltk.downloader punkt