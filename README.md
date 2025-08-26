# 🏆 Football League Season Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://football-season-predictor.streamlit.app/)

A **Streamlit web app** that predicts the final standings of major European football leagues using machine learning on historical season data.  

---

## 🚀 Live Demo  
👉 [football-season-predictor.streamlit.app](https://football-season-predictor.streamlit.app/)

---

## ✨ Features
-  **Multi-League Support** – LaLiga, Premier League
-  **ML Model** – Random Forest Classifier (scikit-learn)  
-  **Performance Analysis** – Interactive error & accuracy charts  
-  **Custom Predictions** – Input future year & promoted teams  
-  **Interactive UI** – Built with Streamlit  

---

## 🛠️ Tech Stack
- **Python**, **scikit-learn**, **pandas**, **numpy**  
- **Streamlit** for UI
  
---

## 📂 Project Structure
.<br />
├── data/ # League data (CSV files per season)<br />
│ ├── laliga/ <br />
│ └── premier_league/ <br />
├── predictor/ # Core package (model + processing) <br />
├── app.py # Streamlit app <br />
├── setup.py # Package setup <br />
└── README.md 



---

## ⚙️ Quick Start

```bash
# 1. Clone repo
git clone <your-repo-url>
cd <your-repo>

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # (Windows: .\venv\Scripts\activate)

# 3. Install dependencies
pip install -r requirements.txt
pip install -e .

# 4. Run Streamlit app
streamlit run app.py
