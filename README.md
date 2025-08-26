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
- **Python**, **scikit-learn**, **pandas**
- **Streamlit** for UI
  
---

## 📂 Project Structure
```
.
├── app.py                  
├── data/                   
    ├── laliga/            
    └── premier_league/     
├── predictor/              
    ├── __init__.py         
    └── predictor.py        
├── README.md            
└── setup.py          
```


---

## 📊 Data Sources  

The historical match and season data used in this project is sourced from:  

- [Spanish LaLiga Dataset (DataHub)](https://datahub.io/core/spanish-la-liga)  
- [English Premier League Dataset (Football-Data.co.uk)](https://www.football-data.co.uk/englandm.php)  




---

## ⚙️ Quick Start

```bash
# 1. Clone repo
git clone https://github.com/aryankrsingh004/Football_Season_Predictor.git
cd Football_Season_Predictor

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # (Windows: .\venv\Scripts\activate)

# 3. Install dependencies
pip install -r requirements.txt
pip install -e .

# 4. Run Streamlit app
streamlit run app.py
