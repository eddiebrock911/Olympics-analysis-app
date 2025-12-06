# 🏅 Olympics Data Analysis Helper

यह फ़ाइल `helper.py` एक **utility module** है जो Olympics dataset पर विभिन्न प्रकार की **data analysis और visualization support** के लिए functions प्रदान करती है। इसका उपयोग मुख्यतः Streamlit या Jupyter आधारित analytics apps में किया जाता है।

---

## 📌 Features

* ✅ Medal tally निकालना (Year और Country wise)
* ✅ Year और Country dropdown के लिए dynamic lists
* ✅ Time-series analysis
* ✅ Country wise medal trends
* ✅ Sports vs Year heatmaps
* ✅ Most successful athletes analysis
* ✅ Age distribution analysis (Gold, Silver, Bronze)
* ✅ Male vs Female participation trend
* ✅ Height और Weight based sports analysis

---

## 🗂 Project Structure

```
project-folder/
│
├── helper.py        # Data analysis utility functions
├── app.py           # (Optional) Streamlit / main app
├── athlete_events.csv   # Olympics dataset (example)
├── README.md        # Project documentation
```

---

## ⚙️ Requirements

```bash
pip install pandas
```

Python version: **3.8+ recommended**

---

## ▶️ How to Use

```python
import helper
import pandas as pd

# Load dataset
df = pd.read_csv("athlete_events.csv")

# Medal Tally
medals = helper.medal_tally(df, year="Overall", country="Overall")

# Country & Year List
years, countries = helper.country_year_list(df)

# Year-wise medal trend for a country
trend = helper.yearwise_medal_tally(df, country="India")

# Most successful athletes
top_players = helper.most_successful(df, sport="Overall")
```

---

## 🔍 Function Overview

### 1. `_dedup_medals(df)`

Duplicate medal entries हटाता है ताकि medals over‑count न हों।

### 2. `medal_tally(df, year, country)`

Year और Country के आधार पर Gold, Silver, Bronze और Total medals देता है।

### 3. `country_year_list(df)`

Dropdown menus के लिए Year और Country की list लौटाता है।

### 4. `data_over_time(df, col)`

किसी feature का Year wise trend देता है।

### 5. `yearwise_medal_tally(df, country)`

किसी एक country का year‑by‑year medal count देता है।

### 6. `country_event_heatmap(df, country)`

Country vs Sport vs Year आधारित heatmap data देता है।

### 7. `sports_event_heatmap(df)`

Overall sports participation heatmap देता है।

### 8. `most_successful(df, sport, country)`

Most successful athletes की ranking निकालता है।

### 9. `age_distributions(df)`

Athletes की age distribution देता है (overall + medal wise)।

### 10. `male_female_trend(df)`

Year wise Male vs Female participation trend देता है।

### 11. `height_weight_data(df, sport)`

Height और Weight based sports analysis के लिए cleaned data देता है।

---

## ❗ Common Issues

### Dataset Column Missing

अगर dataset में ये columns नहीं हैं तो error आएगा:

* `Year`
* `region`
* `Sport`
* `Event`
* `Medal`
* `Name`
* `Age`

✔ Solution: Standard Olympics dataset का ही उपयोग करें।

---

## 🚀 Future Improvements

* Country flags integration
* Medal prediction module
* Athlete performance forecasting
* Interactive dashboard integration

---

## 📜 License

Educational और personal project usage के लिए free है। Commercial use से पहले proper license जोड़ें।

---

## 👨‍💻 Developer

**Ankit Kumar** (Instagram)[https://www.instagram.com/__ankit._.op_/]

Python | Data Analysis | Machine Learning

---

✅ यह README.md आपकी GitHub repository के लिए पूरी तरह ready है। Direct paste करके use करें।

