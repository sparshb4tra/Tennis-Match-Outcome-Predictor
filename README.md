# 🎾 Tennis Match Outcome Predictor

## 🏆 Overview
The **Tennis Match Outcome Predictor** is a machine learning project developed to predict professional ATP tennis match outcomes using historical data. As demonstrated in our comparative analysis, ensemble methods significantly outperform traditional classifiers for this task, with XGBoost achieving remarkable 95.86% accuracy.

## 📊 Dataset
The dataset consists of ATP (Association of Tennis Professionals) matches from the 2024 season, containing:
- **Player Information**: Winner & loser IDs, names, handedness, height, age, and country
- **Match Statistics**: Aces, double faults, break points, service points won
- **Match Details**: Score, tournament round, duration, surface type
- **Ranking Information**: Player rankings and ranking points

## 🏗️ Project Pipeline
1. **Data Loading**: Importing ATP match data from structured CSV files
2. **Data Cleaning**: Handling missing values, removing duplicates, and converting data types
3. **Preprocessing & Feature Engineering**:
   - Creating rank difference features
   - Calculating ace percentages
   - Normalizing numerical features
   - Encoding categorical variables
4. **Model Implementation**: Training and evaluating five machine learning classifiers
5. **Performance Analysis**: Comparing model accuracy, precision, recall, and F1-scores

## 🧠 Machine Learning Models & Performance

| Model | Accuracy | Key Features |
|-------|----------|-------------|
| **XGBoost** | **95.86%** | Best performer with lowest log loss (0.0957) |
| **Random Forest** | **89.68%** | Second-best accuracy, good feature importance insights |
| **SVM** | 64.99% | Moderate performance with linear boundary |
| **Logistic Regression** | 65.23% | Simple interpretable model |
| **Naive Bayes** | 61.82% | Baseline probabilistic classifier |

### 📈 Key Findings
- **Ensemble methods** (XGBoost and Random Forest) significantly outperform traditional classifiers
- **Player rankings** and **ace percentages** consistently emerge as the most influential predictors
- Detailed feature importance analysis reveals which factors most strongly influence tennis match outcomes

## 🚀 Installation & Usage
```bash
# Clone the repository
git clone https://github.com/yourusername/tennis-match-predictor.git
cd tennis-match-predictor

# Install dependencies
pip install -r requirements.txt

# Run the main script
python main.py
```

## 📋 Requirements
- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn

## 🔬 Feature Importance
Our analysis revealed that the following features have the highest predictive value:

1. Player rankings (winner_rank, loser_rank)
2. Ranking difference between players
3. Ace percentages (both winner and loser)
4. Match duration
5. Tournament round
6. Surface type
7. Player handedness

## 📊 Model Evaluation
Each model was evaluated using:
- Accuracy
- Precision, Recall, and F1-Score
- Log Loss
- ROC curves

## 🔮 Future Work
- Incorporate additional features like head-to-head records and player fatigue metrics
- Implement temporal analysis to account for recent form and career trajectories
- Explore deep learning approaches (RNNs, Transformers)
- Develop real-time in-match prediction capabilities
- Extend models to WTA (women's tennis) data

## 📚 Published Paper
Coming soon...

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/sparshb4tra/Tennis-Match-Outcome-Predictor/issues).

## 📝 License
[MIT](https://choosealicense.com/licenses/mit/)

## 📧 Contact
Sparsh Batra - [Portfolio](https://sbatra.xyz) - me@sbatra.xyz

Project Link: [https://github.com/sparshb4tra/Tennis-Match-Outcome-Predictor](https://github.com/sparshb4tra/Tennis-Match-Outcome-Predictor)
