# Gritap AMR

## Background
Antimicrobial resistance (AMR) occurs when pathogens change and find ways to resist the effects of anti-infectives. AMR, particularly in Gram-negative bacteria, is widely recognized as one of the biggest threats to global public health today, causing 700,000 deaths annually[1].

Gritap AMR uses Machine learning (ML) models using sourced data (ATLAS antimicrobial resistance (AMR) data), to explore data trends, build predictive ML models, and forecast resistance dynamics through an interactive web interface.

Gritap AMR is powered by Pfizer’s ATLAS (Antimicrobial Testing Leadership and Surveillance) dataset[2]. ATLAS is a large-scale global program that aggregates AMR data from three surveillance initiatives (TEST, AWARE, INFORM), spanning more than 14 years across over 60–73 countries.
The dataset includes cumulative data on more than 556,000 bacterial isolates, with updates released approximately every 6 to 12 months via the ATLAS website and mobile app

Specifically, the dataset accessible through the Vivli AMR Register includes:   
- Around 917,049 antibiotic isolates and 21,631 antifungal isolates as of June 2024.
- It covers both pediatric data and limited genotypic information, such as the presence or absence of β-lactamase genes.
- MIC (minimal inhibitory concentration) data, along with metadata including country, specimen type, year, organism, antimicrobial used, and basic demographics (e.g., age range, gender) are included.

Read more about the Pfizer's ATLAS Program dataset [here](https://amr.vivli.org/members/research-programs/).

    
LINK to the deployed PROTOTYPE: https://gritapamr.onrender.com/

Github link with python and streamlit source code: https://github.com/AlBadruSsenoga/gritapamr

## Four Sections

Gritap AMR is a clinical decision-support system that delivers 4 integrated capabilities that take the user from exploration to action:
1.	**Data Analysis**: Explore interactive, hover-enabled visualizations (Plotly) of demographic, bacterial, and resistance data, enriched with real-time observations, implications, and recommendations for clinical action.
2.	**Train Model**: Build predictive models for any antibiotic–bacteria pair using nine classification algorithms (e.g., XGBoost, Random Forest, Logistic Regression). Enhance interpretability with DAG-based causal effect estimation via DoWhy, revealing which biological or contextual factors drive resistance.
3.	**Make a Forecast**: Anticipate resistance threats with Prophet-based forecasting of future resistance trends, helping governments and hospitals plan years ahead.
4.	**Make Prediction**: Generate real-time, condition-specific resistance predictions with 97% accuracy, backed by causal effect analysis that explains why resistance is likely.



## Getting Started

### Prerequisites
- Python 3.9 or later
- Recommended: set up a virtual environment (`venv`, `conda`)

### Installation

```bash
git clone https://github.com/AlBadruSsenoga/gritapamr.git
cd gritapamr
pip install -r requirements.txt
```
### Run app
streamlit run app.py

---
## Functionality

- **Data Analysis** - Demographic, species, and antibiotic resistance visualizations with insights
- **Train Model** - Interface to train classification models with causal analysis and export
- **Make a Forecast** - Forecast MIC trajectories with Prophet, offering trend visualizations
- **Make Prediction**	- Predict susceptibility status, estimate causal effects, and share results




References
1. Review on Antimicrobial Resistance. Tackling drug-resistant infections globally: final report and recommendations. May 2016. Available at: https://amr-review.org/sites/default/files/160525_Final%20paper_with%20cover.pdf Last accessed May 2020.

2. AMR Industry Alliance. Pfizer – Antimicrobial Testing Leadership and Surveillance (ATLAS). Available at: https://www.amrindustryalliance.org/case-study/antimicrobial-testing-leadership-and-surveillance-atlas/ Last accessed May 2020.
