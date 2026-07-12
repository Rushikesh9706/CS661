# BharatScope 🇮🇳

**Interactive Visual Analytics of Indian Economic Development**

BharatScope is a comprehensive data analytics platform that provides interactive dashboards for exploring India's economic progress, policy outcomes, human development indices, state-level performance, and sectoral transformation. Built with Streamlit and Plotly, it offers deep insights into India's development trajectory across multiple dimensions.

---

## Features

### 📈 **Economic Development Explorer**
- Analyze Gross State Domestic Product (GSDP) trends across Indian states
- Track GSDP growth rates and per-capita income evolution
- Compare economic indicators year-over-year
- Visualize top-performing states in economic metrics

### 📜 **Policy Impact Explorer**
- Examine the impact of major Indian economic policies over time
- Correlate policy implementation with key indicators (poverty, employment, growth)
- Interactive policy timeline visualization
- Track indicator changes across different policy eras

### 🏥 **Human Development Dashboard**
- Explore literacy rates across states and decades
- Monitor population growth and demographic trends
- Track infant mortality rate improvements
- Analyze poverty trends across regions
- Compare state-level human development indices (HDI)

### ⚖️ **State Comparison Dashboard**
- Benchmark two states head-to-head across multiple indicators
- Compare against national averages
- Multiple visualization types: dumbbell, lollipop, bullet, butterfly, bubble, reference-dot, and radial charts

### 🌾 **Sector Dashboard**
- Track agricultural, industrial, and service sectors' share in GDP
- Monitor sectoral transformation over time
- Analyze structural changes in India's economy

---

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Rushikesh9706/CS661.git
cd CS661

pip install -r requirements.txt


streamlit run app.py

The application will be available at http://localhost:8501

Dependencies
streamlit - Interactive web framework for data apps
pandas - Data manipulation and analysis
plotly - Interactive visualization library
openpyxl - Excel file handling
See requirements.txt for exact versions.

File Structure
Code
CS661/
├── app.py                                    # Main Streamlit application
├── requirements.txt                          # Python dependencies
├── CS661 Dataset - Sheet1.csv               # Main development indicators dataset
├── GDP.xlsx                                  # State-wise GDP data
├── employment data.csv                      # Employment statistics
├── Indian Policy Timeline.xlsx              # Major policy information
├── Infant Mortality.xlsx                    # Infant mortality rates by state
├── Literacy rate.xlsx                       # Literacy data
├── population.xlsx                          # Population census data
├── Poverty.xlsx                             # Poverty reference data
├── agriculture_industrial.xlsx              # Sector composition data
└── all_states_interpolated_poverty.csv      # Interpolated poverty ratios
Usage Guide
Navigation
The application features six main sections accessible via the top navigation bar:

🏠 Home - Overview and quick navigation to all dashboards
📈 Economic Development - GSDP and per-capita income analysis
📜 Policy Impact - Policy timeline and indicator correlations
🏥 Human Development - Literacy, population, poverty, and mortality trends
⚖️ State Comparison - Head-to-head state benchmarking
🌾 Sector Dashboard - Agricultural, industrial, and service sectors analysis
Sidebar Controls
Year Selector - Choose specific years for analysis (varies by dashboard)
State/UT Selector - Filter data by state or view all-India aggregates
Interactive Features
Hover over charts for detailed metrics
Click legend items to show/hide data series
Zoom and pan on interactive plots
Download visualizations as images
Technical Details
Data Processing
Normalization: State names are standardized across datasets for consistency
Type Conversion: Numeric columns are converted with error handling for missing/invalid data
Interpolation: Poverty data includes interpolated values for years with gaps
Caching: Data loading is optimized with Streamlit's @st.cache_data decorator
Visualization Approach
Comparative Visualization: Multiple chart types (dumbbell, lollipop, bullet, butterfly, bubble, radial) for nuanced comparisons
Color Coding: Consistent color scheme (blue for State 1, orange for State 2, green for India average)
Responsive Design: Adaptive layout for different screen sizes
Data Quality
Comprehensive error handling for missing values
Multiple data source fallbacks for indicators
Year-based filtering to ensure temporal consistency
Key Insights
BharatScope enables analysis of:

Regional Disparities: Identify gaps between leading and lagging states
Sectoral Shifts: Understand India's transition from agriculture to services
Policy Effectiveness: Correlate policy implementation with economic indicators
Development Progress: Track improvements in human development indices
Demographic Trends: Analyze population growth and mortality improvements
Sample Queries
Which state has the highest GSDP growth in 2023?
How has literacy improved in rural states between 1991-2011?
What impact did economic liberalization have on unemployment?
How does Maharashtra's HDI compare to Bihar's?
Which states have successfully reduced infant mortality?
Future Enhancements
Real-time data updates from government sources
Additional indicators (healthcare, education investment)
Predictive models for future trends
Export functionality for reports and analysis
Mobile-responsive interface optimization
Data download capabilities
Contributing
Contributions are welcome! Please feel free to:

Report bugs or data inconsistencies
Suggest new visualizations or indicators
Improve data quality or coverage
Enhance the user interface
License
This project is open source. Check the repository for license information.

Support
For issues, questions, or suggestions:

Open an issue on GitHub
Check existing issues and discussions
Review the data sources for accuracy
Data Acknowledgments
Data sourced from:

Central Statistics Office (CSO), Ministry of Statistics & Programme Implementation
National Sample Survey Office (NSSO)
Census of India
Ministry of Labour & Employment
State government portals
Last Updated: July 12, 2026

Explore India's development journey with BharatScope 📊


