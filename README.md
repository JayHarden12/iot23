# 🔒 IoT-23 Botnet Classification System

An advanced machine learning platform for IoT security analysis, featuring state-of-the-art models for botnet family classification and binary malware detection using the IoT-23 dataset.

## ✨ Key Features

### 📊 **Data Explorer**
- Interactive data visualization with Plotly charts
- Class distribution analysis with pie charts and bar graphs
- Data quality metrics and statistics
- Sample data download functionality
- Real-time memory usage monitoring

### 🤖 **Model Training**
- **5 Advanced ML Algorithms:**
  - Random Forest Classifier
  
  - 1D Convolutional Neural Network (CNN)
  - Long Short-Term Memory (LSTM)
- Comprehensive hyperparameter tuning
- Cross-validation with multiple repeats
- Real-time training progress monitoring
- Performance metrics tracking

### 🔮 **Predictions**
- Upload new data for real-time predictions
- Confidence scores and probability estimates
- Interactive prediction distribution analysis
- Results download in CSV format

### 📈 **Analytics Dashboard**
- Feature correlation analysis
- Model performance comparison
- Class balance analysis
- Automated model recommendations
- Interactive visualizations

### ⚡ **Performance & Reliability**
- Intelligent caching system
- Memory usage monitoring
- Automatic garbage collection
- Progress indicators for all operations
- Error handling and validation

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- 8GB+ RAM recommended (for large datasets)
- 2GB+ free disk space

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd iot23-classification
```

2. **Create virtual environment:**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download IoT-23 dataset:**
   - Place `IoT-23.zip` in the project root, or
   - Point the app to an extracted dataset directory

5. **Launch the application:**
```bash
streamlit run app.py
```

6. **Open your browser:**
   - Navigate to `http://localhost:8501`
   - The application will load with a modern, responsive interface

### 🎯 First Steps

1. **Load Data**: Use the Data Explorer tab to load sample data
2. **Configure Model**: Choose your preferred algorithm and hyperparameters
3. **Train Model**: Start training with real-time progress monitoring
4. **Make Predictions**: Upload new data and get instant predictions
5. **Analyze Results**: Use the Analytics tab for detailed insights

## 📊 Dataset Information

### IoT-23 Dataset
The IoT-23 dataset is a comprehensive collection of IoT network traffic data containing both benign and malicious samples. The application intelligently handles the large dataset size through:

- **Smart Sampling**: Configurable row limits per file to manage memory usage
- **Automatic Label Detection**: Identifies label columns across different file formats
- **PRD Family Support**: Focuses on specific botnet families (Benign, Mirai, Torii, Kenjiro, Trojan)
- **Format Flexibility**: Supports CSV, TSV, and log file formats

### Supported Tasks
- **Multiclass Classification**: Botnet family identification
- **Binary Classification**: Malicious vs benign detection

## 🏗️ Architecture

### Project Structure
```
iot23-classification/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
├── IoT-23.zip              # IoT-23 dataset (9GB)
└── src/
    └── iot23/
        ├── __init__.py
        ├── data_loader.py   # Dataset loading and sampling
        ├── preprocess.py    # Data preprocessing pipeline
        ├── modeling.py      # Model definitions and training
        └── utils.py         # Utility functions
```

### Key Components

#### Data Processing Pipeline
- **Data Loader**: Handles zip files and directory scanning
- **Preprocessor**: Automatic feature engineering and scaling
- **Validator**: Comprehensive data validation and error handling

#### Model Training System
- **Multiple Algorithms**: 5 different ML approaches
- **Hyperparameter Tuning**: Interactive parameter adjustment
- **Cross-Validation**: Multiple training runs for robust results
- **Performance Monitoring**: Real-time metrics and memory tracking

#### User Interface
- **Responsive Design**: Modern, mobile-friendly interface
- **Interactive Visualizations**: Plotly-powered charts and graphs
- **Real-time Feedback**: Progress indicators and status updates
- **Error Handling**: Comprehensive error messages and recovery

## 🔧 Configuration

### Model Parameters

#### Random Forest
- `n_estimators`: Number of trees (50-600)
- `max_depth`: Maximum tree depth (0=unlimited)
- `min_samples_split`: Minimum samples to split (2-20)

- `n_estimators`: Number of boosting rounds (50-600)
- `learning_rate`: Step size shrinkage (0.01-0.3)
- `max_depth`: Maximum tree depth (1-20)

- `C`: Regularization parameter (0.1-100.0)
- `kernel`: Kernel type (rbf, linear, poly, sigmoid)
- `gamma`: Kernel coefficient (scale, auto, or numeric)

#### Deep Learning (CNN/LSTM)
- `epochs`: Training epochs (3-50)
- `batch_size`: Batch size (32-1024)
- `patience`: Early stopping patience (1-10)

### Performance Settings
- **Caching**: Enable/disable result caching
- **Memory Cleanup**: Automatic garbage collection
- **Sampling**: Adjustable data sampling parameters

## 🚨 Troubleshooting

### Common Issues

#### Memory Problems
- **Symptom**: Application crashes or becomes unresponsive
- **Solution**: Reduce `Max files` and `Rows per file` in sidebar
- **Prevention**: Monitor memory usage in sidebar

#### Data Loading Errors
- **Symptom**: "No data loaded" message
- **Solution**: Check dataset path and file formats
- **Debug**: Use "Preview files" in sidebar

#### Model Training Failures
- **Symptom**: Training fails with error message
- **Solution**: Check data quality and class distribution
- **Debug**: Review error messages and adjust parameters

#### Performance Issues
- **Symptom**: Slow loading or training
- **Solution**: Enable caching and reduce sample size
- **Optimization**: Use performance settings in sidebar

### Error Codes
- `ValueError`: Invalid input parameters or data
- `FileNotFoundError`: Dataset path not found
- `MemoryError`: Insufficient memory for operation
- `ImportError`: Missing required dependencies

## 📈 Performance Metrics

### Model Evaluation
- **Accuracy**: Overall classification accuracy
- **F1-Score**: Macro and weighted F1 scores
- **Balanced Accuracy**: Handles class imbalance
- **MCC**: Matthews Correlation Coefficient
- **ROC-AUC**: Receiver Operating Characteristic
- **PR-AUC**: Precision-Recall Area Under Curve

### System Performance
- **Memory Usage**: Real-time memory monitoring
- **Training Time**: Per-epoch and total training time
- **Caching**: Cache hit rates and performance
- **Throughput**: Data processing speed

## 🔮 Future Enhancements

### Planned Features
- **SHAP Explanations**: Model interpretability for tree-based models
- **Hyperparameter Optimization**: Automated parameter tuning
- **Model Persistence**: Save/load trained models
- **Batch Processing**: Process multiple files simultaneously
- **API Integration**: REST API for model serving
- **Cloud Deployment**: Docker and cloud deployment options

### Advanced Analytics
- **Feature Engineering**: Automated feature creation
- **Ensemble Methods**: Combine multiple models
- **Time Series Analysis**: Temporal pattern detection
- **Anomaly Detection**: Unsupervised learning approaches

## 📚 References

### Dataset
- **IoT-23 Dataset**: [Stratosphere Lab](https://www.stratosphereips.org/datasets-iot23)
- **Paper**: "IoT-23: A labeled dataset with malicious and benign IoT network traffic"

### Technologies
- **Streamlit**: Web application framework
- **Scikit-learn**: Machine learning library
 
- **TensorFlow**: Deep learning platform
- **Plotly**: Interactive visualization library

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:
- Code style and standards
- Testing requirements
- Documentation updates
- Feature requests and bug reports

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions, issues, or contributions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the documentation
- Contact the development team
