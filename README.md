
# BiLSTM Fraud Detection

This repository contains a machine learning project that utilizes a BiLSTM model to detect fraudulent transactions. The project encompasses data preprocessing, model training, evaluation, and deployment using a Flask web application.

## Project Structure

```
major_project/
├── app.py
├── bilstm_fraud_detection_v1.0.0/
│   ├── model.py
│   ├── train.py
│   ├── create_model_image.py
│   ├── bilstm_fraud_detection_v1.0.0_scaler.npy
│   ├── scaler.npy
│   ├── model_architecture.png
├── logs/
├── requirements.txt
├── __pycache__/
```

## Features

- **BiLSTM Model**: Implements a Bidirectional LSTM for sequence modeling to detect anomalies in transaction data.
- **Data Preprocessing**: Includes scaling and preparation scripts to ready the data for model ingestion.
- **Model Visualization**: Generates a visual representation of the model architecture.
- **Web Interface**: Provides a Flask-based application for users to input transaction data and receive fraud predictions.
- **Logging**: Captures logs for monitoring and debugging purposes.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Abhinaycs/major_project.git
   cd major_project
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Train the Model**

   ```bash
   python bilstm_fraud_detection_v1.0.0/train.py
   ```

2. **Visualize the Model Architecture**

   ```bash
   python bilstm_fraud_detection_v1.0.0/create_model_image.py
   ```

3. **Run the Flask Application**

   ```bash
   python app.py
   ```

   Access the web interface by navigating to `http://127.0.0.1:5000/` in your web browser.

## Project Components

- **app.py**: Initializes and runs the Flask web application.
- **bilstm_fraud_detection_v1.0.0/model.py**: Defines the architecture of the BiLSTM model.
- **bilstm_fraud_detection_v1.0.0/train.py**: Handles data preprocessing and model training.
- **bilstm_fraud_detection_v1.0.0/create_model_image.py**: Generates a visual representation of the model architecture.
- **bilstm_fraud_detection_v1.0.0_scaler.npy** & **scaler.npy**: Saved scaler objects used for data normalization.
- **model_architecture.png**: Image file showing the model's architecture.
- **logs/**: Directory containing log files generated during training and inference.
- **requirements.txt**: Lists all Python dependencies required to run the project.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, feel free to fork the repository and submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any questions or inquiries, please contact [Abhinaycs](https://github.com/Abhinaycs).
