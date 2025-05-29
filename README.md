# FinWise AI - Financial Wellness Assistant

FinWise AI is an intelligent, conversational multi-agent system designed to be your personal financial wellness and credit card assistant. It provides expert guidance on credit card benefits, fraud prevention, financial planning, and more.

## Features

- 💳 Credit card recommendations and comparisons
- 💰 Cashback and reward optimization
- 🔐 Fraud prevention and security tips
- 📈 Credit score improvement guidance
- 🧠 Smart financial planning advice

## Multi-Agent System

FinWise AI consists of five specialized agents:

- **BenefitBot**: Explains card perks and rewards
- **SaverBot**: Helps maximize cashback and savings
- **GuardBot**: Handles fraud prevention and card safety
- **ScoreBot**: Advises on credit score building
- **PlannerBot**: Provides long-term financial planning guidance

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to:
```
http://localhost:5000
```

## Project Structure

```
finwise_ai/
├── app.py              # Main Flask application
├── requirements.txt    # Project dependencies
├── templates/         
│   └── index.html     # Main chat interface
└── README.md          # This file
```

## Development

To run the application in development mode:

```bash
flask run --debug
```

## Security Notes

- The application uses in-memory session storage for demonstration purposes
- In production, implement proper session management and database storage
- Never store sensitive financial information
- Implement proper authentication and authorization
- Use HTTPS in production

## License

MIT License 