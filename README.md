# AI Data Analyst
cddfffd
A production-style AI-powered data analysis application that allows users to upload CSV datasets, ask questions in natural language, generate evidence-backed insights, visualize results, and inspect the underlying Pandas and SQL logic.

## Demo Video

[▶️ Watch the Demo Video](https://drive.google.com/file/d/1SkbrjwtcXDoGrqcBkUascEMKay7Ia67B/view?usp=sharing)

## Screenshots

### 1. Application Dashboard

[📸 View Screenshot 1](https://drive.google.com/file/d/1URTnD0fosZ-KX70Omi2C4B-Txgxm-qJl/view?usp=sharing)

### 2. Data Analysis

[📸 View Screenshot 2](https://drive.google.com/file/d/1floqS-R_GjXIcbAdkRJyQNQMsRvmmyFm/view?usp=sharing)

### 3. Anomaly Detection

[📸 View Screenshot 3](https://drive.google.com/file/d/11kF_RjdGk1CL88s6wt4ckqXOGqZowkkf/view?usp=sharing)

### 4. Charts and Insights

[📸 View Screenshot 4](https://drive.google.com/file/d/1oZbGAQsMXIK9Fnbq4OVDmpLfLFBIPJoa/view?usp=sharing)

## Features

- CSV upload and validation
- Multi-file dataset loading
- Dataset profiling and data-quality checks
- Natural-language data analysis
- Local LLM support through Ollama
- Evidence-backed deterministic analysis
- Business summaries and insights
- Revenue, grouping, ranking and metric analysis
- Monthly trend analysis
- Bar, line and pie visualizations
- Generated Pandas code
- Generated SQL queries
- Deterministic anomaly detection with explanations
- Conversation context for follow-up questions
- Modular Python architecture
- Automated tests
- Docker support
- No paid API required for the core workflow

## Architecture


flowchart TD
    USER[User]

    subgraph UI["Presentation Layer"]
        APP["Streamlit UI<br/>app.py"]
        CHAT["Natural Language Questions"]
        UPLOAD["CSV Upload"]
        OUTPUT["Answers + Evidence<br/>Charts + Pandas + SQL"]
    end

    subgraph CORE["Application Layer"]
        ENGINE["DataEngine<br/>data_engine.py"]
        ANALYST["Analyst<br/>analyzer.py"]
        CONTEXT["Conversation Context"]
    end

    subgraph ANALYSIS["Analysis Layer"]
        PANDAS["Pandas / NumPy"]
        ANOMALY["Anomaly Detection<br/>IQR Method"]
        SQL["SQL Generation"]
        VIZ["Plotly Visualization"]
    end

    subgraph LLM["Optional Local AI Layer"]
        OLLAMA["Ollama<br/>Local LLM"]
        PLANNER["LLM Planning / Fallback"]
    end

    DATA[("CSV Dataset(s)")]

    USER --> APP
    APP --> UPLOAD
    APP --> CHAT

    UPLOAD --> ENGINE
    ENGINE --> DATA

    CHAT --> ANALYST
    ENGINE --> ANALYST
    CONTEXT --> ANALYST

    ANALYST --> PANDAS
    ANALYST --> ANOMALY
    ANALYST --> SQL
    ANALYST --> VIZ

    ANALYST --> OLLAMA
    OLLAMA --> PLANNER
    PLANNER --> ANALYST

    PANDAS --> OUTPUT
    ANOMALY --> OUTPUT
    SQL --> OUTPUT
    VIZ --> OUTPUT

    ANALYST --> OUTPUT
    OUTPUT --> APP
    APP --> USER


### Architecture Flow

1. The user uploads one or more CSV files through the Streamlit interface.
2. `DataEngine` loads and validates the datasets.
3. The `Analyst` receives the natural-language question and available dataset context.
4. Deterministic Pandas/NumPy analysis performs the actual calculations.
5. The system can generate equivalent SQL and display the analytical logic.
6. Anomaly detection applies deterministic statistical rules and explains the detected values.
7. Plotly generates charts when the requested analysis is visualizable.
8. Ollama can provide local LLM planning without requiring a paid API.
9. Conversation context allows follow-up questions such as:
   - "Which region generated the highest revenue?"
   - "How much revenue did that region generate?"
   - "What about East?"
10. The final response is displayed with the answer, reasoning/evidence, code, SQL and visualization where applicable.

## Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| Language | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Plotly |
| Anomaly Detection | Statistical IQR method |
| Local LLM | Ollama |
| Testing | Pytest |
| Containerization | Docker |
| Data Format | CSV |

## Project Structure


ai-data-analyst/
│
├── app.py
│
├── core/
│   ├── __init__.py
│   ├── analyzer.py
│   ├── data_engine.py
│   └── llm.py
│
├── data/
│   └── sample_sales.csv
│
├── tests/
│   └── test_engine.py
│
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt


## Windows PowerShell Setup

Create and activate a virtual environment:


python -m venv .venv
.\.venv\Scripts\Activate.ps1


Install dependencies:


pip install -r requirements.txt


Run the application:


streamlit run app.py


The application will be available at:


http://localhost:8501


## Local LLM

The application supports Ollama for local LLM planning.

This allows the LLM component to run locally without requiring paid API billing.

Install Ollama separately and pull the supported model:


ollama pull llama3.2:3b


Optional environment variables:


$env:LLM_ENABLED="true"
$env:OLLAMA_MODEL="llama3.2:3b"
$env:OLLAMA_BASE_URL="http://localhost:11434"


The core deterministic analysis does not depend on a paid external API.

## Try the Sample Dataset

Upload:


data/sample_sales.csv


Example questions:


Which region generated the highest revenue?



How much revenue did that region generate?



What about East?


What are the top 5 customers by revenue?



Show monthly sales trends.



Which product is underperforming?



What is the average revenue per transaction?



Detect anomalies in the dataset.



Give me a business summary of the uploaded dataset.


## Example Analysis Flow

The application supports both direct and contextual questions.

Example:


User:
Which region generated the highest revenue?

Assistant:
West generated the highest revenue with 244,000.00.


Follow-up:


User:
How much revenue did that region generate?


The application uses the previous conversational context to identify `West` and calculate the corresponding revenue.

Further follow-ups such as:


What about East?


are also supported.

## Evidence-Backed Analysis

The system separates calculation from language generation.

For supported analytical questions:

1. The question is interpreted.
2. The relevant dataset columns are identified.
3. Pandas performs the actual calculation.
4. Evidence is generated from the calculated result.
5. Equivalent Pandas code is displayed.
6. Equivalent SQL can also be displayed.
7. Plotly visualization is generated when applicable.

This prevents the LLM from being the source of truth for numeric calculations.

## Anomaly Detection

The application includes deterministic anomaly detection using the Interquartile Range (IQR) method.

For a numeric metric:


IQR = Q3 - Q1

Lower Bound = Q1 - 1.5 × IQR

Upper Bound = Q3 + 1.5 × IQR


Values outside these bounds are flagged as potential anomalies.

The application also explains the calculation by displaying the relevant quartiles and anomaly boundaries.

## Generated Pandas and SQL

For supported analytical questions, the application exposes:

- Generated Pandas code
- Equivalent SQL query
- Reasoning/evidence used to derive the answer

Pandas remains the execution engine for the uploaded CSV data.

The SQL is provided as an equivalent analytical representation.

## Data Quality

The Data Quality section provides dataset-level checks including:

- Number of rows
- Number of columns
- Missing values
- Duplicate rows
- Column data types
- Missing-value percentage
- Unique values

This helps identify basic data-quality issues before analysis.

## Multi-File Support

The data engine supports loading multiple CSV files in the same session.

Each uploaded dataset is validated and stored independently, allowing the application architecture to be extended for dataset relationships and joins.

## Docker

Build the Docker image:


docker build -t ai-data-analyst .


Run the application:


docker run -p 8501:8501 ai-data-analyst


Then open:


http://localhost:8501


## Testing

Run the test suite:

pytest -q


Python syntax can also be validated with:


python -m py_compile .\app.py
python -m py_compile .\core\analyzer.py
python -m py_compile .\core\data_engine.py


## Demo Flow

The recommended demonstration flow is:

1. Open the AI Data Analyst application.
2. Upload `sample_sales.csv`.
3. Ask:
   `Which region generated the highest revenue?`
4. Show the calculated answer and chart.
5. Show the generated Pandas code and SQL.
6. Ask:
   `How much revenue did that region generate?`
7. Demonstrate conversation context.
8. Ask:
   `Show monthly sales trends.`
9. Display the generated line chart.
10. Trigger anomaly detection.
11. Show the anomaly explanation.

## Design Principles

### Deterministic Analysis

Numeric results are calculated using Pandas/NumPy rather than relying on an LLM to invent values.

### Local-First AI

Ollama can run the LLM locally, reducing dependency on paid external APIs.

### Explainability

The application exposes the reasoning/evidence, generated Pandas code and SQL so that users can understand how results were produced.

### Modular Architecture

Data loading, analysis, LLM interaction and UI logic are separated into dedicated modules.

### Safe Execution

The application does not execute arbitrary Python code generated by an LLM. Supported analytical operations are handled through controlled application logic.

## Assumptions and Implementation Notes

1. Each uploaded CSV is limited to 25 MB.
2. CSV files are expected to contain a valid tabular structure.
3. Pandas is the primary execution engine for analytical calculations.
4. SQL is generated as an equivalent analytical representation.
5. Anomaly detection currently uses the deterministic IQR method.
6. LLM usage is optional for the core deterministic workflow.
7. Arbitrary LLM-generated Python is not executed.
8. Conversation context is maintained within the Streamlit session.
9. Chart generation is based on supported analytical patterns and available dataset columns.
10. The project is designed as a take-home assignment demonstrating an extensible AI data-analysis architecture rather than a fully hardened enterprise deployment.

## Limitations

- Advanced semantic search is not currently implemented.
- Forecasting is not currently implemented.
- Complex multi-table joins are not the primary workflow.
- Authentication and role-based access are outside the current assignment scope.
- Production-scale distributed execution is outside the current scope.
- Advanced observability and evaluation infrastructure are not included in the starter implementation.

## Future Improvements

Possible extensions include:

- Automated multi-table joins
- Forecasting and time-series models
- Additional chart types
- Dashboard generation
- Semantic dataset search
- Tool-calling agent workflows
- Exportable reports
- Authentication
- Caching
- Evaluation datasets and automated quality scoring
- Production observability
- Cloud deployment

## Assignment Deliverables

| Deliverable | Status |
|---|---|
| CSV upload | Complete |
| Natural-language analysis | Complete |
| Business insights | Complete |
| Charts | Complete |
| Pandas code generation | Complete |
| SQL generation | Complete |
| Anomaly detection | Complete |
| Reasoning / evidence | Complete |
| Conversation context | Complete |
| Data quality | Complete |
| Multi-file loading | Supported |
| Local LLM | Supported |
| Docker | Included |
| Tests | Included |
| Demo video | Included |
| Screenshots | Included |
| Architecture diagram | Included |

## License

This project was created as a take-home assignment and demonstration project.
