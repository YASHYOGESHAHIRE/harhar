from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import pandas as pd
import os
from fastapi.staticfiles import StaticFiles
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
# Directory to store uploaded datasets
UPLOAD_DIR = "uploaded_datasets"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Helper function to process dataset and generate preprocessing code
def process_dataset_and_problem(dataset_path: str, problem_statement: str):
    # Load dataset with pandas
    df = pd.read_csv(dataset_path)

    # Example steps for preprocessing (can be modified)
    preprocessing_steps = [
        "Load Dataset",
        "Remove Missing Values",
        "One-Hot Encode Categorical Variables",
        "Scale Features",
        "Split Data into Train/Test"
    ]

    # Example code generation (modify based on actual use case)
    code = f"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('{dataset_path}')

# Remove missing values
df = df.dropna()

# One-hot encode categorical variables
df = pd.get_dummies(df)

# Split features and target
X = df.drop('target_column', axis=1)
y = df['target_column']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# TODO: Add model training and evaluation steps based on the problem statement
"""
    return preprocessing_steps, code

# Route to handle dataset and problem statement submission
@app.post("/process/")
async def process_file(dataset: UploadFile = File(...), problem_statement: str = Form(...)):
    dataset_path = os.path.join(UPLOAD_DIR, dataset.filename)
    
    # Save the uploaded dataset
    with open(dataset_path, "wb") as f:
        f.write(await dataset.read())
    
    # Process the dataset and problem statement to generate code and flowchart steps
    steps, code = process_dataset_and_problem(dataset_path, problem_statement)
    
    # Return the flowchart steps and the generated code as JSON
    return JSONResponse({
        "flowchart_steps": steps,
        "generated_code": code
    })

# Route to handle code execution (simulated for now)
@app.post("/execute/")
async def execute_code(request: dict):
    code = request.get("code", "")
    
    # Simulate code execution for now (you can later run the code in a safe environment)
    return {"message": "Code executed successfully", "executed_code": code}
