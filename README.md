
# CSE4288F24_Grp17

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Functions](#functions)

## Overview
This project is part of the CSE4288 course group project.
The goal of this project is to analyze and visualize salary data based on 
various factors such as education level, job title, gender, age, and years of experience.

## Project Structure


## Installation
To set up the project, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/ahmetabdullahgultekin/CSE4288F24_Grp.git
    cd CSE4288F24_Grp
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
To run the project, execute the `main.py` script:
```sh
python src/main.py
```

## Data
The project uses a CSV file `salary_data.csv` containing the following columns:
- `Education Level`
- `Job Title`
- `Gender`
- `Salary`
- `Age`
- `Years of Experience`

## Functions
### Data Processing
- `load_data(file_path)`: Loads the data from a CSV file.
- `preprocess_data(df)`: Preprocesses the data by handling missing values and encoding categorical columns.
- `save_data(df, file_path)`: Saves the processed data to a CSV file.

### Visualization
- `plot_correlation_matrix(df)`: Plots the correlation matrix of the DataFrame.
- `plot_education_level(df)`: Plots the distribution of education levels.
- `plot_gender_vs_education(df)`: Plots the relationship between gender and education level.
- `plot_numerical_columns(df)`: Plots histograms for numerical columns.
- `plot_gender_vs_salary(df)`: Plots the relationship between gender and salary.
- `plot_histogram(df)`: Plots a histogram of the specified column.
```

This `README.md` provides a comprehensive overview of the project, including installation instructions,
 usage, data description, functions, and contribution guidelines.
