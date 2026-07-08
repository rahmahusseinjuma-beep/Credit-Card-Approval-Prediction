-- ============================================================
-- Credit Card Approval Prediction - Database Schema
-- Matches the Entity Relationship Diagram (see Project Design Phase)
-- ============================================================

CREATE TABLE Users (
    UserID      INTEGER PRIMARY KEY AUTOINCREMENT,
    Name        VARCHAR(100) NOT NULL,
    Email       VARCHAR(150) NOT NULL UNIQUE,
    Password    VARCHAR(255) NOT NULL,
    Role        VARCHAR(50) NOT NULL  -- e.g. 'Analyst', 'Compliance Officer', 'Customer'
);

CREATE TABLE Applicant_Details (
    ApplicantID      INTEGER PRIMARY KEY AUTOINCREMENT,
    UserID           INTEGER NOT NULL,
    IncomeType       VARCHAR(50),
    EducationType    VARCHAR(50),
    FamilyStatus     VARCHAR(50),
    HousingType      VARCHAR(50),
    EmploymentDays   INTEGER,
    FOREIGN KEY (UserID) REFERENCES Users(UserID)
);

CREATE TABLE Credit_History (
    HistoryID      INTEGER PRIMARY KEY AUTOINCREMENT,
    ApplicantID    INTEGER NOT NULL,
    MonthsBalance  INTEGER,
    PaymentStatus  VARCHAR(5),   -- raw multi-class code: 0-5, C, X
    OverdueStatus  VARCHAR(50),
    FOREIGN KEY (ApplicantID) REFERENCES Applicant_Details(ApplicantID)
);

CREATE TABLE ML_Model (
    ModelID        INTEGER PRIMARY KEY AUTOINCREMENT,
    ModelName      VARCHAR(100) NOT NULL,
    AlgorithmType  VARCHAR(50) NOT NULL,  -- e.g. 'Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost'
    Accuracy       FLOAT,
    ModelFile      VARCHAR(255)  -- path to the saved .pkl file
);

CREATE TABLE Approval_Prediction (
    PredictionID    INTEGER PRIMARY KEY AUTOINCREMENT,
    ApplicantID     INTEGER NOT NULL UNIQUE,  -- UNIQUE enforces the 1:1 relationship with Applicant_Details
    ModelID         INTEGER NOT NULL,
    ApprovalResult  VARCHAR(20) NOT NULL,     -- 'Approved' or 'Rejected'
    RiskCategory    VARCHAR(20),              -- 'Low Risk' or 'High Risk'
    PredictionDate  DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (ApplicantID) REFERENCES Applicant_Details(ApplicantID),
    FOREIGN KEY (ModelID) REFERENCES ML_Model(ModelID)
);

-- ============================================================
-- Seed data: the four models trained and compared in Epic 4
-- ============================================================
INSERT INTO ML_Model (ModelName, AlgorithmType, Accuracy, ModelFile) VALUES
    ('Logistic Regression Model', 'Logistic Regression', 0.591, 'model/logistic_regression.pkl'),
    ('Decision Tree Model',       'Decision Tree',       0.955, 'model/best_model.pkl'),
    ('Random Forest Model',       'Random Forest',       0.960, 'model/random_forest.pkl'),
    ('XGBoost Model',             'XGBoost',             0.935, 'model/xgboost.pkl');
