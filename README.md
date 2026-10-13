# Tourism Project as Assignment No 10
# Automated MLOps Pipeline for Wellness Tourism Package Purchase Prediction
# Business Context
"Visit with Us" is a leading travel company aiming to transform its customer engagement strategy through data-driven decision-making. With the launch of a new Wellness Tourism Package, the organization faces challenges in accurately identifying customers who are most likely to purchase the package.

The existing manual customer selection process is inefficient, inconsistent, and prone to human error, resulting in missed sales opportunities and ineffective marketing campaigns. To overcome these challenges, the company seeks to implement a scalable, automated, and reliable predictive system that enables precise customer targeting and continuous model improvement.

By adopting an MLOps-driven approach, this project integrates data preprocessing, machine learning model development, deployment, and CI/CD automation using GitHub Actions, ensuring operational efficiency and adaptability to evolving customer behavior.

# Objective
The primary objective of this project is to design and deploy an end-to-end MLOps pipeline that predicts whether a customer is likely to purchase the Wellness Tourism Package before being contacted by the marketing team.

As an MLOps Engineer, the responsibilities include:

Automating data cleaning, preprocessing, and feature transformation
Building, training, and evaluating a predictive machine learning model
Implementing CI/CD pipelines using GitHub Actions
Ensuring model scalability, reproducibility, and continuous improvement
This solution enables the business to make data-driven marketing decisions, optimize customer outreach, and improve customer acquisition while reducing operational overhead.

# Project Scope
The MLOps pipeline automates the following stages:

Data ingestion and validation
Data cleaning and preprocessing
Feature engineering and transformation
Model training and evaluation
Model deployment readiness
CI/CD automation for continuous integration and updates

# Data Description
The dataset consists of customer demographics and interaction attributes used to predict the likelihood of purchasing the Wellness Tourism Package.

# Target Variable
ProdTaken: Indicates whether the customer purchased the package

**0** – No

**1** – Yes

# Customer Details

**Customer ID:** Unique identifier for each customer

**Age:** Age of the customer

**Type of Contact:** Mode of contact (Company Invited / Self Inquiry)

**City Tier:** City category based on development and population (Tier 1, Tier 2, Tier 3)

**Occupation:** Customer’s profession (Salaried, Freelancer, etc.)

**Gender:** Gender of the customer

**Number Of Person Visiting:** Total number of people traveling with the customer

**Preferred Property Star:** Preferred hotel star rating

**Marital Status:** Marital status (Single, Married, Divorced)

**Number Of Trips:** Average number of annual trips

**Passport:** Passport availability (0: No, 1: Yes)

**Own Car:** Car ownership (0: No, 1: Yes)

**Number Of Children Visiting:** Number of children below 5 years

**Designation:** Job designation

**Monthly Income:** Monthly income of the customer

# Customer Interaction Data
**Pitch Satisfaction Score:** Customer satisfaction score for the sales pitch

**Product Pitched:** Product pitched to the customer

**Number Of Follow ups:** Number of follow-ups after the sales pitch

**Duration Of Pitch:** Duration (in minutes) of the sales pitch

# Technology Stack

**Programming Language:** Python

**Machine Learning:** Scikit-learn

**Version Control:** Git & GitHub

**CI/CD Automation:** GitHub Actions

**MLOps Practices:** Automated training, testing, and deployment readiness

# Business Impact
This predictive MLOps solution enables:

Accurate identification of high-potential customers
Reduced manual effort and operational inefficiencies
Improved marketing campaign performance
Scalable and reproducible model deployment
Faster adaptation to changing customer behavior

# Conclusion
By implementing an automated MLOps pipeline, "Visit with Us" can significantly enhance its marketing effectiveness, improve customer targeting accuracy, and drive sustainable business growth. This project demonstrates the practical application of MLOps principles to solve real-world business challenges in the tourism industry.
