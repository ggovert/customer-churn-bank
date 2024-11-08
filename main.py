import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from openai import OpenAI
from dotenv import load_dotenv
import utils as ut 

load_dotenv()

# connect to GROQ api
client = OpenAI(base_url="https://api.groq.com/openai/v1",
                api_key=os.environ.get("GROQ_API"))

# ----------------------------------
# using power of GROQ, we will create explanation of the prediction

def explain_predictions(probability, input_dict, surname, df):

  # Format the feature importance as a proper Python dictionary string
  feature_importance = {
      "NumOfProducts": 0.330930,
      "IsActiveMember": 0.195791,
      "Age": 0.109685,
      "Geography_Germany": 0.081833,
      "Balance": 0.054735,
      "Geography_Spain": 0.044963,
      "Gender_Male": 0.043983,
      "CreditScore": 0.036846,
      "EstimatedSalary": 0.035971,
      "Tenure": 0.033146,
      "HasCrCard": 0.032117
  }

  prompt = f"""
  You are an expert data scientist at a bank, where you specialize in interpreting and explaining predictions of machine learning models.

  Your machine learning model has predicted that a customer named {surname} has a {round(probability * 100, 1)}% probability of churning, based on the information provided below.

  Here is the customer's information:
  {input_dict}

  Here are the machine learning model's top 10 most important features for predicting churn:
  {feature_importance}

  {pd.set_option('display.max_columns', None)}
  Here are summary statistics for churned customers:
  {df[df['Exited'] == 1].describe().to_string()}

  Here are summary statistics for non-churned customers:
  {df[df['Exited'] == 0].describe().to_string()}

  - If the customer's probability  {probability*100}%  has over a 40% risk of churning, generate a 3-sentence explanation of why they are at risk of churning.
  - If the customer's probability  {probability*100}%  has less than a 40% risk of churning, generate a 3-sentence explanation of why they might not be at risk of churning.
  - Your explanation should be based on the customer's information, the summary statistics of churned and non-churned customers, and the feature importances provided.

  Don't mention the probability of churning, or the machine learning model, or say anything like "Based on the machine learning model's prediction and top 10 most important features", just explain the prediction.
  """

  print("EXPLANATION PROMPT", prompt)
  raw_response = client.chat.completions.create(
    model = "llama-3.2-3b-preview",
    messages = [{
      "role" : "user",
      "content" : prompt
    }],
  )
  return raw_response.choices[0].message.content 


def explain_predictionss(probability, input_dict, surname, df):
    # Calculate averages for comparison
#     churned_avg = df[df['Exited'] == 1].mean()
#     retained_avg = df[df['Exited'] == 0].mean()

    prompt = f"""
    You are a banking expert analyzing customer churn risk. Analyze this customer:

    CUSTOMER INFORMATION:
    Name: {surname}
    Number of Products: {input_dict['NumOfProducts']}
    Active Member: {'Yes' if input_dict['IsActiveMember'] else 'No'}
    Age: {input_dict['Age']}
    Location: {'Germany' if input_dict['Geography_Germany'] else 'Spain' if input_dict['Geography_Spain'] else 'France'}
    Balance: ${input_dict['Balance']:,.2f}
    Credit Score: {input_dict['CreditScore']}
    Tenure: {input_dict['Tenure']} years
     


    KEY FACTORS AFFECTING CHURN (in order of importance):
    1. Number of Products (33% impact)
    2. Active Membership (20% impact)
    3. Age (11% impact)
    4. Location (8% impact)
    5. Balance (5% impact)
    6. Geography_Spain (4% impact)
    7. Gender_Male (4% impact)
    8. Credit Score (3% impact) 
    9. Estimated Salary (3% impact)
    10. Tenure (3% impact)          
    11. Has Credit Card (3% impact)


    {'HIGH RISK ANALYSIS NEEDED - Explain why this customer is likely to leave' if probability > 0.4 else 'LOW RISK ANALYSIS NEEDED - Explain why this customer is likely to stay in'}


        Please provide your analysis in the following bullet point format:

   • Main Risk Factor: [Explain the primary factor driving the prediction]
   • Secondary Factors: [List 2-3 contributing factors]
   • Customer Profile Impact: [How the customer's profile compares to typical patterns]
   • Recommendation: [Specific action for the bank]

    Do not mention probabilities or models in your response.
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.2-3b-preview",
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating explanation: {str(e)}")
        return "Unable to generate explanation at this time."

  
def generate_email(probability, input_dict, explanation, surname): 
  
  # Define a threshold for high churn probability
  churn_threshold = 0.4


  prompt = f"""
  You are a Manager of customer service representative at a bank, and you have been asked to reach out to a customer named {surname}. 
  Your goal is to encourage the customer to remain loyal to the bank by providing tailored offers and highlighting the benefits of continuing their relationship with HB Bank.


  Customer details:
        - Name: {surname}
        - Churn Risk: {round(probability*100, 1)}% (For internal use only)
        - Customer Profile: {input_dict}

  The machine learning model has predicted that this customer has a {round(probability * 100, 1)}% probability of churning. Here is the explanation provided by the data scientist: {explanation}

        Generate a personalized email for the customer:
- If the customer is at risk of churning, provide specific incentives to stay, focused on their unique needs and preferences.
- If the customer is not at risk of churning, encourage further engagement with loyalty rewards and personalized benefits.

        The email should:
        - Avoid mentioning anything about churn probability, machine learning models, or predictive analytics.
        - Be friendly and customer-focused, emphasizing how HB Bank can support their financial goals.
        - List incentives in bullet point format based on the customer's information, such as age, tenure, estimated salary, or products they hold with the bank.
        
Here's the email structure:

---

Dear {surname},

We value you as part of our HB Bank community and want to make sure you get the best out of your relationship with us.

To further support your goals, we'd like to offer a few exclusive incentives just for you:

- [List out incentives based on the customer's information, such as tailored financial products, loyalty rewards, or premium services.]

We're here to make your banking experience even better. If there's anything more we can do to support your needs, please reach out anytime.

Warm regards,  
HB Bank Customer Care Team  

  """

  response = client.chat.completions.create(
    model = "llama-3.2-3b-preview",
    messages = [{
      "role" : "user",
      "content" : prompt
    }],
  )

  return response.choices[0].message.content


#-----------------------------------
# to load the pickle machine learning model file
def load_model(filename):
  with open(filename, "rb") as file:
    return pickle.load(file)


xgboost_model = load_model("xgb_model.pkl")
# catboost_model = load_model("cat_model.pkl")
# naive_bayes_model = load_model("nb_model.pkl")
knn_model = load_model("knn_model.pkl")
# decision_tree_model = load_model("dt_model.pkl")
random_forest_model = load_model("rf_model.pkl")
# voting_classifier_model = load_model("voting_model.pkl")
# xgboost_SMOTE_model = load_model("xgb_smote.pkl")
# xgboost_featureEngineered_model = load_model("xgb_featureEngineered.pkl")
# logistic_model = load_model("logistic_model.pkl")
# gradient_boost_model = load_model("gb_model.pkl")


def prepare_input(credit_score, location, gender, age, tenure, balance,
                  num_products, has_credit_card, is_active_member,
                  estimated_salary):
  input_dict = {
      "CreditScore": credit_score,
      "Age": age,
      "Tenure": tenure,
      "Balance": balance,
      "NumOfProducts": num_products,
      "HasCreditCard": int(has_credit_card),
      "IsActiveMember": int(is_active_member),
      "EstimatedSalary": estimated_salary,
      "Geography_Germany": 1 if location == "Germany" else 0,
      "Geography_Spain": 1 if location == "Spain" else 0,
      "Gender_Male": 1 if gender == "Male" else 0,
  }
  input_df = pd.DataFrame([input_dict])
  return input_df, input_dict


def make_predictions(input_df, input_dict):
  probabilities = {
      "XGBoost": xgboost_model.predict_proba(input_df)[0][1],
      "Random Forest": random_forest_model.predict_proba(input_df)[0][1],
      "K-Nearest Neighbors": knn_model.predict_proba(input_df)[0][1],
      # "Catboost": catboost_model.predict_proba(input_df)[0][1],
  }

  avg_probability = np.mean(list(probabilities.values()))

  col1, col2 = st.columns(2)
  with col1:
    fig= ut.create_gauge_chart(avg_probability)
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"The customer has average {avg_probability*100:.2f}% probability of churning.")

  with col2:
    fig_probs =   ut.create_model_probability_chart(probabilities)
    st.plotly_chart(fig_probs, use_container_width=True)



  if avg_probability < 0.4:
        churn_text = "The customer is less likely to churn."
        bg_color = "#d4edda"  # light green
        text_color = "#155724"  # dark green 
  elif avg_probability < 0.6:
        churn_text = "The customer has a 50-50 probability of churning."
        bg_color = "#fff3cd"  # light yellow
        text_color = "#856404"  # dark yellow/brown
  else:
        churn_text = "The customer is most likely to churn."
        bg_color = "#f8d7da"  # light red
        text_color = "#721c24"  # dark red

  st.markdown(
        f"""
        <div style="
            background-color: {bg_color};
            color: {text_color};
            padding: 20px;
            border-radius: 8px;
            font-size: 18px;
            text-align: center;
            margin-top: 20px;
            width: 100%;
        ">
            {churn_text}
        </div>
        """,
        unsafe_allow_html=True)



#   st.markdown("### Model Probabilities")
#   for model, prob in probabilities.items():
#     st.write(f"{model}: {prob}")
#   st.write(f"Average Probability: {avg_probability}")



  return avg_probability



#---------------------------------------
# interface
st.title("Customer Churn Prediction")
st.header("Detailed Analysis")
df = pd.read_csv("churn.csv")

customers = [
    f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()
]

selected_customer_option = st.selectbox("Select a customer: ", customers)

if selected_customer_option:
  selected_customer_id = int(selected_customer_option.split(" - ")[0])
  selected_surname = selected_customer_option.split(" - ")[1]

  print("Selected Customer ID", selected_customer_id)
  print(selected_surname)
  selected_customer = df.loc[df["CustomerId"] == selected_customer_id].iloc[0]

  print(selected_customer)
  col1, col2 = st.columns(2)

  with col1:
    credit_score = st.number_input("Credit Score",
                                   min_value=300,
                                   max_value=850,
                                   value=int(selected_customer["CreditScore"]))

    location = st.selectbox("Location", ["Spain", "France", "Germany"],
                            index=["Spain", "France", "Germany"
                                   ].index(selected_customer["Geography"]))

    gender = st.radio("Gender", ["Male", "Female"],
                      index=0 if selected_customer["Gender"] == "Male" else 1)

    age = st.number_input("Age",
                          min_value=18,
                          max_value=100,
                          value=int(selected_customer["Age"]))

    tenure = st.number_input("Tenure (years)",
                             min_value=0,
                             max_value=50,
                             value=int(selected_customer["Tenure"]))

  with col2:
    balance = st.number_input("Balance",
                              min_value=0.0,
                              value=float(selected_customer["Balance"]))


    num_products = st.number_input("Number of Products",
                                   min_value=0,
                                   max_value=19,
                                   value=selected_customer["NumOfProducts"])

    has_credit_card = st.checkbox("Has Credit Card",
                                  value=bool(selected_customer["HasCrCard"]))

    is_active_member = st.checkbox("Is Active Member",
                                   value=bool(
                                       selected_customer["IsActiveMember"]))

    estimated_salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=float(selected_customer["EstimatedSalary"]))

  input_df, input_dict = prepare_input(credit_score, location, gender, age,
                                       tenure, balance, num_products,
                                       has_credit_card, is_active_member,
                                       estimated_salary)

  st.markdown("""
      <style>
      .center-button {
          display: flex;
          justify-content: center;
          padding-top: 20px;
      }
      .stButton>button {
          color: white;
          background-color: green;
          border-radius: 5px;
          width: 100%;
          height: 50px;
      }
      </style>
      """,
              unsafe_allow_html=True)



  if st.button("Predict"):
    # Run the prediction function if the button is clicked
    avg_probability = make_predictions(input_df, input_dict)

    try:
      explanation = explain_predictionss(avg_probability, input_dict,
                                        selected_customer["Surname"], df)
      st.markdown("---")
      st.subheader("Explanation of Prediction")
      st.markdown(explanation)

      email = generate_email(avg_probability, input_dict, explanation,
                           selected_customer["Surname"])
      st.markdown("---")
      st.subheader("Email to Customer")
      st.markdown(email)

    except Exception as e:
      st.error(f"An error occurred while generating the explanation: {e}")
      st.markdown(explanation)

      

