import streamlit.components.v1 as components
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# 加载模型和测试数据
model = joblib.load('APP.pkl')
X_test = pd.read_csv("x_test_app.csv")

# 定义特征名称
feature_names = ["AGE", "SEX", "SMOKE", "HP", "COPD", "DM", "CHD",
                 "VATS", "PATHO","ST","BLOOD","TNM","BMI", "FEV1","DLCO",
                 "肺段","P-FEV1%", "P-DLCO%", "FEV1-DLCO一起","危险因素个数"]

# 创建 Streamlit 用户界面
st.title("Lung Cancer Predictor")
AGE = st.number_input("AGE:", min_value=0, max_value=120, value=41)
SEX = st.selectbox("SEX:", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
SMOKE= st.selectbox("SMOKE:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
HP= st.selectbox("HP:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
COPD= st.selectbox("COPD:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
DM= st.selectbox("DM:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
CHD= st.selectbox("CHD:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
VATS= st.selectbox("VATS:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
PATHO= st.selectbox("PATHO:", options=[1, 2,3,4])
ST = st.number_input("ST:", min_value=0, max_value=400)
BLOOD = st.number_input("BLOOD:", min_value=0, max_value=2000)
TNM= st.selectbox("TNM:", options=[1, 2,3,4])
BMI = st.number_input("BMI:", min_value=0, max_value=50)
FEV1 = st.number_input("FEV1:", min_value=0, max_value=200)
DLCO = st.number_input("DLCO:", min_value=0, max_value=1000)
肺段= st.selectbox("肺段:", options=[3,4,5,6])
P_FEV1_percent = st.number_input("P-FEV1%:", min_value=0, max_value=100)
P_DLCO_percent = st.number_input("P-DLCO%:", min_value=0, max_value=100)
FEV1_DLCO一起= st.selectbox("FEV1-DLCO一起:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
危险因素个数= st.selectbox("危险因素个数:", options=[0,1,2,3,4,5])
# 收集所有输入特征
feature_values = [AGE, SEX, SMOKE, HP, COPD, DM, CHD, VATS, PATHO, ST, BLOOD, TNM, BMI, FEV1, DLCO, 肺段, P_FEV1_percent, P_DLCO_percent, FEV1_DLCO一起, 危险因素个数]
features = np.array([feature_values])

# 进行预测
predicted_class = model.predict(features)[0]
predicted_proba = model.predict_proba(features)[0]

# 显示预测结果
st.write(f"**Predicted Class:** {predicted_class} (1: Disease, 0: No Disease)")
st.write(f"**Prediction Probabilities:** {predicted_proba}")

# 显示建议
predicted_class = int(predicted_class)
probability = predicted_proba[predicted_class] * 100
if predicted_class == 1:
    advice = f"According to our model, you have a high risk of lung cancer. The model predicts that your probability of having lung cancer is {probability:.1f}%. It's advised to consult with your healthcare provider for further evaluation and possible intervention."
else:
    advice = f"According to our model, you have a low risk of lung cancer. The model predicts that your probability of not having lung cancer is {probability:.1f}%. However, maintaining a healthy lifestyle is important. Please continue regular check-ups with your healthcare provider."
st.write(advice)

# 显示 SHAP 解释
st.subheader("SHAP Force Plot Explanation")
explainer_shap = shap.TreeExplainer(model)
shap_values = explainer_shap.shap_values(pd.DataFrame([feature_values], columns=feature_names))
if predicted_class == 1:
    shap.force_plot(explainer_shap.expected_value[1], shap_values[:,:,1], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
else:
    shap.force_plot(explainer_shap.expected_value[0], shap_values[:,:,0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
st.image("shap_force_plot.png", caption='SHAP Force Plot Explanation')
