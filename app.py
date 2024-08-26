import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import xgboost as xgb

#st.markdown("### 本地图片示例")
#创建两列布局
left_column, col1, col2, col3, right_column = st.columns(5)

# 在左侧列中添加其他内容
left_column.write("")

# 在右侧列中显示图像
right_column.image('F:\\model\\jdy\\logo.png', caption='', width=100)

# Title
st.write("<p style='font-size: 20px;'>Machine Learning Models Can Identify Patients at High-Risk of Coronary Heart Disease and with Severe Disease using an Immunoglobulin Light Chain-Based Index</p>", unsafe_allow_html=True)

# Input bar 1
a = st.number_input('CHI', min_value=0.00, max_value=1.00, value=0.52)
b = st.number_input('Age', min_value=0, max_value=100, value=61)
c = st.number_input('Troponin', min_value=0.00, max_value=210.00, value=3.54)
d = st.number_input('BNP', min_value=0.00, max_value=40000.00, value=90.49)
e = st.number_input('HDL', min_value=0.00, max_value=10.00, value=1.09)
f = st.number_input('TC', min_value=0.00, max_value=20.00, value=4.55)
g = st.number_input('LYM#', min_value=0.00, max_value=10.00, value=1.78)
h = st.number_input('NEU#', min_value=0.00, max_value=30.00, value=4.63)
i = st.number_input('NLR', min_value=0.00, max_value=100.00, value=2.59)
j = st.number_input('LYM%', min_value=0.00, max_value=1.00, value=0.25)
k = st.number_input('Myoglobin', min_value=0.00, max_value=500.00, value=91.19)
l = st.number_input('CK-MB', min_value=0.00, max_value=100.00, value=12.70)

# If button is pressed
if st.button("Submit"):
    
    # Unpickle classifier    
    dirs = 'F:\\model\\jdy'
    mm = joblib.load(dirs + '/XGBoost.pkl')

    # Store inputs into dataframe
    X = pd.DataFrame([[a, b, c, d, e, f, g, h, i, j, k, l]], 
                     columns=["CHI", "Age", "Troponin", "BNP", "HDL", "TC", "LYM#", "NEU#", "NLR", "LYM%", "Myoglobin", "CK-MB"])
    
    # Get prediction
    for index, row in X.iterrows():
        data1 = row.to_frame()
        data2 = pd.DataFrame(data1.values.T, columns=data1.index)
        result111 = mm.predict(data2)
        result222 = str(result111).replace("[", "")
        result = str(result222).replace("]", "")  # 预测结果
        result333 = mm.predict_proba(data2)
        result444 = str(result333).replace("[[", "")
        result555 = str(result444).replace("]]", "")
        strlist = result555.split(' ')
        result_prob_neg = round(float(strlist[0]) * 100, 2)
        if len(strlist[1]) == 0:
            result_prob_pos = 'The conditions do not match and cannot be predicted'
        else:
            result_prob_pos = round(float(strlist[1]) * 100, 2)  # 预测概率        

    explainer = shap.TreeExplainer(mm) 
    shap_values = explainer.shap_values(data2)
    shap_values = shap_values.reshape((1, -1)) 

    # Output prediction
    st.text(f"The probability of high-risk of Coronary Heart Disease is: {str(result_prob_pos)}%")
    # st.text({str(shap_values[0])})

# Footer
st.write("<p style='font-size: 12px;'>Disclaimer: This mini app is designed to provide general information and is not a substitute for professional medical advice or diagnosis. Always consult with a qualified healthcare professional if you have any concerns about your health.</p>", unsafe_allow_html=True)
st.markdown('<div style="font-size: 12px; text-align: right;">Powered by MyLab+ i-Research Consulting Team</div>', unsafe_allow_html=True)
