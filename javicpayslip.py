# get streamlit 

#streamlitrun app.py   IN CONSOLE

from typing import Any
import streamlit as st 
import pandas as pd 
import altair as alt
import numpy as np
#import matplotlib.pyplot as plt
#import plotly.figure_factory as ff

import os
#import seaborn as sns
#import cufflinks as cf
import warnings
#import cufflinks as cf
#import plotly.express as px 
#import plotly.graph_objects as go
import requests
import io  


#import pickle

########################### Display text ###########################################

Sal_info1 = 'https://raw.githubusercontent.com/ongalajacob/Javic_ML/main/2021_data/javicjun_schms_table_sal_basicsalary.csv'
Sal_Montly = 'https://raw.githubusercontent.com/ongalajacob/Javic_ML/main/2021_data/javicjun_schms_table_sal_monthly.csv'
job_description = 'https://raw.githubusercontent.com/ongalajacob/Javic_ML/main/2021_data/javicjun_schms_table_job_description.csv'
employees = 'https://raw.githubusercontent.com/ongalajacob/Javic_ML/main/2021_data/javicjun_schms_table_employees.csv'


def main():
    Sal_info_df = pd.read_csv(Sal_info1,dtype={'Year' : 'category'})
    Sal_Montly_df = pd.read_csv(Sal_Montly,dtype={'Month' : 'category'})
    job_description_df = pd.read_csv(job_description)
    employees_df = pd.read_csv(employees,dtype={'National_ID' : 'category','Staff_ID' : 'category','PHONE1' : 'category','PHONE2' : 'category'})

    

    employee = pd.merge(left=employees_df, right=job_description_df, how='left', left_on='JOB', right_on='Job_ID')
    employee = employee[['id', 'name', 'Staff_ID', 'National_ID','NSSF', 'NHIF', 'PIN', 'PHONE1', 'PHONE2', 'Statuus', 'Title']]
    Sal_info = pd.merge(left= Sal_info_df , right=employee, how='left', left_on='StaffID', right_on='id')
    Sal_info.rename(columns = {'id_x':'id', }, inplace = True)
    Sal_info = Sal_info[['id', 'Year', 'Staff_ID',  'name','National_ID', 'Title', 'NSSF', 'NHIF', 'PIN', 'PHONE1', 'BasicSalary', 'onNHIF', 'onNSSF',
        'BankName', 'BankBranch', 'BankAcc', 'BIC']][Sal_info.Statuus !='exits']
    Sal_info.sort_values(by=['id'], inplace=True, ascending=False)
    Monthly_sal = pd.merge(left= Sal_Montly_df , right=Sal_info, how='left', left_on='SalaryID', right_on='id')
    Monthly_sal.rename(columns = {'id_x':'id', }, inplace = True)
    Monthly_sal['Gross_Sal'] =Monthly_sal['BasicSalary']+Monthly_sal['Responsibility']+Monthly_sal['Overtime']+Monthly_sal['Special']+ Monthly_sal['OtherAllowaAmnt']
    Monthly_sal['Absent_wot_pay'] =(Monthly_sal['NoOfDaysAbsent']* Monthly_sal['BasicSalary']/30) 
    Monthly_sal['Deductions'] =Monthly_sal['Absent_wot_pay'] +Monthly_sal['SalAdvance']+Monthly_sal['OtherDecAmount']
    Monthly_sal['Net_Sal'] =Monthly_sal['Gross_Sal']- Monthly_sal['Deductions'] 

    Monthly_sal=Monthly_sal[['id',  'Year', 'Month',   'Staff_ID', 'name', 'National_ID', 'Title','BasicSalary', 'Absent_wot_pay',
                        'Responsibility', 'Overtime', 'Special', 'OtherAllow', 'OtherAllowaAmnt','Gross_Sal',  'NoOfDaysAbsent', 'SalAdvance',
        'OtherDeductions', 'OtherDecAmount','Deductions','Net_Sal', 'BankName', 'BankBranch', 'BankAcc', 'BIC', 'NSSF', 'NHIF', 'PIN', 'PHONE1','onNHIF', 'onNSSF' ]] 
    Monthly_sal.sort_values(by=['id'], inplace=True, ascending=False) 
        
    html_temp1 = """
    <div style="background-color:white;padding:1.5px">
    <h1 style="color:black;text-align:center;">JAVIC JUNIOR SCHOOL </h1>
    </div><br>"""

    html_temp2 = """
    <div style="background-color:white;padding:1.5px">
    <h3 style="color:black;text-align:center;">e-payslip </h3>
    </div><br>"""
    st.markdown(html_temp1,unsafe_allow_html=True)
    _,_,_, col2, _,_,_ = st.beta_columns([1,1,1,2,1,1,1])
    #with col2:
    #st.image(im, width=150)

    st.markdown(html_temp2,unsafe_allow_html=True)
    #st.title('This is for a good design')
    st.markdown('<style>h1{color: red;}</style>', unsafe_allow_html=True)


    Infos="""
    enter your ID Number to view your payslip
    """
    st.markdown(Infos)

    selection = st.sidebar.selectbox('Select option to view:', 
        ('My Payslip', 'General School Information'))
    if selection == 'My Payslip':
        NatID =st.text_input(label='Enter your National ID Number')
        if NatID=="":
            st.write('Enter your ID number Number  in the space above to continue')
        if NatID !="":
            st.write('These are the kids registered with the phone number provided')
            employee.rename(columns = {'name':'Name','Staff_ID':'Staff ID' ,'National_ID':'ID No.' }, inplace = True)
            employee= employee[['Name', 'Staff ID', 'ID No.','NSSF', 'NHIF', 'PIN', 'PHONE1', 'Title']]
            st.dataframe(employee[employee['ID No.']==NatID ].T)
            st.markdown("<h5 style='text-align: center; color: blue;'>--------------------------------------------------------------------------------------------------------------------</h5>", unsafe_allow_html=True)
            st.markdown("<h5 style='text-align: center; color: red;'>If you can't see your details, contact the manager</h5>", unsafe_allow_html=True)
            st.markdown("<h5 style='text-align: center; color: blue;'>--------------------------------------------------------------------------------------------------------------------</h5>", unsafe_allow_html=True)
            
            
            if st.checkbox("Show Payslip Status"):
                yr = st.selectbox("Select Month",options=['2020' , '2021', '2022', '2023', '2024', '2025', '2026', '2027', '2028', '2029', '2030'])
                mnt = st.selectbox("Select Month",options=['Jan' , 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
                Monthly_sal=Monthly_sal[((Monthly_sal.Year==yr) & (Monthly_sal.Month ==mnt))]
                Monthly_sal=Monthly_sal[Monthly_sal['National_ID']==NatID ]
                Monthly_sal= Monthly_sal[['BasicSalary', 'Responsibility', 'Overtime', 'Special', 'OtherAllow',  'OtherAllowaAmnt', 'Gross_Sal', 'NoOfDaysAbsent', 'Absent_wot_pay','SalAdvance',
                'OtherDeductions', 'OtherDecAmount', 'Deductions', 'Net_Sal']]
                Earnings=  Monthly_sal[['BasicSalary', 'Responsibility', 'Overtime', 'Special', 'OtherAllow',  'OtherAllowaAmnt', 'Gross_Sal']]
                Deductions=  Monthly_sal[[ 'NoOfDaysAbsent', 'Absent_wot_pay','SalAdvance', 'OtherDeductions', 'OtherDecAmount', 'Deductions']]
                
                Earnings.rename(columns = {'BasicSalary':'Basic Salary','OtherAllow':'Other Allow Stated' ,'Gross_Sal':'Gross Salary' }, inplace = True)
                Deductions.rename(columns = {'NoOfDaysAbsent':'No of Days Absent','Absent_wot_pay':'Absent without pay deductions ' ,'SalAdvance':'Salary Advance','OtherDeductions':'Other Deductions Stated' ,'Deductions':'Total Deductions'}, inplace = True)
           
                st.dataframe(Earnings.T)
                st.dataframe(Monthly_sal.Gross_Sal)
                st.dataframe(Deductions.T)
                st.dataframe(Monthly_sal.Net_Sal)
                
                
    elif selection == 'General School Information':
        st.title("Nothing to view yet")
        

        

        
            
if __name__ =='__main__':
    main() 


#@st.cache

#st.balloons()


