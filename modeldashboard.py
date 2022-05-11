import streamlit as st
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from lmfit.models import ExponentialModel, GaussianModel,PolynomialModel,RectangleModel,SkewedGaussianModel,PowerLawModel,SplitLorentzianModel

# progress_bar = st.sidebar.progress(0)
# status_text = st.sidebar.empty()
# last_rows = np.random.randn(1, 1)
# chart = st.line_chart(last_rows)
def fitting_prog(s1):
    peaks = findPeaks(s1[s1.index<6000])
    
#     x=np.array(s1.index)
    x=np.array(s1[s1.index<6000].index)
    y=np.array(s1[s1.index<6000])
    pol_mod = PolynomialModel(degree=1,prefix='pol_')
    skew_gauss=RectangleModel(prefix='skg_',form='logistic')
    names = ['40S','60S','80S']
    pars = pol_mod.guess(y, x=x)
    pars.update(skew_gauss.make_params())
    pars[f'pol_c1'].set(value=-0.01, min=-2,max=0)
    pars['skg_center1'].set(value=peaks['peak'][0]-200,min=peaks['peak'][0]-700,max=peaks['peak'][0]+100)
    pars['skg_center2'].set(value=peaks['peak'][0]+200,min=peaks['peak'][0]-100,max=peaks['peak'][0]+700)
    # pars['skg_sigma1'].set(value=15, min=200,max=400)
#     print(pars)
    mod=pol_mod+skew_gauss
    for i in range(len(peaks)-2):
        gauss1 = SplitLorentzianModel(prefix=f'g{i}_')
        pars.update(gauss1.make_params())
        pars[f'g{i}_center'].set(value=peaks['peak'][i+1],min=peaks['peak'][i+1]-10,max=peaks['peak'][i+1]+10)
        pars[f'g{i}_sigma'].set(value=15, min=3,max=130)
        pars[f'g{i}_sigma_r'].set(value=15, min=3,max=150)
        pars[f'g{i}_amplitude'].set(value=1, min=0.2)
        mod+=gauss1
    init = mod.eval(pars, x=x)
    out = mod.fit(y, pars, x=x)
    # fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    # axes[0].plot(x, y)
    # axes[0].plot(x, init, '--', label='initial fit')
    # axes[0].plot(x, out.best_fit, '-', label='best fit')
    # axes[0].legend()
    fig=plt.figure(figsize=(12,10))

    # axes[1].plot(x, y)
    plt.plot(s1.index, s1)
    plt.plot(x, out.best_fit, '-', label='best fit')
    comps = out.eval_components(x=x)
    plt.plot(x, comps['skg_'], '--', label=f'logistic rectangular component')
    c1=out.params['pol_c0']
    data={'40S':0,'60S':0,'80S':0,'Polysome1':0,'Polysome2':0,'Polysome3':0,'Polysome4':0}
    for i in range(len(peaks)-2):
#         print(f"SplitLorentzian_{i} intergral :", comps[f'g{i}_'].sum())
        if(i<3):
            plt.plot(x, comps[f'g{i}_'], '--', label=f'{names[i]} component')
            plt.text(peaks['peak'][i+1],out.params[f'g{i}_height'].value+c1+0.05,names[i],rotation=45,fontsize=20)
            # print(f"{names[i]} intergral :", comps[f'g{i}_'].sum())
            data[f'{names[i]}']=comps[f'g{i}_'].sum()
        else:
            plt.text(peaks['peak'][i+1],out.params[f'g{i}_height'].value+c1+0.05,f"polysome {i-2}",rotation=45,fontsize=20)
            # print(f"Polysome{i-2} intergral :", comps[f'g{i}_'].sum())
            plt.plot(x, comps[f'g{i}_'], '--', label=f'Polysome{i-2} component')
            data[f'Polysome{i-2}']=comps[f'g{i}_'].sum()
    plt.plot(x, comps['pol_'], '--', label='Quadratic component')
    
    plt.ylim(-0.1,1.3)
    plt.legend(fontsize=12)
    st.pyplot(fig)
    return data
def scatter_plot(df):
    st.markdown(f"# Coronavirus simulation data plot")
    col1,col2,col3,col4=st.columns(4)
    with col1:
        labelfont=st.selectbox("Select label font size",range(15,50))
    with col2:
        ticksfont=st.selectbox("ticks label font size",range(15,50))
    with col3:
        legendfont=st.selectbox("Legend font size",range(15,50))
    with col4:
        log=st.selectbox("Set yaxis scale Log",[False,True])
    X=st.sidebar.selectbox('Select x axis',['Infectiontime','runtime'])
    Y=st.sidebar.selectbox('Column Name',list(df.columns)[1:])
    
    fig = plt.figure(figsize = (10, 5))
    plt.plot(df[X],df[Y],label=Y)
    plt.xlabel(X,fontsize=labelfont)
    plt.ylabel("Counts",fontsize=labelfont)
    plt.xticks(fontsize=ticksfont)
    plt.yticks(fontsize=ticksfont)
    plt.legend(fontsize=legendfont)
    if log:
        plt.yscale('log')
    st.pyplot(fig)
    st.button("Re-run")
def Multi_plot(df):
    X=st.sidebar.selectbox('Select x axis',['Infectiontime','runtime'])
    if X=="runtime":
        z=st.sidebar.select_slider("Select x axis range",list(df[X]),value=[df[X].min(),df[X].max()])
    else:
        z=st.sidebar.select_slider("Select x axis range",sorted(set(df[X])),value=[df[X].min(),df[X].max()])
    Y=st.sidebar.multiselect('Column Name',list(df.columns),default=list(df.columns)[2:10])
    minIndex=df[df[X]==z[0]].index
    maxIndex=df[df[X]==z[1]].index
   
    st.markdown(f"# Coronavirus simulation data plot")
    col1,col2,col3,col4=st.columns(4)
    with col1:
        labelfont=st.selectbox("Select label font size",range(15,50))
    with col2:
        ticksfont=st.selectbox("ticks label font size",range(15,50))
    with col3:
        legendfont=st.selectbox("Legend font size",range(10,50))
    with col4:
        log=st.selectbox("Set yaxis scale log",[False,True])
    fig = plt.figure(figsize = (10, 8))
    plt.plot(df[X].iloc[minIndex[0]:maxIndex[-1]],df[Y].iloc[minIndex[0]:maxIndex[-1]],label=Y)
    plt.legend(fontsize=legendfont,bbox_to_anchor=(1.0,1.2))
    plt.xlabel(X,fontsize=labelfont)
    plt.ylabel("Counts",fontsize=labelfont)
    plt.xticks(fontsize=ticksfont)
    plt.yticks(fontsize=ticksfont)
    plt.xlim(z)
    if log:
        plt.yscale('log')
    st.pyplot(fig)
    st.button("Reload data")
def findPeaks(s=pd.Series(),prominence=0.005,width=50):
    peaks, properties = signal.find_peaks(s, prominence=prominence, width=width,distance=50)
    return pd.DataFrame(index=pd.Series(peaks,name='peak'),data=properties).reset_index()
@st.cache
def convert_df(df):
   return df.to_csv().encode('utf-8')
def main():
    
    uploaded_files =st.file_uploader("upload data file",accept_multiple_files=True)
    df_out=pd.DataFrame(columns=['40S','60S','80S','Polysome1','Polysome2','Polysome3','Polysome4'])
    df_out=df_out.T
    if st.sidebar.button('Run test file'):
            df=pd.read_excel("test.xlsx",names=['vol','Abs','x'])
            s1=df['Abs']
            data=fitting_prog(s1)
            df_out["test.xlsx"]=pd.Series(data)
            st.markdown("# Dataframe Overview")
            st.dataframe(df_out,1000, 1000)
            csv = convert_df(df_out)
            st.download_button(
                "Press to Download",
                csv,
                "file.csv",
                "text/csv",
                key='download-csv'
            )
    else:
        st.sidebar.header('Uploaded files')
        if uploaded_files is not None:
            for uploaded_file in uploaded_files:
                st.write("Processing file: "+uploaded_file.name)
                st.sidebar.write(uploaded_file.name)
                df=pd.read_excel(uploaded_file,names=['vol','Abs','x'])
                s1=df['Abs']
                data=fitting_prog(s1)
                df_out[uploaded_file.name]=pd.Series(data)
            st.markdown("# Dataframe Overview")
            # z=st.sidebar.slider("Select rows",0,df_out.shape[0],value=[0,df_out.shape[0]])
            # st.markdown(f"slected range is {z}")
            # Y=st.sidebar.multiselect('Select Column',list(df_out.columns),default=list(df_out.columns)[0:1])
            st.dataframe(df_out,1000, 1000)
            csv = convert_df(df_out)

            st.download_button(
                    "Press to Download",
                    csv,
                    "file.csv",
                    "text/csv",
                    key='download-csv'
                )
        
            # st.pyplot(fig)
            # df[file_name]=pd.Series(data)
        # df=pd.read_pickle(uploaded_file)
        # page = st.sidebar.selectbox(
        #     "Select a Page",
        #     [
        #         "Single Plot",
        #         "Multi Plot",
        #         "DataFrame"
        #     ]
        # )
    
        # if page=="Single Plot":
        #     scatter_plot(df)
        # if page=="Multi Plot":
        #     Multi_plot(df)
        # if page=="DataFrame":
        #     DataFrame(df)
if __name__ == "__main__":
    main()
# plt.plot(df['Infectiontime'],df[selected_column])
# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
   
