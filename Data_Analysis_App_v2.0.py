import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
import customtkinter as ctk
from tkinter import filedialog, messagebox
from CTkListbox import CTkListbox
from customtkinter import CTkFrame, CTkButton, CTkTextbox, CTkLabel, CTkComboBox, CTkCheckBox,CTkEntry,CTkInputDialog, CTkScrollableFrame
import pandas as pd
import os, io
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay,RocCurveDisplay,classification_report
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import streamlit as st

root = tk.Tk()
root.title("Data Analysis App v2.0 by RAY")
root.geometry("1200x600")
root.configure(background="#1a2a3c")
ctk.set_widget_scaling(1.0)  # Sets font and widget size
ctk.set_window_scaling(1.0)  # Sets window geometry size
tab_control = ttk.Notebook(root)
tab1= CTkFrame(tab_control)
tab2 = CTkFrame(tab_control)
tab3 = CTkFrame(tab_control)
tab4 = CTkFrame(tab_control)
tab5 = CTkFrame(tab_control)
tab6= CTkFrame(tab_control)
tab7 = CTkFrame(tab_control)
tab8 = CTkFrame(tab_control)
tab9= CTkFrame(tab_control)
tab10= CTkFrame(tab_control)
tab_control.add(tab1, text='EDA')
tab_control.add(tab2, text='Plotting')
tab_control.add(tab3, text='Encoding')
tab_control.add(tab4, text='Training')
tab_control.add(tab5, text='Linear Regression')
tab_control.add(tab6, text='Logistic Regression')
tab_control.add(tab7, text='RandomForest')
tab_control.add(tab8, text='KNN')
tab_control.add(tab9, text='SVM')
tab_control.add(tab10, text='ML Comparison')
tab_control.pack(expand=True, fill="both")
#----------------------  Add content to Tab 1-----------------
tab1_frame1=CTkScrollableFrame(tab1)
tab1_frame1.pack(fill='both', side='left', padx='5', pady='5')
tab1_frame2=CTkFrame(tab1)
tab1_frame2.pack(fill='both', side='left', padx='5', pady='5', expand=True)
#------------------TEXT BOX- Tab-1----------------------------------
text_box = CTkTextbox(tab1_frame2, font=("Consolas", 16), wrap="none")
text_box.pack(fill = 'both', expand = True, padx=10, pady=10)
#----------------------  Add content to Tab 2-----------------
tab2_frame1=CTkFrame(tab2,height=50, width=1200)
tab2_frame1.pack(fill='both')
tab2_frame2=CTkFrame(tab2)
tab2_frame2.pack()
tab2_frame3=CTkScrollableFrame(tab2)
tab2_frame3.pack(fill='both',expand=True)
# -----------Add content to Tab 3-------------
tab3_frame1=CTkFrame(tab3)
tab3_frame1.pack(fill='both', side='left', padx='5', pady='5')
tab3_frame2=CTkScrollableFrame(tab3)
tab3_frame2.pack(fill='both', side='left', padx='5', pady='5',expand=True)
# -----------Add content to Tab 4-------------
tab4_frame1=CTkFrame(tab4,height=50)
tab4_frame1.pack(fill='both')
tab4_frame2=CTkScrollableFrame(tab4)
tab4_frame2.pack(fill='both',expand=True)
#------------Add content to Tab-5------------------------
tab5_frame0=CTkFrame(tab5,height=20, width=1200)
tab5_frame0.pack(fill='both', anchor='n', padx='5', pady='5')
tab5_frame1=CTkScrollableFrame(tab5)
tab5_frame1.pack(fill='both',padx='5', pady='5')
tab5_frame2=CTkScrollableFrame(tab5,height=100)
tab5_frame2.pack(fill='both',padx='5', pady='5', expand=True)
#------------Add content to Tab-6------------------------
tab6_frame0=CTkFrame(tab6,height=20, width=1200)
tab6_frame0.pack(fill='both', anchor='n', padx='5', pady='5')
tab6_frame1=CTkScrollableFrame(tab6)
tab6_frame1.pack(fill='both', padx='5', pady='5')
tab6_frame2=CTkScrollableFrame(tab6)
tab6_frame2.pack(fill='both',padx='5', pady='5', expand=True)
#------------Add content to Tab-7------------------------
tab7_frame0=CTkFrame(tab7,height=20, width=1200)
tab7_frame0.pack(fill='both', anchor='n', padx='5', pady='5')
tab7_frame1=CTkScrollableFrame(tab7)
tab7_frame1.pack(fill='both', padx='5', pady='5')
tab7_frame2=CTkScrollableFrame(tab7)
tab7_frame2.pack(fill='both', padx='5', pady='5', expand=True)
#------------Add content to Tab-8----------------------------
tab8_frame0=CTkFrame(tab8,height=20, width=1200)
tab8_frame0.pack(fill='both', anchor='n', padx='5', pady='5')
tab8_frame1=CTkScrollableFrame(tab8)
tab8_frame1.pack(fill='both', padx='5', pady='5')
tab8_frame2=CTkScrollableFrame(tab8)
tab8_frame2.pack(fill='both', padx='5', pady='5', expand=True)
#------------Add content to Tab-9-----------------------------
tab9_frame0=CTkFrame(tab9,height=20, width=1200)
tab9_frame0.pack(fill='both', anchor='n', padx='5', pady='5')
tab9_frame1=CTkScrollableFrame(tab9)
tab9_frame1.pack(fill='both', padx='5', pady='5')
tab9_frame2=CTkScrollableFrame(tab9)
tab9_frame2.pack(fill='both', padx='5', pady='5', expand=True)
#------------Add content to Tab-10-----------------------------
tab10_frame0=CTkFrame(tab10)
tab10_frame0.pack(fill='both', padx='5', pady='5')
tab10_frame1=CTkFrame(tab10)
tab10_frame1.pack(fill='both', padx='5', pady='5', expand=True)
#--------------------Clear Frame Function---------------------
def clear_frame(frame):
    for widget in frame.winfo_children():
        widget.destroy()


# ------------------EDA Functions-------------------------

def show_box(text):
    """EDA Tab Screen to show results of DataFrame and operations performed on df"""
    text_box.configure(state="normal")
    text_box.delete("1.0", "end")
    text_box.insert("end", text)
    text_box.configure(state="disabled")


def openfile():
    """open file from local drive"""
    global dataset, df, file_path
    global LRr2, log_acc, rf_clf_acc, knn_acc, svm_acc
    LRr2, log_acc, rf_clf_acc, knn_acc, svm_acc = None, None, None, None, None #Reset Accuracy vars for comparison
    
    

    # 1. Use st.file_uploader to accept the same file types
    uploaded_file = st.file_uploader(
        "Choose a file", 
        type=["csv", "xlsx", "xls"]
    )
    
    # 2. Process the file buffer if a file was uploaded
    if uploaded_file is not None:
        try:
            # Instead of checking a path string, we check the file's name property
            if uploaded_file.name.endswith('.csv'):
                dataset = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                dataset = pd.read_excel(uploaded_file)
            
            # Create your copy
            df = dataset.copy()
            
            # Display the data to confirm it loaded
            st.success(f"Successfully loaded: {uploaded_file.name}")
            st.dataframe(df.head())
            
        except Exception as e:
            st.error(f"Error loading file: {e}")
    else:
        st.info("Please upload a CSV or Excel file to proceed.")
        

openfile()


def column_options():
    """Generate df Column List to use inside comboBoxes"""
    global df, col_options, plot_col_options
    col_options = df.columns.tolist()
    plot_col_options = [""] + df.columns.tolist()


column_options()


def reopen_file():
    openfile()
    column_options()
    fill_null_combo_box.configure(values=col_options)
    fill_null_combo_box.set(col_options[0])
    rm_outlier_combo_box.configure(values=col_options)
    rm_outlier_combo_box.set(col_options[0])
    tab2_combo_box_x.configure(values=plot_col_options)
    tab2_combo_box_x.set(plot_col_options[0])
    tab2_combo_box_y.configure(values=plot_col_options)
    tab2_combo_box_y.set(plot_col_options[0])
    tab2_combo_box_hue.configure(values=plot_col_options)
    tab2_combo_box_hue.set(plot_col_options[0])


# -------------------------
# Text Info
# -------------------------
def dfhead_func():
    show_box(df.head().to_string())


def dfshape_func():
    show_box(df.shape)


def dfdescribe_func():
    show_box(df.describe().to_string())


def dfisnull_func():
    show_box(df.isnull().sum().to_string())


def duplicates_func():
    show_box(df.duplicated().sum())


def rem_duplicates_func():
    """Remove the duplicate rows from df"""
    global df
    try:
        dup_count = df.duplicated().sum()
        df = df.drop_duplicates()
        show_box(f"{dup_count} duplicates has been removed")
    except Exception as e:
        show_box(e)


def dffillna_func(event=None):
    """Fill Null Values by selecting column and value e.g. mean, mode, median and other"""
    global df
    fill_col = fill_null_combo_box.get()
    try:
        cb_var = tab1_combo_box.get()
        if cb_var == 'Mean':
            df[fill_col] = df[fill_col].fillna(df[fill_col].mean())
        elif cb_var == 'Mode':
            df[fill_col] = df[fill_col].fillna(df[fill_col].mode().iloc[0])
        elif cb_var == 'Median':
            df[fill_col] = df[fill_col].fillna(df[fill_col].median())
        else:
            df[fill_col] = df[fill_col].fillna("other")
        show_box(f"Null Values Filled Successfully with {cb_var} values")
    except TypeError:
        show_box(f"Type Error: Non-Numeric Columns can't filled with {cb_var} value")


def dfdropna_func():
    """Drop Null Values from selected column"""
    global df
    try:
        fill_col = fill_null_combo_box.get()
        df.dropna(subset=[fill_col], inplace=True, ignore_index=True)
        show_box(f"Null Rows Values of Column \"{fill_col}\" dropped")
    except Exception as e:
        show_box(e)


def drop_col_func():
    """Drop entire column if unnecessary for analysis"""
    global df, col_options, plot_col_options
    try:
        fill_col = fill_null_combo_box.get()
        df = df.drop(columns=[fill_col])
        show_box(f"Column {fill_col} dropped successfully")
        col_options = df.columns.tolist()
        plot_col_options = [""] + df.columns.tolist()
        fill_null_combo_box.configure(values=col_options)
        fill_null_combo_box.set(col_options[0])
        rm_outlier_combo_box.configure(values=col_options)
        rm_outlier_combo_box.set(col_options[0])
        tab2_combo_box_x.configure(values=plot_col_options)
        tab2_combo_box_x.set(plot_col_options[0])
        tab2_combo_box_y.configure(values=plot_col_options)
        tab2_combo_box_y.set(plot_col_options[0])
        tab2_combo_box_hue.configure(values=plot_col_options)
        tab2_combo_box_hue.set(plot_col_options[0])
    except Exception as e:
        show_box(e)


def dfinfo_func():
    """Information of df"""
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    show_box(info_str)


def correlation_func():
    """Show correlation of Numeric Columns of df"""
    global corr
    try:
        corr = df.corr(numeric_only=True)
        show_box(corr)
    except Exception as e:
        show_box(e)


def remove_outlier_func():
    """Remove outliers from df column-wise using threshold quantile value"""
    global df
    try:
        min_q_val = float(min_quantile_option.get())
        max_q_val = float(max_quantile_option.get())
        select_column = rm_outlier_combo_box.get()
        min_th = df[select_column].quantile(min_q_val)
        max_th = df[select_column].quantile(max_q_val)
        df = df[(df[select_column] > min_th) & (df[select_column] < max_th)]
        messagebox.showinfo("Outlier Remover", f"Outliers Removed from Column {select_column}")
    except Exception as e:
        show_box(e)


def custom_cmd_func():
    """user defined code to apply on df"""
    global df
    custom_cmd_dialog = ctk.CTkInputDialog(text="Write Custom Code:", title="User Code")
    user_code = custom_cmd_dialog.get_input()
    if user_code:
        try:
            show_box(user_code)
            exec(user_code, globals())
            column_options()
        except Exception as e:
            messagebox.showinfo("Error-Invalid Command", e)
    else:
        messagebox.showinfo("Custom Command", "No Code available")


def dtypes_func():
    show_box(df.dtypes.to_string())


def value_counts_func():
    fill_col = fill_null_combo_box.get()
    show_box(df[fill_col].value_counts())


def unique_values_func():
    fill_col = fill_null_combo_box.get()
    show_box(f"Unique values in {fill_col}:\n {df[fill_col].unique()}")


def to_csv_func():
    """Save the cleaned df to a file in local drive"""
    try:
        file_name = Path(file_path).stem
        file_name = file_name + "_cleaned.csv"
        df.to_csv(file_name, index=False, encoding='utf-8-sig')
        messagebox.showinfo("Save File", "File Saved in current working directory")
    except Exception as e:
        messagebox.showinfo("Save File", e)


def reset_df_func():
    """Reset dataframe to the original state when first time was loaded"""
    global df
    try:
        df = dataset.copy()
        messagebox.showinfo("Reset DF", "df reset successfully")
        column_options()
    except Exception as e:
        show_box(e)

#------------------Plot functions------------------------
def create_plot():
    """Generates line plots on the given axes."""
    clear_frame(tab2_frame3)
    global ax, canvas_widget, toolbar,t,s,clr
    try:
        t = tab2_combo_box_x.get()
        s = tab2_combo_box_y.get()
        clr = tab2_combo_box_clr.get()
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(t,s, color=clr ,data=df)
        ax.set_xlabel(t)
        ax.set_ylabel(s)
        ax.set_title(t+" Vs "+s)
        ax.grid(True)
        #------------------Matplotlib--------------------------------
        canvas_widget = FigureCanvasTkAgg(fig, master=tab2_frame3)# 4. Embed the Figure into a Tkinter Canvas Widget
        canvas_widget.draw()# 5. Draw the canvas and place the widget in the GUI using 'pack'
        canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(canvas_widget, tab2_frame3)# 6. (Optional) Add a standard navigation toolbar
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
    except Exception as e:
        messagebox.showinfo("Error",e)
def clear_plot():
    """Clears the axes data and redraws the canvas."""
    if canvas_widget.get_tk_widget():
        canvas_widget.get_tk_widget().destroy()
    if toolbar:
        toolbar.destroy() # This removes the Tkinter Frame and all its buttons

def create_barplot():
    """Generates barplot on the given axes."""
    clear_frame(tab2_frame3)
    global ax, canvas_widget, toolbar,t,s, clr,hue_var
    try:
        t = tab2_combo_box_x.get()
        s = tab2_combo_box_y.get()
        clr = tab2_combo_box_clr.get()
        hue_var = tab2_combo_box_hue.get()
        sns.set_palette(plot_colors)
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        sns.barplot(x=t,y=s, ax=ax, color=clr ,data=df, hue=hue_var if hue_var else None, edgecolor='white')
        ax.set_xlabel(t)
        ax.set_ylabel(s)
        ax.set_title(t+" Vs "+s)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        #------------------Matplotlib--------------------------------
        canvas_widget = FigureCanvasTkAgg(fig, master=tab2_frame3)# 4. Embed the Figure into a Tkinter Canvas Widget
        canvas_widget.draw()# 5. Draw the canvas and place the widget in the GUI using 'pack'
        canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(canvas_widget, tab2_frame3)# 6. (Optional) Add a standard navigation toolbar
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
    except Exception as e:
        messagebox.showinfo("Error",e)
def create_scatterplot():
    """Generates scatterplot on the given axes."""
    clear_frame(tab2_frame3)
    global ax, canvas_widget, toolbar,t,s,clr
    try:
        t = tab2_combo_box_x.get()
        s = tab2_combo_box_y.get()
        clr = tab2_combo_box_clr.get()
        hue_var = tab2_combo_box_hue.get()
        sns.set_palette(plot_colors)
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        # ax.scatter(t,s, color='blue',data=df, edgecolor='white')
        sns.scatterplot(data=df, x=t, y=s, hue=hue_var if hue_var else None, ax=ax, color = clr)
        ax.set_xlabel(t)
        ax.set_ylabel(s)
        ax.set_title(t+" Vs "+s)
        ax.grid(True)
        #------------------Matplotlib--------------------------------
        canvas_widget = FigureCanvasTkAgg(fig, master=tab2_frame3)# 4. Embed the Figure into a Tkinter Canvas Widget
        canvas_widget.draw()# 5. Draw the canvas and place the widget in the GUI using 'pack'
        canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(canvas_widget, tab2_frame3)# 6. (Optional) Add a standard navigation toolbar
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
    except Exception as e:
        messagebox.showinfo("Error",e)
def create_histplot():
    """Generates histplot on the given axes."""
    clear_frame(tab2_frame3)
    global ax, canvas_widget, toolbar,t,clr
    try:
        t = tab2_combo_box_x.get()
        clr = tab2_combo_box_clr.get()
        hue_var = tab2_combo_box_hue.get()
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        sns.histplot(ax = ax, x=t, color=clr ,data=df, edgecolor='Black',hue=hue_var if hue_var else None, multiple="stack")
        ax.set_xlabel(t)
        ax.set_title(t+" Histogram")
        ax.grid(True)
        #------------------Matplotlib--------------------------------
        canvas_widget = FigureCanvasTkAgg(fig, master=tab2_frame3)# 4. Embed the Figure into a Tkinter Canvas Widget
        canvas_widget.draw()# 5. Draw the canvas and place the widget in the GUI using 'pack'
        canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(canvas_widget, tab2_frame3)# 6. (Optional) Add a standard navigation toolbar
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
    except Exception as e:
        messagebox.showinfo("Error",e)
def create_pairplot():
    """Generates pairplot on the given axes."""
    clear_frame(tab2_frame3)
    global ax, canvas_widget, toolbar,t,s
    try:
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        pp=sns.pairplot(df)
        fig=pp.figure
        fig.suptitle("Pair Plot of Data", y=1.02)
        ax.grid(True)
        #------------------Matplotlib--------------------------------
        canvas_widget = FigureCanvasTkAgg(fig, master=tab2_frame3)# 4. Embed the Figure into a Tkinter Canvas Widget
        canvas_widget.draw()# 5. Draw the canvas and place the widget in the GUI using 'pack'
        canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(canvas_widget, tab2_frame3)# 6. (Optional) Add a standard navigation toolbar
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
    except Exception as e:
        messagebox.showinfo("Error",e)
def create_heatmap():
    """Generates heatmap on the given axes."""
    clear_frame(tab2_frame3)
    global ax, canvas_widget, toolbar,cmapvar
    try:
        cmapvar = tab2_combo_box_cmapclr.get()
        corr = df.corr(numeric_only=True)
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        sns.heatmap(corr, annot=True, fmt=".2f",cmap=cmapvar,ax=ax)
        ax.set_title("Correlation Heatmap")
        ax.grid(True)
        #------------------Canvas- Widget--------------------------------
        canvas_widget = FigureCanvasTkAgg(fig, master=tab2_frame3)# 4. Embed the Figure into a Tkinter Canvas Widget
        canvas_widget.draw()# 5. Draw the canvas and place the widget in the GUI using 'pack'
        canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(canvas_widget, tab2_frame3)# 6. (Optional) Add a standard navigation toolbar
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
    except Exception as e:
        messagebox.showinfo("Error",e)
def create_boxplot():
    """Generates boxplot on the given axes."""
    clear_frame(tab2_frame3)
    global ax, canvas_widget, toolbar,t
    try:
        t = tab2_combo_box_x.get()
        plt.rcParams['figure.figsize']=[8,4]
        plt.rcParams['figure.autolayout']=True
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        df[t].plot(kind='box', ax = ax, title = 'boxplot')
        ax.set_title("boxplot")
        ax.grid(True)
        #------------------Canvas- Widget--------------------------------
        canvas_widget = FigureCanvasTkAgg(fig, master=tab2_frame3)# 4. Embed the Figure into a Tkinter Canvas Widget
        canvas_widget.draw()# 5. Draw the canvas and place the widget in the GUI using 'pack'
        canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(canvas_widget, tab2_frame3)# 6. (Optional) Add a standard navigation toolbar
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
    except Exception as e:
        messagebox.showinfo("ERROR",f"Select X Value or Numeric Column only, {e}")

def piechart():
    """Generates piechart on the given axes."""
    clear_frame(tab2_frame3)
    global ax, canvas_widget, toolbar,t
    try:
        t = tab2_combo_box_x.get()
        counts = df[t].value_counts()
        counts.name = None
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        counts.plot(ax=ax, kind='pie', autopct='%1.1f%%', startangle=90)
        ax.set_title(f"Pie Chart of {t}")
        ax.grid(True)
        #------------------Canvas- Widget--------------------------------
        canvas_widget = FigureCanvasTkAgg(fig, master=tab2_frame3)# 4. Embed the Figure into a Tkinter Canvas Widget
        canvas_widget.draw()# 5. Draw the canvas and place the widget in the GUI using 'pack'
        canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(canvas_widget, tab2_frame3)# 6. (Optional) Add a standard navigation toolbar
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
    except Exception as e:
        messagebox.showinfo("ERROR",f"Select X Value :\n {e}")

def get_encode_columns():
    clear_frame(tab3_frame2)
    global encoding_vars
    encoding_vars = {item: tk.IntVar(value=0) for item in col_options}
    for item in col_options:
        cb_encoding = CTkCheckBox(tab3_frame2, text=item, variable=encoding_vars[item],onvalue=1, offvalue=0)
        cb_encoding .pack(anchor=tk.W) # Anchor to the west (left)

def encoding_select():
    """Reads the state of the checkboxes and prints the selected items."""
    global encoding_items
    try:
        encoding_items = []
        # Iterate through the dictionary to check the state of each variable
        for item_name, var_object in encoding_vars.items():
            if var_object.get() == 1:  # Check if the value is the 'onvalue' (default is 1)
                encoding_items.append(item_name)
    except Exception as e:
        messagebox.showinfo("Error",e)

def one_hot_encoder():
    """implement one-hot encoding on df columns"""
    global df
    try:
        encoding_select()
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded_data = encoder.fit_transform(df[encoding_items])# .fit_transform expects a 2D array,
        # 2. Convert the resulting NumPy array back to a DataFrame with meaningful names
        ohe_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(encoding_items))
        # 3. Combine with original data (optional)
        #categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        df= pd.concat([df, ohe_df], axis=1)
        messagebox.showinfo("One-Hot Encoding", "Successfully Encoded")
    except Exception as e:
        messagebox.showinfo("Error",e)

def label_encoder():
    """implement label encoding on df columns"""
    global df
    try:
        encoding_select()
        le = LabelEncoder()
        # 2. Apply and update original DataFrame
        df[encoding_items[0]] = le.fit_transform(df[encoding_items[0]])
        messagebox.showinfo("Label Encoder","LABEL ENCODED Successfully")
    except Exception as e:
        messagebox.showinfo("Error",e)

def get_dummies():
    """implement get dummies method on df columns"""
    global df
    try:
        encoding_select()
        df = pd.get_dummies(df, columns=encoding_items)
        messagebox.showinfo("Get Dummies","Successfully got dummies")
    except Exception as e:
        messagebox.showinfo("Error",e)


# ------------------Training Functions---------------------------------
def get_feature():
    """Reads the state of the checkboxes and store the selected items for ML Training"""
    global ohe_vars, target_combo_box_y, target_combo_box_yvar, tr_size_entry, rand_state_entry
    clear_frame(tab4_frame2)
    try:
        ohe_options = df.columns.tolist()
        feature_label_x = CTkLabel(tab4_frame2, text='X = ')
        feature_label_x.pack(anchor='nw', padx=10, pady=10)
        ohe_vars = {item: tk.IntVar(value=0) for item in ohe_options}
        for item in ohe_options:
            cb_training = CTkCheckBox(tab4_frame2, text=item, variable=ohe_vars[item], onvalue=1, offvalue=0)
            cb_training.pack(anchor=tk.W)  # Anchor to the west (left)
        # ----------Y-Target------------
        feature_label_y = CTkLabel(tab4_frame2, text='Y = ')
        feature_label_y.pack(side='left', padx=10, pady=10)
        target_combo_box_yvar = tk.StringVar()
        target_combo_box_y = ttk.Combobox(tab4_frame2, textvariable=target_combo_box_yvar, values=col_options,
                                          state="readonly")
        target_combo_box_y.current(0)  # Set default value
        target_combo_box_y.pack(side='left', padx=10, pady=10)
        feature_label_train_size = CTkLabel(tab4_frame2, text='Train Size = ')
        feature_label_train_size.pack(side='left', padx=10, pady=10)
        tr_size_entry = CTkEntry(tab4_frame2, placeholder_text="0.2")
        tr_size_entry.pack(side='left', padx='5', pady='2')
        feature_label_rand_state = CTkLabel(tab4_frame2, text='Random State = ')
        feature_label_rand_state.pack(side='left', padx=10, pady=10)
        rand_state_entry = CTkEntry(tab4_frame2, placeholder_text="42")
        rand_state_entry.pack(side='left', padx='5', pady='2')
    except Exception as e:
        messagebox.showinfo("Error", e)


def select_feature():
    """check the selected items from get features"""
    global X, y
    try:
        selected_items = []
        # Iterate through the dictionary to check the state of each variable
        for item_name, var_object in ohe_vars.items():
            if var_object.get() == 1:  # Check if the value is the 'onvalue' (default is 1)
                selected_items.append(item_name)
        X = df[selected_items]
        y = df[target_combo_box_y.get()]
    except Exception as e:
        messagebox.showinfo("Error", e)


def train_features():
    """perform train_test_split on features."""
    global X_train, X_test, y_train, y_test
    try:
        select_feature()
        tr_size = float(tr_size_entry.get() or "0.2")
        rand_state = int(rand_state_entry.get() or "42")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=tr_size, random_state=rand_state)
        messagebox.showinfo("Train Test Split", "Successfully Trained")
    except Exception as e:
        messagebox.showinfo("Error", e)


def standardscaler():
    """Apply StandardScaler function on X_train and X_test"""
    global X_train, X_test
    try:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        messagebox.showinfo("Standard Scaler", "Successfully Scaled")
    except Exception as e:
        messagebox.showinfo("Error", e)

def cls_textbox(text_box,frame): #clear screen Machine Learning Text box
    """Clear the Result screen of Machine Learning Tab"""
    try:
        content = text_box.get("1.0", "end-1c")
        if content.strip():
            text_box.configure(state="normal")
            text_box.delete("1.0", "end")
            text_box.configure(state="disabled")
            clear_frame(frame)
    except Exception as e:
        messagebox.showinfo("Error",e)
def show_box_Lin_Reg():
    """show text box in ML Tab for Linear Regression results"""
    try:
        message="Successfully Processed"
        text_box_lin_reg.configure(state="normal")
        text_box_lin_reg.delete("1.0", "end")
        text_box_lin_reg.insert("end", message)
        text_box_lin_reg.insert("end", "\nCoefficient   = ")
        text_box_lin_reg.insert("end", LRcoef)
        text_box_lin_reg.insert("end", "\nIntercept     = ")
        text_box_lin_reg.insert("end", LRintercept)
        text_box_lin_reg.insert("end", "\nMean Absolute Error  =  ")
        text_box_lin_reg.insert("end", LRmae)
        text_box_lin_reg.insert("end", "\nMean Squared Error   =  ")
        text_box_lin_reg.insert("end", LRmse)
        text_box_lin_reg.insert("end", "\nR2 Score   =   ")
        text_box_lin_reg.insert("end", LRr2)
        text_box_lin_reg.configure(state="disabled")
    except Exception as e:
        messagebox.showinfo("Error",e)
def show_box_Log_Reg():
    """show text box in ML Tab for Logistic Regression results"""
    try:
        message="Successfully Processed"
        text_box_log_reg.configure(state="normal")
        text_box_log_reg.delete("1.0", "end")
        text_box_log_reg.insert("end", message)
        text_box_log_reg.insert("end", "\nAccuracy   =  ")
        text_box_log_reg.insert("end", log_acc)
        text_box_log_reg.insert("end", "\nClassification Report =\n")
        text_box_log_reg.insert("end", log_class_report)
        text_box_log_reg.insert("end", "\ny_pred =\n")
        text_box_log_reg.insert("end", y_pred_log)
        text_box_log_reg.insert("end", "\ny_probability_log  =  ")
        text_box_log_reg.insert("end", y_proba_log)
        text_box_log_reg.configure(state="disabled")
    except Exception as e:
        messagebox.showinfo("Error",e)
def show_box_rf_clf():
    """show text box in ML Tab for Random Forest Classifier results"""
    try:
        message="Successfully Processed"
        text_box_rfc.configure(state="normal")
        text_box_rfc.delete("1.0", "end")
        text_box_rfc.insert("end", message)
        text_box_rfc.insert("end", "\nFeature Importance  =  ")
        text_box_rfc.insert("end", rf_clf_importance)
        text_box_rfc.insert("end", "\nAccuracy   =  ")
        text_box_rfc.insert("end", rf_clf_acc)
        text_box_rfc.configure(state="disabled")
    except Exception as e:
        messagebox.showinfo("Error",e)
def show_box_knn():
    """show text box in ML Tab for KNN results"""
    try:
        message="Successfully Processed"
        text_box_knn.configure(state="normal")
        text_box_knn.delete("1.0", "end")
        text_box_knn.insert("end", message)
        text_box_knn.insert("end", "\nAccuracy   =  ")
        text_box_knn.insert("end", knn_acc)
        text_box_knn.insert("end", "\ny_test =  ")
        text_box_knn.insert("end", y_test)
        text_box_knn.insert("end", "\ny_pred =\n")
        text_box_knn.insert("end", y_pred_knn)
        text_box_knn.configure(state="disabled")
    except Exception as e:
        messagebox.showinfo("Error",e)
def show_box_svm():
    """show text box in ML Tab for KNN results"""
    try:
        message="Successfully Processed"
        text_box_svm.configure(state="normal")
        text_box_svm.delete("1.0", "end")
        text_box_svm.insert("end", message)
        text_box_svm.insert("end", "\nAccuracy   =  ")
        text_box_svm.insert("end", svm_acc)
        text_box_svm.insert("end", "\ny_test =  ")
        text_box_svm.insert("end", y_test)
        text_box_svm.insert("end", "\ny_pred =\n")
        text_box_svm.insert("end", y_pred_svm)
        text_box_svm.configure(state="disabled")
    except Exception as e:
        messagebox.showinfo("Error",e)

#-----------Machine Learning Functions ----------------------
def Linear_Reg_func():
    """Linear Regression Model for Machine Learning"""
    cls_textbox(text_box_lin_reg,tab5_frame2)
    global LRcoef, LRintercept,y_pred,LRmae,LRmse,LRr2
    try:
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        LRcoef = model.coef_
        LRintercept = model.intercept_
        LRmae = mean_absolute_error(y_test, y_pred)
        LRmse = mean_squared_error(y_test, y_pred)
        LRr2  = model.score(X_test, y_test)
        show_box_Lin_Reg()
        lin_reg_plot()
    except Exception as e:
        messagebox.showinfo("Error",e)
def lin_reg_plot():
    """Generates Linear Regression model prediction on the given axes."""
    global ax, canvas_widget, toolbar
    try:
        fig = Figure(figsize=(5, 3), dpi=100)
        ax = fig.add_subplot(111)
        ax.scatter(y_test, y_pred, color = "red")
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], color='blue', linestyle='--', label='Regression Line')
        ax.set_title("Prediction Plot")
        ax.grid(True)
        #------------------Matplotlib--------------------------------
        canvas_widget = FigureCanvasTkAgg(fig, master=tab5_frame2)# 4. Embed the Figure into a Tkinter Canvas Widget
        canvas_widget.draw()# 5. Draw the canvas and place the widget in the GUI using 'pack'
        canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(canvas_widget, tab5_frame2)# 6. (Optional) Add a standard navigation toolbar
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
    except Exception as e:
        messagebox.showinfo("Error",e)

def Logistic_Reg_func():
    """Logistic Regression Model for Machine Learning"""
    cls_textbox(text_box_log_reg,tab6_frame2)
    global y_pred_log,y_proba_log,log_acc,log_model,log_class_report
    try:
        maxx_iteration = int(Log_Reg_max_iter_entry.get() or "200")
        log_model=LogisticRegression(max_iter=maxx_iteration)
        log_model.fit(X_train,y_train)
        y_pred_log = log_model.predict(X_test) # the predict() method uses the default 0.5 threshold
        y_proba_log = log_model.predict_proba(X_test)
        log_acc = accuracy_score(y_test, y_pred_log)
        log_class_report = classification_report(y_test, y_pred_log,zero_division=1)
        show_box_Log_Reg()
        confusion_matrix()
    except Exception as e:
        messagebox.showinfo("Error",e)
def confusion_matrix():
    try:
        fig = Figure(figsize=(5, 3), dpi=100)
        ax = fig.add_subplot(121)
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred_log, ax=ax,cmap='Blues')
        ax.set_title("Confusion Matrix")
        ax.grid(True)
        ax = fig.add_subplot(122)
        RocCurveDisplay.from_estimator(log_model, X_test, y_test, ax=ax)
        ax.set_title("RoC Curve")
        ax.grid(True)
        plt.show()
        #------------------Matplotlib--------------------------------
        canvas_widget = FigureCanvasTkAgg(fig, master=tab6_frame2)# 4. Embed the Figure into a Tkinter Canvas Widget
        canvas_widget.draw()# 5. Draw the canvas and place the widget in the GUI using 'pack'
        canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(canvas_widget, tab6_frame2)# 6. (Optional) Add a standard navigation toolbar
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
    except Exception as e:
        messagebox.showinfo("Error",e)

def random_forest_classifier():
    """Random Forest Classifier Model for Machine Learning"""
    cls_textbox(text_box_rfc,tab7_frame2)
    global rf_clf_acc, rf_clf_importance
    try:
        no_of_estim = int(rfc_n_estimators_entry.get() or "100")
        rf_clf = RandomForestClassifier(n_estimators=no_of_estim,random_state=42)
        rf_clf.fit(X_train, y_train)
        y_pred_rf_clf = rf_clf.predict(X_test)
        rf_clf_acc = accuracy_score(y_test, y_pred_rf_clf)
        rf_clf_importance = rf_clf.feature_importances_
        show_box_rf_clf()
    except Exception as e:
        messagebox.showinfo("Error",e)

def knn():
    """KNN Model for Machine Learning"""
    cls_textbox(text_box_knn,tab8_frame2)
    global y_pred_knn, knn_acc
    try:
        no_of_neighbors = int(knn_n_neighbors_entry.get() or "3")
        knn_model = KNeighborsClassifier(n_neighbors=no_of_neighbors)
        knn_model.fit(X_train, y_train)
        y_pred_knn = knn_model.predict(X_test)
        knn_acc = accuracy_score(y_test, y_pred_knn)
        show_box_knn()
    except Exception as e:
        messagebox.showinfo("Error",e)

def svm():
    """SVM Model for Machine Learning"""
    global y_pred_svm, svm_acc
    cls_textbox(text_box_svm,tab9_frame2)
    try:
        svm_prob = svm_probability_radio_var.get()
        svm_kernelVar = svm_kernel_entry.get() or "rbf"
        svm_random_state= int(svm_random_state_entry.get() or "42")
        svm_model = SVC(kernel=svm_kernelVar, probability=svm_prob, random_state=svm_random_state)
        svm_model.fit(X_train, y_train)
        y_pred_svm = svm_model.predict(X_test)
        svm_acc = accuracy_score(y_test, y_pred_svm)
        show_box_svm()
    except Exception as e:
        messagebox.showinfo("Error",e)

#------------------EDA-Tab-Buttons----------------------------
file_open_btn=CTkButton(tab1_frame1, text="Open File", fg_color ='green', command=reopen_file)
file_open_btn.pack(padx='5',pady='2')
dfhead_btn=CTkButton(tab1_frame1, text="Head", command=dfhead_func)
dfhead_btn.pack(padx='5',pady='2')
dfshape_btn=CTkButton(tab1_frame1, text="Shape", command=dfshape_func)
dfshape_btn.pack(padx='5',pady='2')
dfinfo_btn=CTkButton(tab1_frame1, text="Info", command=dfinfo_func)
dfinfo_btn.pack(padx='5',pady='2')
dfdescribe_btn=CTkButton(tab1_frame1, text="Describe", command=dfdescribe_func)
dfdescribe_btn.pack(padx='5',pady='2')
dfdtypes_btn=CTkButton(tab1_frame1, text="Data Types", command=dtypes_func)
dfdtypes_btn.pack(padx='5',pady='2')
#--------------------corr-Button-------------------------------
dfcorr_btn=CTkButton(tab1_frame1, text="Correlation", command=correlation_func)
dfcorr_btn.pack(padx='5',pady='2')
duplicate_btn=CTkButton(tab1_frame1, text="Duplicates", command=duplicates_func)
duplicate_btn.pack(padx='5',pady='2')
rem_duplicate_btn=CTkButton(tab1_frame1, text="Remove Duplicates", command=rem_duplicates_func)
rem_duplicate_btn.pack(padx='5',pady='2')
#-------------------------Null-Values---------------------------
dfnull_btn=CTkButton(tab1_frame1, text="Show Null Values", command=dfisnull_func)
dfnull_btn.pack(padx='5',pady='2')
#------------------------Fill Nll Values ------------------------
dffill_label=CTkLabel(tab1_frame1,text="Deal Null Values",font=("Arial", 20, "bold"))
dffill_label.pack()
col_fill_label=CTkLabel(tab1_frame1,text="Select Column")
col_fill_label.pack()
fill_col_option = tk.StringVar()
fill_null_combo_box = ttk.Combobox(tab1_frame1, textvariable=fill_col_option, values=col_options, state="readonly")
fill_col_option.set(col_options[0]) # Set default value
fill_null_combo_box.pack(pady=5)
drop_col_btn=CTkButton(tab1_frame1, text="Drop Column", command=drop_col_func)
drop_col_btn.pack(padx='5',pady='2')
dfdropna_btn=CTkButton(tab1_frame1, text="Drop Null Values", command=dfdropna_func)
dfdropna_btn.pack(padx='5',pady='2')
# ------------------------fill with-options------------------
selected_fill_option = tk.StringVar()
fill_options = ["Mean", "Mode", "Median","other"]
tab1_combo_box = ttk.Combobox(tab1_frame1, textvariable=selected_fill_option, values=fill_options, state="readonly")
selected_fill_option.set(fill_options[0]) # Set default value
tab1_combo_box.pack(pady=5)
dffillna_btn=CTkButton(tab1_frame1, text="Fill Null Values", command=dffillna_func)
dffillna_btn.pack(padx='5',pady='2')
df_values_cnt_btn=CTkButton(tab1_frame1, text="Unique Values", command=unique_values_func)
df_values_cnt_btn.pack(padx='5',pady='2')
dfunique_btn=CTkButton(tab1_frame1, text="Values Count", command=value_counts_func)
dfunique_btn.pack(padx='5',pady='2')
#-----------------Remove Outlier-----------------------------------
rm_outlier_label=CTkLabel(tab1_frame1,text="Remove Outliers",font=("Arial", 20, "bold"))
rm_outlier_label.pack()
col_rm_outlier_label=CTkLabel(tab1_frame1,text="Select Column")
col_rm_outlier_label.pack()
rm_outlier_combo_box_colvar = tk.StringVar()
rm_outlier_combo_box = ttk.Combobox(tab1_frame1, textvariable=rm_outlier_combo_box_colvar, values=col_options, state="readonly")
rm_outlier_combo_box_colvar.set(col_options[0]) # Set default value
rm_outlier_combo_box.pack(padx=5, pady=2)
min_quantile_option =CTkEntry(tab1_frame1, placeholder_text="min.quantile value")
min_quantile_option.pack(padx='5',pady='2')
max_quantile_option =CTkEntry(tab1_frame1, placeholder_text="max.quantile value")
max_quantile_option.pack(padx='5',pady='2')
rm_outlier_btn=CTkButton(tab1_frame1, text="Remove Outlier", command=remove_outlier_func)
rm_outlier_btn.pack(padx='5',pady='2')
#------------------------Advance options-----------------------------
custom_cmd_btn=CTkButton(tab1_frame1, text="Apply Custom Code",fg_color="firebrick", command=custom_cmd_func)
custom_cmd_btn.pack(padx='5',pady='2')
save_file_btn=CTkButton(tab1_frame1, text="Save File",fg_color="green", command=to_csv_func)
save_file_btn.pack(padx='5',pady='2')
reset_df_btn=CTkButton(tab1_frame1, text="Reset df",fg_color="cyan",text_color="red", command=reset_df_func)
reset_df_btn.pack(padx='5',pady='2')

#-----------------plotting tab design----------------
tab2_label_x=CTkLabel(tab2_frame1, text='x =')
tab2_label_x.pack(side='left',padx=10, pady=5)
#------------------x = ------------------------------
tab2_combo_box_xvar = tk.StringVar()
tab2_combo_box_x = ttk.Combobox(tab2_frame1, textvariable=tab2_combo_box_xvar, values=plot_col_options, state="readonly")
tab2_combo_box_xvar.set(plot_col_options[0]) # Set default value
tab2_combo_box_x.pack(side='left',padx=10, pady=5)
#------------------y=------------------------------
tab2_combo_box_yvar = tk.StringVar()
tab2_label_y=CTkLabel(tab2_frame1, text='y =')
tab2_label_y.pack(side='left',padx=10, pady=5)
tab2_combo_box_y = ttk.Combobox(tab2_frame1, textvariable=tab2_combo_box_yvar, values=plot_col_options, state="readonly")
tab2_combo_box_yvar.set(plot_col_options[0]) # Set default value
tab2_combo_box_y.pack(side='left',padx=10, pady=5)
#------------------hue------------------------------
tab2_combo_box_huevar = tk.StringVar()
tab2_label_hue=CTkLabel(tab2_frame1, text='hue =')
tab2_label_hue.pack(side='left',padx=10, pady=5)
tab2_combo_box_hue = ttk.Combobox(tab2_frame1, textvariable=tab2_combo_box_huevar, values=plot_col_options, state="readonly")
tab2_combo_box_huevar.set(plot_col_options[0]) # Set default value
tab2_combo_box_hue.pack(side='left',padx=10, pady=5)
#------------------color------------------------------
plot_colors = ['Red','Green','Blue','Cyan','Magenta','Yellow','Black','White']
tab2_combo_box_clrvar = tk.StringVar()
tab2_label_clr=CTkLabel(tab2_frame1, text='Color =')
tab2_label_clr.pack(side='left',padx=10, pady=5)
tab2_combo_box_clr = ttk.Combobox(tab2_frame1, textvariable=tab2_combo_box_clrvar, values=plot_colors, state="readonly")
tab2_combo_box_clr.current(0) # Set default value
tab2_combo_box_clr.pack(side='left',padx=10, pady=5)
#-------------------cmap color------------------------
cmap_colors = ['viridis','coolwarm','plasma','inferno','magma','cividis','Blues', 'Reds', 'RdBu', 'Spectral']
tab2_combo_box_cmapclrvar = tk.StringVar()
tab2_label_cmapclr=CTkLabel(tab2_frame1, text='Cmap color =')
tab2_label_cmapclr.pack(side='left',padx=10, pady=5)
tab2_combo_box_cmapclr = ttk.Combobox(tab2_frame1, textvariable=tab2_combo_box_cmapclrvar, values=cmap_colors, state="readonly")
tab2_combo_box_cmapclr.current(0) # Set default value
tab2_combo_box_cmapclr.pack(side='left',padx=10, pady=5)
# tab2_canvas.create_line(0, 0, 300, 200, fill="blue", width=2)
#------------------Plot Button Design-----------------------------------------------------------------
plot_btn=CTkButton(tab2_frame2, text="Line Plot(x)", hover_color="darkblue",command=lambda: create_plot())
plot_btn.pack(side='left',padx='5',pady='2')
barplot_btn=CTkButton(tab2_frame2, text="Bar Plot(x,y)",hover_color="darkblue", command=lambda: create_barplot())
barplot_btn.pack(side='left',padx='5',pady='2')
scatterplot_btn=CTkButton(tab2_frame2, text="Scatter Plot(x,y)", hover_color="darkblue",command=lambda: create_scatterplot())
scatterplot_btn.pack(side='left',padx='5',pady='2')
histplot_btn=CTkButton(tab2_frame2, text="Hist Plot(x)", hover_color="darkblue",command=lambda: create_histplot())
histplot_btn.pack(side='left',padx='5',pady='2')
heatmap_btn=CTkButton(tab2_frame2, text="Heatmap", hover_color="darkblue",command=lambda: create_heatmap())
heatmap_btn.pack(side='left',padx='5',pady='2')
pairplot_btn=CTkButton(tab2_frame2, text="Pairplot", hover_color="darkblue",command=lambda: create_pairplot())
pairplot_btn.pack(side='left',padx='5',pady='2')
boxplot_btn=CTkButton(tab2_frame2, text="Boxplot(x)", hover_color="darkblue",command=lambda: create_boxplot())
boxplot_btn.pack(side='left',padx='5',pady='2')
piechart_btn=CTkButton(tab2_frame2, text="PieChart(x)", hover_color="darkblue",command=lambda: piechart())
piechart_btn.pack(side='left',padx='5',pady='2')
clear_btn=CTkButton(tab2_frame2, text="Clear", fg_color="green", hover_color="darkblue",command=clear_plot)
clear_btn.pack(side='left',padx='5',pady='2')

#------------Encoding Column Selection ---------------------------------
get_encode_btn=CTkButton(tab3_frame1, text="Get Columns", hover_color="red",command=get_encode_columns)
get_encode_btn.pack(padx='5',pady='2')
ohe_encode_btn=CTkButton(tab3_frame1, text="One-Hot Encoder", hover_color="red",command=one_hot_encoder)
ohe_encode_btn.pack(padx='5',pady='2')
le_encode_btn=CTkButton(tab3_frame1, text="Label Encoder", hover_color="red",command=label_encoder)
le_encode_btn.pack(padx='5',pady='2')
get_dummies_btn=CTkButton(tab3_frame1, text="Get Dummies", hover_color="red",command=get_dummies)
get_dummies_btn.pack(padx='5',pady='2')

#----------X-Feature------------
select_feature_btn=CTkButton(tab4_frame1, text="Get Features", hover_color="green",command=lambda:get_feature())
select_feature_btn.pack(side='left',padx='5',pady='2')
train_btn=CTkButton(tab4_frame1, text="Train Features", hover_color="green",command=lambda: train_features())
train_btn.pack(side='left',padx='5',pady='2')
stand_scaler_btn=CTkButton(tab4_frame1, text="Standard Scaler", hover_color="red",command=lambda:standardscaler())
stand_scaler_btn.pack(side='left',padx='5',pady='2')

#------------------------ Get Paramters----------------------------
Lin_Reg_pms_label = CTkLabel(tab5_frame0, text = "Parameters: ", font = ("Consolas",14))
Lin_Reg_pms_label.pack(side = 'left', padx=2, pady =5)
Log_Reg_pms_label = CTkLabel(tab6_frame0, text = "Parameters: ", font = ("Consolas",14))
Log_Reg_pms_label.pack(side = 'left', padx=2, pady =5)
Log_Reg_max_iter_label = CTkLabel(tab6_frame0, text = "max_iter=")
Log_Reg_max_iter_label.pack(side = 'left', padx=2, pady =5)
Log_Reg_max_iter_entry = CTkEntry(tab6_frame0, placeholder_text= "200")
Log_Reg_max_iter_entry.pack(side = 'left', padx=2, pady =5)
rfc_n_estimators_label = CTkLabel(tab7_frame0, text = "rfc_n_estimators=")
rfc_n_estimators_label.pack(side = 'left', padx=2, pady =5)
rfc_n_estimators_entry = CTkEntry(tab7_frame0, placeholder_text= "100")
rfc_n_estimators_entry.pack(side = 'left', padx=2, pady =5)
knn_n_neighbors_label = CTkLabel(tab8_frame0, text = "KNN_n_neighbors=")
knn_n_neighbors_label.pack(side = 'left', padx=2, pady =5)
knn_n_neighbors_entry = CTkEntry(tab8_frame0, placeholder_text= "3")
knn_n_neighbors_entry.pack(side = 'left', padx=2, pady =5)
svm_random_state_label = CTkLabel(tab9_frame0, text = "Random_state=")
svm_random_state_label.pack(side = 'left',padx=2, pady =5)
svm_random_state_entry = CTkEntry(tab9_frame0, placeholder_text= "42")
svm_random_state_entry.pack(side = 'left',padx=5, pady =5)
svm_kernel_label = CTkLabel(tab9_frame0, text = "SVM_kernel=")
svm_kernel_label.pack(side = 'left', padx=2, pady =5)
svm_kernel_entry = CTkEntry(tab9_frame0, placeholder_text= "rbf")
svm_kernel_entry.pack(side = 'left', padx=2, pady =5)
svm_probability_label = CTkLabel(tab9_frame0, text = "SVM_probability=")
svm_probability_label.pack(side = 'left', padx=2, pady =5)
svm_probability_radio_var = ctk.BooleanVar(value=True)
svm_probability_radio_1 = ctk.CTkRadioButton(tab9_frame0, text="True", variable=svm_probability_radio_var, value=True)
svm_probability_radio_2= ctk.CTkRadioButton(tab9_frame0, text="False", variable=svm_probability_radio_var, value=False)
svm_probability_radio_1.pack(side ='left',pady=2, padx=5)
svm_probability_radio_2.pack(side ='left',pady=2, padx=5)
#-------------------------MAchine Learning Buttons--------------------------------
Lin_Reg_btn=CTkButton(tab5_frame0, text="Linear Regression", command=lambda:Linear_Reg_func())
Lin_Reg_btn.pack(side ='left',padx='5',pady='2')
Log_Reg_btn=CTkButton(tab6_frame0, text="Logistic Regression", command=lambda:Logistic_Reg_func())
Log_Reg_btn.pack(side ='left',padx='5',pady='2')
rfc_btn=CTkButton(tab7_frame0, text="Random Forest", command=lambda:random_forest_classifier())
rfc_btn.pack(side ='left',padx='5',pady='2')
knn_btn=CTkButton(tab8_frame0, text="KNN", command=lambda:knn())
knn_btn.pack(side ='left',padx='5',pady='2')
svm_btn=CTkButton(tab9_frame0, text="SVM", command=lambda:svm())
svm_btn.pack(side ='left',padx='5',pady='2')
lin_reg_cls_btn=CTkButton(tab5_frame0, text="Clear Screen", command=lambda:cls_textbox(text_box_lin_reg,tab5_frame2),hover_color="Green")
lin_reg_cls_btn.pack(side ='left',padx='5',pady='2')
log_reg_cls_btn=CTkButton(tab6_frame0, text="Clear Screen", command=lambda:cls_textbox(text_box_log_reg,tab6_frame2),hover_color="Green")
log_reg_cls_btn.pack(side ='left',padx='5',pady='2')
rfc_cls_btn=CTkButton(tab7_frame0, text="Clear Screen", command=lambda:cls_textbox(text_box_rfc,tab7_frame2),hover_color="Green")
rfc_cls_btn.pack(side ='left',padx='5',pady='2')
knn_cls_btn=CTkButton(tab8_frame0, text="Clear Screen", command=lambda:cls_textbox(text_box_knn,tab8_frame2),hover_color="Green")
knn_cls_btn.pack(side ='left',padx='5',pady='2')
svm_cls_btn=CTkButton(tab9_frame0, text="Clear Screen", command=lambda:cls_textbox(text_box_svm,tab9_frame2),hover_color="Green")
svm_cls_btn.pack(side ='left',padx='5',pady='2')
ml_compare_btn=CTkButton(tab10_frame0, text="Compare", command=lambda:ml_compare(),hover_color="Green")
ml_compare_btn.pack(side ='left',padx='5',pady='2')
#------------------TEXT BOX- Machine Learning---------
#------------------TEXT BOX- Machine Learning----------------------------------
text_box_lin_reg= CTkTextbox(tab5_frame1, font=("Consolas", 20))
text_box_lin_reg.pack(fill="both", expand=True, padx=5, pady=5)
text_box_log_reg = CTkTextbox(tab6_frame1, font=("Consolas", 20))
text_box_log_reg.pack(fill="both", expand=True, padx=5, pady=5)
text_box_rfc = CTkTextbox(tab7_frame1, font=("Consolas", 20))
text_box_rfc.pack(fill="both", expand=True, padx=5, pady=5)
text_box_knn = CTkTextbox(tab8_frame1, font=("Consolas", 20))
text_box_knn.pack(fill="both", expand=True, padx=5, pady=5)
text_box_svm = CTkTextbox(tab9_frame1, font=("Consolas", 20))
text_box_svm.pack(fill="both", expand=True, padx=5, pady=5)

def ml_compare():
    """Generate histplot to compare ML algorithms."""
    clear_frame(tab10_frame1)
    try:
        compare_data = [LRr2, log_acc, rf_clf_acc, knn_acc, svm_acc]
        labels = ['Linear Reg', 'Log Reg', 'Random Forest', 'KNN', 'SVM']
        # 2. Error Handling: Filter out missing values (None or undefined)
        # This creates a clean pair of (label, accuracy) for only present data
        filtered_data = []
        filtered_labels = []
        for label, acc in zip(labels, compare_data):
            if acc is not None:
                filtered_data.append(acc)
                filtered_labels.append(label)
        if not filtered_data:
            messagebox.showinfo("Error","No accuracy data available to plot.")
        else:
            fig = Figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)
            sns.set_theme(style="whitegrid")
            sns.barplot(ax=ax, x=filtered_labels, y=filtered_data, color='red')
            ax.set_title('Machine Learning Algorithm Accuracy Comparison', fontsize=15)
            ax.set_ylabel('Accuracy Score', fontsize=12)
            ax.set_ylim(0, 1.0) # Standardize y-axis for accuracy (0 to 1)
            #Annotate bars with their values
            for p in ax.patches:
                ax.annotate(format(p.get_height(), '.2f'),
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha = 'center', va = 'center',
                            xytext = (0, 9),
                            textcoords = 'offset points')
            plt.show()
            # ------------------Matplotlib--------------------------------
            canvas_widget = FigureCanvasTkAgg(fig, master=tab10_frame1)# 4. Embed the Figure into a Tkinter Canvas Widget
            canvas_widget.draw()# 5. Draw the canvas and place the widget in the GUI using 'pack'
            canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            toolbar = NavigationToolbar2Tk(canvas_widget, tab10_frame1)# 6. (Optional) Add a standard navigation toolbar
            toolbar.update()
            toolbar.pack(side=tk.BOTTOM, fill=tk.X)
    except NameError as e:
        messagebox.showinfo("Name Error:",f"One of the variables in compare_data is not defined: {e}")
    except Exception as e:
        messagebox.showinfo("Exception Error:",f"An unexpected error occurred: {e}")


# ------------------------------

root.mainloop()
