import sys
import traceback

try:
    import subprocess

    # --- AUTOMATIC PACKAGE INSTALLER ---
    def check_and_install(package, import_name=None):
        import_name = import_name or package
        try:
            __import__(import_name)
        except ImportError:
            print(f"\n[INFO] The '{package}' library is missing. Installing automatically via pip...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    print("[INFO] Checking system dependencies...")
    check_and_install('pandas')
    check_and_install('numpy')
    check_and_install('matplotlib', 'matplotlib')
    check_and_install('seaborn')
    check_and_install('scikit-learn', 'sklearn')
    check_and_install('scikit-bio', 'skbio')
    check_and_install('shap')

    # --- STANDARD IMPORTS ---
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    from skbio.stats.composition import clr
    import shap

    # 1. File Input
    print("\nHINT: Ensure you type the file name WITH the extension (e.g., my_data.csv or my_data.txt)\n")
    path = input("Enter the file name or path: ").strip().replace("'", "").replace('"', "")

    # --- UNIVERSAL FILE READER ---
    print("\n[INFO] Reading file...")
    try:
        df = pd.read_csv(path, sep=';', decimal=',')
        if df.shape[1] < 2:
            df = pd.read_csv(path, sep='\t', decimal=',')
        if df.shape[1] < 2:
            df = pd.read_csv(path, sep=',', decimal='.')
        if df.shape[1] < 2:
            raise ValueError("Data separator not recognized. Ensure it is a valid CSV or TXT.")
    except Exception as e:
        raise ValueError(f"Could not read the file. Check if the name is correct. Detail: {e}")

    # 2. Data Preparation
    df = df.dropna(axis=1, how='all')
    df.columns = df.columns.str.strip()

    for col in df.columns:
        if df[col].dtype == object:
             df[col] = df[col].astype(str).str.replace(',', '.')

    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(how='any')

    print(f"[INFO] Valid samples (rows) found for analysis: {len(df)}")

    X_raw = df.iloc[:, :-1].copy()
    Y_raw = df.iloc[:, -1:].copy()
    Y_name = str(Y_raw.columns[0]).strip()

    if Y_name.startswith('Unnamed') or Y_name == '':
        Y_name = "Target_Variable"
        Y_raw.columns = [Y_name]

    print(f"\n[INFO] Target variable detected: '{Y_name}'")

    # --- SUB-COMPOSITIONAL CLR TRANSFORMATION ---
    vessel_cols = [c for c in X_raw.columns if c in ['SVP_%', 'MVP_%']]
    tissue_cols = [c for c in X_raw.columns if c in ['VLuP_%', 'VWP_%', 'FLuP_%', 'FWP_%', 'RP_%', 'APP_%']]

    if vessel_cols:
        print(f"[INFO] Applying CLR on Vessel sub-composition.")
        matriz_vessels = X_raw[vessel_cols].replace(0, 1e-6)
        X_raw[vessel_cols] = clr(matriz_vessels)

    if tissue_cols:
        print(f"[INFO] Applying CLR on Tissue sub-composition.")
        matriz_tissues = X_raw[tissue_cols].replace(0, 1e-6)
        X_raw[tissue_cols] = clr(matriz_tissues)

    # --- CONTROLLED BASELINE REMOVAL ---
    cols_to_drop = [c for c in ['SVP_%', 'VWP_%'] if c in X_raw.columns]
    if cols_to_drop:
        X_raw = X_raw.drop(columns=cols_to_drop)
        print(f"[INFO] Intentionally removed {cols_to_drop} AFTER CLR to act as statistical baselines.")

    # --- STANDARDIZATION ---
    scaler = StandardScaler()
    X_sc_df = pd.DataFrame(scaler.fit_transform(X_raw), columns=X_raw.columns)
    Y_sc = scaler.fit_transform(Y_raw).flatten()

    # --- 3. EXTRA-TREES REGRESSION (COM LEAVE-ONE-OUT CROSS-VALIDATION) ---
    print("\n[INFO] Training Extra-Trees Regressor with Leave-One-Out Cross-Validation...")
    et_model = ExtraTreesRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    
    # Realização da Validação Cruzada (A prova de fogo preditiva)
    loo = LeaveOneOut()
    Y_pred_cv = cross_val_predict(et_model, X_sc_df, Y_sc, cv=loo)
    
    # Cálculo das métricas REAIS
    et_r2_cv = r2_score(Y_sc, Y_pred_cv)
    et_mse_cv = mean_squared_error(Y_sc, Y_pred_cv)

    # Treinamento final na base completa para a geração do SHAP
    et_model.fit(X_sc_df, Y_sc)

    # Formatação das métricas para publicação
    r2_str = f"{et_r2_cv:.4f}"
    mse_str = f"{et_mse_cv:.2e}" if et_mse_cv < 0.001 else f"{et_mse_cv:.4f}"

    print(f"[RESULT] Realistic Extra-Trees R² (LOOCV): {r2_str}")
    print(f"[RESULT] Realistic MSE (LOOCV): {mse_str}")

    # --- 4. GENERATE TEXT RESULTS REPORT ---
    results_file_name = f"results_ET_{Y_name}.txt"
    with open(results_file_name, "w", encoding="utf-8") as f:
        f.write(f"=== EXTRA-TREES INFLUENCE ANALYSIS: {Y_name} ===\n\n")
        f.write(f"Dependent Variable (Response): {Y_name}\n")
        f.write(f"VIF Filtering: NOT APPLIED (Preserving natural biological collinearity)\n")
        f.write(f"Validation Method: Leave-One-Out Cross-Validation (LOOCV)\n")
        f.write(f"Variables used in the model: {', '.join(X_sc_df.columns)}\n")
        
        f.write(f"\n--- EXTRA-TREES GLOBAL PERFORMANCE (LOOCV) ---\n")
        f.write(f"Number of Trees (Estimators): 500\n")
        f.write(f"Predictive R²: {r2_str}\n")
        f.write(f"Mean Squared Error (MSE): {mse_str}\n\n")

    # --- 5. SHAP ANALYSIS & VISUALIZATION ---
    print("\n[INFO] Calculating SHAP values for Feature Importance...")

    explainer = shap.TreeExplainer(et_model)
    shap_values = explainer.shap_values(X_sc_df)

    # Plot 1: SHAP Summary Plot (Dots)
    plt.figure(figsize=(10, 8))
    plt.style.use('default')
    shap.summary_plot(shap_values, X_sc_df, show=False, plot_size=(10, 8))
    
    titulo_shap = f"SHAP Summary: Drivers of {Y_name} (Extra-Trees)\n[LOOCV R² = {et_r2_cv:.2f} | MSE = {mse_str}]"
    plt.title(titulo_shap, fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    shap_graph_name = f"shap_summary_ET_{Y_name}.png"
    plt.savefig(shap_graph_name, dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Global Feature Importance (Bar Plot)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sc_df, plot_type="bar", show=False)
    
    titulo_bar = f"Global Feature Importance for {Y_name}\n[LOOCV R² = {et_r2_cv:.2f} | MSE = {mse_str}]"
    plt.title(titulo_bar, fontsize=14, fontweight='bold', pad=20)
    
    plt.xlabel("mean(|SHAP value|) - Average impact on model output magnitude")
    plt.tight_layout()
    bar_graph_name = f"shap_bar_ET_{Y_name}.png"
    plt.savefig(bar_graph_name, dpi=300, bbox_inches='tight')
    plt.close()

    # --- 6. DETAILED SHAP EXPORT (CSV for Excel) ---
    df_shap_export = pd.DataFrame(shap_values, columns=X_sc_df.columns)
    df_shap_export.insert(0, 'Row_Index_Species', range(1, len(df_shap_export) + 1))
    export_name = f"shap_values_table_ET_{Y_name}.csv"
    df_shap_export.to_csv(export_name, index=False, sep=';', decimal=',')

    print(f"\n[SUCCESS] Analysis completed successfully for {Y_name}!")
    print(f"  -> Global text report saved: {results_file_name}")
    print(f"  -> SHAP dot plot saved: {shap_graph_name}")
    print(f"  -> SHAP bar plot saved: {bar_graph_name}")
    print(f"  -> Detailed SHAP values for Excel saved: {export_name}")

    input("\nPress ENTER to close the program...")

# --- GLOBAL ERROR TRAP ---
except Exception as e:
    print("\n" + "="*50)
    print(" 🚨 CRITICAL ERROR ENCOUNTERED 🚨 ")
    print("="*50)
    traceback.print_exc()
    print("="*50)
    input("\nThe program paused. Take a photo or copy the text above, then press ENTER to exit...")